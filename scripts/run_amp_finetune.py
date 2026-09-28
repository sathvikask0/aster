"""Unfreeze the last ESM-2 blocks and train them inside the AMP benchmark.

Everything published from the AMP path so far trained over cached ESM-2 vectors,
so the protein encoder never learned anything about peptides. This run puts it in
the graph with its last blocks trainable, on the composition-matched benchmark
where the shortcut is at chance by construction -- so whatever a fine-tune buys
has to come from residue order, not bulk composition.

Read the output in this order.

1. `entity_only` also gets a live encoder. It never sees the question. If it
   moves as much as the hypothesis model, fine-tuning bought a better peptide
   classifier and nothing about typed decisions.
2. `dual_live` vs `dual_frozen` is the only pair that isolates unfreezing:
   same head, same data, same reading rule, one factor different.
3. `encoder_drift` is printed next to each accuracy. A gain with ~0 drift is a
   head effect wearing a fine-tuning label; a gain with a huge drift is usually
   a destroyed encoder and a compensating head.

`task_id` stays frozen and is reported as the lookup floor only. It has no
entity path worth unfreezing, so it is a reference row, not a matched control.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aster.real.amp import LABEL_SEMANTICS_VERSION, load_amp_benchmark
from aster.real.ceilings import binomial_ci95, composition_ceiling
from aster.real.embed import embed_sequences, embed_texts, pick_device
from aster.real import ablation
from aster.real.evaluate import (
    REPORTING_RULE, build_tensor_dict, evaluate, mechanism_swap, print_result,
)
from aster.real.finetune import (
    LIVE_TEXT_MODES, LiveAster, TextTable, encoder_drift, load_protein_encoder,
    load_text_encoder, param_groups, snapshot_encoder, tokenize_sequences,
    trainable_report, cache_frozen_prefix, pooled_suffix, snapshot_trainable_state,
)
from aster.real.models import CrossAttentionAster, RealAster
from aster.real.properties import (
    PROPERTIES, build_property_benchmark, label_agreement, load_property_table,
    task_overlap,
)

PROPERTY_FILES = {
    "hemolytic": Path("data/properties/hemolytic_crossval.csv"),
    "antimicrobial": Path("data/properties/antimicrobial_mic.csv"),
}
MIN_SAMPLES = 400
# Mean-pooling models only. A CLS-pooled encoder (bge, e5) would be mean-pooled
# here, which is not how it was trained, so its vectors would be quietly wrong.
TEXT_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # 6L, 384d
    "mpnet": "sentence-transformers/all-mpnet-base-v2",   # 12L, 768d
    "distilroberta": "sentence-transformers/all-distilroberta-v1",  # 6L, 768d
}


def batches(n, bs, shuffle=False, generator=None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def slice_batch(tensors, idx, device):
    out = {}
    for k, v in tensors.items():
        chunk = v[idx]
        out[k] = chunk.to(device) if chunk.device != torch.device(device) else chunk
    return out


def forward_all(model, tensors, bs, device):
    """Logits for a whole split, in eval mode, without holding the graph."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for idx in batches(tensors["y"].shape[0], bs):
            chunks.append(model(slice_batch(tensors, idx, device)).float().cpu())
    return torch.cat(chunks).numpy()


def train(model, tr, va, device, epochs, bs, head_lr, encoder_lr, text_lr=None,
          clip=1.0, seed=42, patience=2, label=""):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(param_groups(
        model, head_lr=head_lr, encoder_lr=encoder_lr, text_lr=text_lr))
    n = tr["y"].shape[0]
    steps = epochs * ((n + bs - 1) // bs) + 10
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps
    )

    best_loss, best_weights, stale = float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        t0, seen, running = time.time(), 0, 0.0
        for step, idx in enumerate(batches(n, bs, shuffle=True, generator=gen), 1):
            batch = slice_batch(tr, idx, device)
            loss = F.cross_entropy(model(batch), batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], clip
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            running += loss.item() * idx.numel()
            seen += idx.numel()
            if step % 100 == 0:
                print(f"    {label} epoch {epoch + 1}: {seen}/{n} rows "
                      f"| loss {running / seen:.4f} | {time.time() - t0:.0f}s", flush=True)

        va_logits = forward_all(model, va, bs, device)
        val_loss = F.cross_entropy(
            torch.from_numpy(va_logits), va["y"].cpu()
        ).item()
        print(f"    epoch {epoch + 1}/{epochs}  train {running / max(seen, 1):.4f} "
              f"| val {val_loss:.4f} | {time.time() - t0:.0f}s", flush=True)
        history.append({"epoch": epoch + 1, "train_loss": running / max(seen, 1),
                        "val_loss": val_loss, "seconds": time.time() - t0})

        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_weights = snapshot_trainable_state(model)
        else:
            stale += 1
            if stale >= patience:
                print(f"    early stop: val loss has not improved in {patience} epochs")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights, strict=False)
    model.eval()
    model.training_history = history
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esm", default="8M", choices=list(("8M", "35M", "150M", "650M")))
    p.add_argument("--trainable-blocks", type=int, default=2,
                   help="Last n ESM-2 blocks (plus the final norm) to unfreeze")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Smaller than the frozen path: the encoder is in the graph now")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--unfreeze-text", action="store_true",
                   help="Also put the text encoder in the graph. Run this AFTER a "
                        "protein-only run: if both towers move at once and the "
                        "number goes up, nothing says which one did it.")
    p.add_argument("--text-model", default="minilm",
                   help="minilm (6L/384d), mpnet (12L/768d), distilroberta "
                        "(6L/768d), or any mean-pooled HF sentence encoder id")
    p.add_argument("--text-trainable-blocks", type=int, default=1)
    p.add_argument("--text-lr", type=float, default=None,
                   help="Defaults to --encoder-lr")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--max-peptides", type=int, default=None)
    p.add_argument("--cache-frozen-prefix", action="store_true",
                   help="Cache frozen block outputs on CPU; recompute and train the suffix")
    p.add_argument("--benchmark", default="amp", choices=["amp", "properties"],
                   help="'amp' holds out pathogens with presumed negatives; "
                        "'properties' holds out a whole property and every "
                        "negative carries a measured concentration")
    p.add_argument("--held-out-property", default="hemolytic",
                   help="properties benchmark: which property to hold out; "
                        "'none' runs the in-distribution rung instead")
    p.add_argument("--property-threshold", action="append", default=[],
                   metavar="NAME=UM",
                   help="Override a property's active/inactive cutoff, e.g. "
                        "antimicrobial=10. Repeatable.")
    p.add_argument("--disjoint-sequences", action="store_true",
                   help="properties benchmark: keep a peptide out of two splits")
    p.add_argument("--negative-policy", default="matched",
                   choices=["random", "covered", "matched", "scrambled"])
    p.add_argument("--no-balance-tasks", action="store_true",
                   help="Leave the training label prior skewed (not recommended)")
    p.add_argument("--skip-frozen-reference", action="store_true",
                   help="Skip the cached-embedding runs. They are minutes on a warm "
                        "cache and they are the only thing that makes the fine-tuned "
                        "number attributable, so skipping is rarely worth it.")
    p.add_argument("--device", default="auto")
    p.add_argument("--mps-memory-fraction", type=float, default=0.75,
                   help="Fraction of recommended MPS working set allowed for allocations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="reports/amp_finetune_results.json")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Save best trainable weights and base-model provenance for each live model")
    args = p.parse_args()
    text_model = TEXT_MODELS.get(args.text_model, args.text_model)

    device = pick_device(args.device)
    if device == "mps":
        if not 0 < args.mps_memory_fraction <= 1:
            p.error("--mps-memory-fraction must be in (0, 1]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    balance = not args.no_balance_tasks
    text_note = (f"text: last {args.text_trainable_blocks} blocks trainable"
                 if args.unfreeze_text else "text: frozen")
    scope = (f"properties | held out: {args.held_out_property}"
             if args.benchmark == "properties"
             else f"AMP | policy={args.negative_policy}")
    print(f"{scope} | ESM-2 {args.esm} | last {args.trainable_blocks} blocks "
          f"trainable | {text_note} | balance={balance} | {device}")
    if args.benchmark == "properties":
        print("  Negatives carry a measured concentration, so a negative is a "
              "recorded result rather than a missing row.")
        print("  NOTE with one training task, task_id has a single id to learn "
              "and is a degenerate floor here, not a meaningful control.")
    if args.unfreeze_text:
        print("  NOTE both towers are live in this run. Compare it against a "
              "protein-only run, not against the frozen reference alone, or the "
              "gain cannot be attributed to either tower.")

    if args.benchmark == "properties":
        thresholds = {}
        for item in args.property_threshold:
            name, _, value = item.partition("=")
            if name not in PROPERTIES or not value:
                p.error(f"--property-threshold expects NAME=UM with NAME in "
                        f"{sorted(PROPERTIES)}; got {item!r}")
            thresholds[name] = float(value)
        tables = {}
        for name in sorted(PROPERTIES):
            path = PROPERTY_FILES[name]
            if not path.exists():
                p.error(f"{path} is missing. See data/properties/README.md; the "
                        "antimicrobial table is built by "
                        "scripts/fetch_property_data.py.")
            tables[name] = load_property_table(name, path)
        overlap = task_overlap(tables)
        agreement = label_agreement(tables, thresholds)
        print(f"\nSequences measured for every property: "
              f"{overlap['measured_for_all']:,}")
        for pair, stats in overlap["pairs"].items():
            print(f"  {pair}: {stats['shared']:,} shared "
                  f"({stats['fraction_of_smaller']:.1%} of the smaller table)")
        if agreement.get("n"):
            disagree = sum(v for k, v in agreement["joint"].items()
                           if len(set(k.split("|"))) > 1)
            print(f"  labels disagree on {disagree:,} of {agreement['n']:,} "
                  f"shared peptides ({disagree / agreement['n']:.1%}) -- the "
                  "tasks are distinct, which is the point of this benchmark")
        held = () if args.held_out_property.lower() in ("none", "") else (
            args.held_out_property,)
        if not held:
            print("  IN-DISTRIBUTION rung: nothing held out, every property "
                  "supplies train, val and test. This measures whether the model "
                  "can learn these properties at all; it is not a transfer test.")
        examples, meta = build_property_benchmark(
            tables, test_tasks=held, thresholds=thresholds,
            seed=args.seed, balance_tasks=balance,
            disjoint_sequences=args.disjoint_sequences,
        )
    else:
        examples, meta = load_amp_benchmark(
            min_samples=MIN_SAMPLES, seed=args.seed,
            negative_policy=args.negative_policy, balance_tasks=balance,
        )

    unique_seqs = sorted({e.sequence for e in examples})
    if args.max_peptides and len(unique_seqs) > args.max_peptides:
        rng = np.random.default_rng(42)
        keep = set(rng.choice(unique_seqs, size=args.max_peptides, replace=False))
        examples = [e for e in examples if e.sequence in keep]
        unique_seqs = sorted(keep)
        print(f"Subsampled to {len(unique_seqs):,} peptides ({len(examples):,} rows)")

    train_ex = [e for e in examples if e.split == "train"]
    val_ex = [e for e in examples if e.split == "val"]
    test_ex = [e for e in examples if e.split == "test"]
    held_out_tasks = sorted({e.task for e in test_ex})
    train_tasks = sorted({e.task for e in train_ex})
    task_to_id = {t: i + 1 for i, t in enumerate(train_tasks)}

    def prior(rows):
        return sum(e.label for e in rows) / max(len(rows), 1)

    print(f"Train {len(train_ex):,} / val {len(val_ex):,} / test {len(test_ex):,}  "
          f"| {len(train_tasks)} train tasks, {len(held_out_tasks)} held out")
    print(f"Label prior: train {prior(train_ex):.3f} | held-out {prior(test_ex):.3f}")

    # Text side: frozen throughout, exactly as on the cached path.
    questions = {t: meta[t]["question"] for t in meta}
    q_mat = embed_texts([questions[t] for t in meta], model=text_model, device=device)
    q_map = {t: q_mat[i] for i, t in enumerate(meta)}
    ans_texts = sorted({o for t in meta for o in meta[t]["options"][:2]})
    ans_mat = embed_texts(ans_texts, model=text_model, device=device)
    ans_map = dict(zip(ans_texts, ans_mat))
    A_map = {(t, i): ans_map[meta[t]["options"][i]] for t in meta for i in (0, 1)}
    d_text = q_mat.shape[1]

    # The ceiling is a property of the data, so it is measured once and both the
    # frozen and the fine-tuned runs are scored against the same numbers.
    print("\n--- Shortcut ceilings on the held-out tasks (out of fold) ---")
    ceilings = {}
    for task in held_out_tasks:
        sub = [e for e in test_ex if e.task == task]
        ceilings[task] = composition_ceiling(
            [e.sequence for e in sub], np.array([e.label for e in sub]),
            folds=5, seed=args.seed,
        )
        c = ceilings[task]
        print(f"  {task:16s} | n={c['n']:5d} | majority {c['majority']:.3f} "
              f"| composition {c['composition_oof']:.3f} "
              f"| ceiling {c['ceiling']:.3f} +/-{c['ceiling_ci95']:.3f}")

    # Mechanism-swap: each held-out target is scored a second time with a prompt
    # from another family. Pre-registered outcome 5; eval-only, so it is cheap.
    swap = ablation.swap_map(meta, held_out_tasks, seed=args.seed)
    print("\n--- Mechanism-swap assignment (donor prompts) ---")
    for line in ablation.describe(swap, meta):
        print(f"  {line}")

    results, run_meta = {}, {}
    y_test = np.array([e.label for e in test_ex])
    y_val = np.array([e.label for e in val_ex])

    # ---------- Fine-tuned runs: live encoder, last n blocks trainable ----------
    encoder, tokenizer = load_protein_encoder(
        args.esm, trainable_blocks=args.trainable_blocks, device=device
    )
    report = trainable_report(encoder)
    protein_revision = getattr(encoder.config, "_commit_hash", None)
    print(f"\nEncoder: {report['trainable_params']:,} of {report['total_params']:,} "
          f"params trainable ({report['trainable_fraction']:.1%})")

    ids, mask = tokenize_sequences(unique_seqs, tokenizer, max_len=args.max_len)
    row_of = {s: i for i, s in enumerate(unique_seqs)}

    def tokens_for(rows):
        pick = torch.tensor([row_of[e.sequence] for e in rows], dtype=torch.long)
        return ids[pick], mask[pick]

    live_tensors = {
        "train": build_tensor_dict(train_ex, None, q_map, A_map, task_to_id, device,
                                  tokens=tokens_for(train_ex)),
        "val": build_tensor_dict(val_ex, None, q_map, A_map, task_to_id, device,
                                 tokens=tokens_for(val_ex)),
        "test": build_tensor_dict(test_ex, None, q_map, A_map, task_to_id, device,
                                  tokens=tokens_for(test_ex)),
    }
    d_entity = encoder.config.hidden_size
    prefix_states, frozen_mat = None, None
    if args.cache_frozen_prefix:
        print("\nCaching only the frozen prefix (full precision, CPU)", flush=True)
        prefix_states = cache_frozen_prefix(
            encoder, ids, mask, args.trainable_blocks, args.batch_size)
        for split, rows in (("train", train_ex), ("val", val_ex), ("test", test_ex)):
            live_tensors[split]["prefix_idx"] = torch.tensor(
                [row_of[e.sequence] for e in rows], dtype=torch.long)
        if not args.skip_frozen_reference:
            encoder.eval()
            parts = []
            with torch.no_grad():
                for idx in batches(len(unique_seqs), args.batch_size):
                    parts.append(pooled_suffix(
                        encoder, prefix_states[idx].to(device), mask[idx].to(device),
                        args.trainable_blocks).float().cpu())
            frozen_mat = torch.cat(parts).numpy()
        print(f"  prefix cache: {prefix_states.numel() * prefix_states.element_size() / 1e9:.2f} GB", flush=True)
    # This encoder was only needed for shapes, tokenization and frozen caches.
    del encoder
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # A live text tower needs indices rather than vectors: cached embeddings
    # cannot carry a gradient. There is one question per task and two answer
    # strings in the whole benchmark, so a step forwards ~23 short sequences.
    question_order = sorted(meta)
    text_tables, text_report = None, None
    if args.unfreeze_text:
        text_probe, text_tok = load_text_encoder(text_model, args.text_trainable_blocks, device)
        q_ids, q_mask = tokenize_sequences(
            [questions[t] for t in question_order], text_tok, max_len=64)
        answer_order = sorted({o for t in meta for o in meta[t]["options"][:2]})
        a_ids, a_mask = tokenize_sequences(answer_order, text_tok, max_len=64)
        text_tables = {
            "questions": TextTable(q_ids, q_mask),
            "answers": TextTable(a_ids, a_mask),
            "question_order": question_order,
            "answer_order": answer_order,
        }
        q_pos = {t: i for i, t in enumerate(question_order)}
        a_pos = {txt: i for i, txt in enumerate(answer_order)}
        for split, rows in (("train", train_ex), ("val", val_ex), ("test", test_ex)):
            live_tensors[split]["q_idx"] = torch.tensor(
                [q_pos[r.task] for r in rows], dtype=torch.long, device=device)
            live_tensors[split]["a_idx"] = torch.tensor(
                [[a_pos[meta[r.task]["options"][k]] for k in (0, 1)] for r in rows],
                dtype=torch.long, device=device)
        text_report = trainable_report(text_probe)
        print(f"Text encoder: {text_report['trainable_params']:,} of "
              f"{text_report['total_params']:,} params trainable "
              f"({text_report['trainable_fraction']:.1%})")
        del text_probe

    live_models = [
        ("cross_attention_live", CrossAttentionAster, "dual"),
        ("dual_live", RealAster, "dual"),
        ("entity_only_live", CrossAttentionAster, "entity_only"),
    ]
    if args.unfreeze_text:
        # The mirror of entity_only on the text side: the control for "answers
        # from the prompt alone" must get the same trainable text blocks the
        # hypothesis model gets, or the comparison is rigged in its favour.
        live_models.append(("question_only_live", CrossAttentionAster, "question_only"))
    print("\n--- Fine-tuned (live encoder) ---")
    for name, cls, mode in live_models:
        print(f"\n  training {name} (mode={mode})")
        torch.manual_seed(args.seed)
        head = cls(d_entity, d_text, len(train_tasks), mode=mode)
        enc, _ = load_protein_encoder(args.esm, args.trainable_blocks, device)
        text_enc = None
        if args.unfreeze_text and mode in LIVE_TEXT_MODES:
            text_enc, _ = load_text_encoder(
                text_model, args.text_trainable_blocks, device)
        model = LiveAster(
            head, enc, text_encoder=text_enc,
            questions=text_tables["questions"] if text_enc is not None else None,
            answers=text_tables["answers"] if text_enc is not None else None,
            prefix_states=prefix_states, trainable_blocks=args.trainable_blocks,
        ).to(device)
        before = snapshot_encoder(model)
        text_before = snapshot_encoder(model, "text_encoder")
        model = train(model, live_tensors["train"], live_tensors["val"], device,
                      epochs=args.epochs, bs=args.batch_size, head_lr=args.head_lr,
                      encoder_lr=args.encoder_lr, text_lr=args.text_lr,
                      seed=args.seed, label=name)
        drift = encoder_drift(model, before)
        text_drift = encoder_drift(model, text_before, "text_encoder")
        res = evaluate(
            forward_all(model, live_tensors["test"], args.batch_size, device),
            forward_all(model, live_tensors["val"], args.batch_size, device),
            y_test, y_val, test_ex, held_out_tasks, ceilings,
        )
        res["encoder_drift"] = drift
        res["training_history"] = model.training_history
        if args.checkpoint_dir:
            checkpoint = Path(args.checkpoint_dir) / f"{name}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": snapshot_trainable_state(model),
                        "base_encoder": model.encoder.config._name_or_path,
                        "base_revision": getattr(model.encoder.config, "_commit_hash", None),
                        "model": name, "mode": mode, "config": vars(args)}, checkpoint)
            res["checkpoint"] = str(checkpoint)
        res["text_encoder_drift"] = text_drift
        res["encoder"] = {
            "live": True,
            "text_live": model.text_is_live,
            "arch_version": getattr(cls, "ARCH_VERSION", 1),
        }
        if mode != "entity_only":
            swapped = (
                ablation.swapped_index_tensor(
                    live_tensors["test"], test_ex,
                    text_tables["question_order"], swap, device)
                if model.text_is_live else
                ablation.swapped_question_tensor(
                    live_tensors["test"], test_ex, q_map, swap, device)
            )
            res["mechanism_swap"] = mechanism_swap(
                model, live_tensors["test"], swapped,
                test_ex, y_test, held_out_tasks,
                lambda m, t: forward_all(m, t, args.batch_size, device), ablation,
            )
        results[name] = res
        extra = f" | drift={drift:.2e}"
        if model.text_is_live:
            extra += f" | text drift={text_drift:.2e}"
        print_result(name, res, held_out_tasks, extra=extra)
        if "mechanism_swap" in res:
            ms = res["mechanism_swap"]
            print(f"    mechanism swap: {ms['own_prompt_accuracy']:.3f} -> "
                  f"{ms['swapped_prompt_accuracy']:.3f} (drop {ms['drop']:+.3f}"
                  f"{'' if ms['drop_is_significant'] else ', inside intervals'})")
            print(f"      {ms['reading']}")
        if drift < 1e-8:
            print("    WARN encoder did not move. This is not a fine-tuning result.")
        if model.text_is_live and text_drift < 1e-8:
            print("    WARN text encoder did not move despite --unfreeze-text.")
        del model, enc, before, text_before, text_enc, head
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---------- Frozen reference: cached vectors, identical heads ----------
    if not args.skip_frozen_reference:
        print("\n--- Frozen reference (cached embeddings, same heads) ---")
        esm_mat = frozen_mat if frozen_mat is not None else embed_sequences(
            unique_seqs, model=args.esm, device=device,
            max_len=args.max_len, batch_size=args.batch_size)
        emb = {s: esm_mat[i] for i, s in enumerate(unique_seqs)}
        frozen_tensors = {
            split: build_tensor_dict(
                rows, np.stack([emb[e.sequence] for e in rows]),
                q_map, A_map, task_to_id, device,
            )
            for split, rows in (("train", train_ex), ("val", val_ex), ("test", test_ex))
        }
        frozen_models = [
            ("cross_attention_frozen", CrossAttentionAster, "dual"),
            ("dual_frozen", RealAster, "dual"),
            ("entity_only_frozen", CrossAttentionAster, "entity_only"),
            ("task_id_frozen", CrossAttentionAster, "task_id"),
            ("question_only_frozen", CrossAttentionAster, "question_only"),
        ]
        for name, cls, mode in frozen_models:
            print(f"\n  training {name} (mode={mode})")
            torch.manual_seed(args.seed)
            head = cls(esm_mat.shape[1], d_text, len(train_tasks), mode=mode).to(device)
            head = train(head, frozen_tensors["train"], frozen_tensors["val"], device,
                         epochs=args.epochs, bs=args.batch_size, head_lr=args.head_lr,
                         encoder_lr=args.encoder_lr, seed=args.seed, label=name)
            res = evaluate(
                forward_all(head, frozen_tensors["test"], 4096, device),
                forward_all(head, frozen_tensors["val"], 4096, device),
                y_test, y_val, test_ex, held_out_tasks, ceilings,
            )
            res["encoder"] = {"live": False, "arch_version": getattr(cls, "ARCH_VERSION", 1)}
            if mode not in ("entity_only", "task_id"):
                res["mechanism_swap"] = mechanism_swap(
                    head, frozen_tensors["test"],
                    ablation.swapped_question_tensor(
                        frozen_tensors["test"], test_ex, q_map, swap, device),
                    test_ex, y_test, held_out_tasks,
                    lambda m, t: forward_all(m, t, 4096, device), ablation,
                )
            results[name] = res
            res["training_history"] = head.training_history
            print_result(name, res, held_out_tasks)
            if "mechanism_swap" in res:
                ms = res["mechanism_swap"]
                print(f"    mechanism swap: {ms['own_prompt_accuracy']:.3f} -> "
                      f"{ms['swapped_prompt_accuracy']:.3f} (drop {ms['drop']:+.3f}"
                      f"{'' if ms['drop_is_significant'] else ', inside intervals'})")

        for live, frozen in (("cross_attention_live", "cross_attention_frozen"),
                             ("dual_live", "dual_frozen"),
                             ("entity_only_live", "entity_only_frozen")):
            if live in results and frozen in results:
                delta = results[live]["overall_accuracy"] - results[frozen]["overall_accuracy"]
                run_meta[f"delta_{live}"] = delta
                print(f"  unfreezing delta [{live} - {frozen}]: {delta:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "command": " ".join(sys.argv),
                "esm": args.esm,
                "protein_revision": protein_revision,
                "device": device,
                "torch_version": torch.__version__,
                "trainable_blocks": args.trainable_blocks,
                "encoder_trainable": report,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "head_lr": args.head_lr,
                "encoder_lr": args.encoder_lr,
                "max_len": args.max_len,
                "max_peptides": args.max_peptides,
                "training_protocol_version": 2,
                "frozen_reference_epochs": args.epochs,
                "frozen_reference_batch_size": args.batch_size,
                "cache_frozen_prefix": args.cache_frozen_prefix,
                "prefix_cache_dtype": str(prefix_states.dtype) if prefix_states is not None else None,
                "min_samples": MIN_SAMPLES,
                "seed": args.seed,
                "benchmark": args.benchmark,
                "held_out_property": (args.held_out_property
                                      if args.benchmark == "properties" else None),
                "property_thresholds": ({n: meta[n]["threshold"] for n in meta}
                                        if args.benchmark == "properties" else None),
                "negatives_are_measured": args.benchmark == "properties",
                "disjoint_sequences": (args.disjoint_sequences
                                       if args.benchmark == "properties" else None),
                "negative_policy": args.negative_policy,
                "balance_tasks": balance,
                "text_encoder": text_model,
                "text_encoder_dim": int(d_text),
                "text_encoder_trainable": bool(args.unfreeze_text),
                "text_trainable_blocks": args.text_trainable_blocks if args.unfreeze_text else 0,
                "text_lr": (args.text_lr if args.text_lr is not None else args.encoder_lr)
                           if args.unfreeze_text else None,
                "text_encoder_trainable_report": text_report,
                "live_text_modes": list(LIVE_TEXT_MODES) if args.unfreeze_text else [],
                "label_semantics_version": LABEL_SEMANTICS_VERSION,
                "frozen_reference_included": not args.skip_frozen_reference,
                "n_train": len(train_ex),
                "n_val": len(val_ex),
                "n_test": len(test_ex),
                "train_test_shared_sequences": len(
                    {e.sequence for e in train_ex} & {e.sequence for e in test_ex}),
            },
            "reporting_rule": REPORTING_RULE,
            "reading_notes": [
                "entity_only_live has a live encoder too; if it moves like the "
                "hypothesis model, the gain is peptide classification, not typed decisions.",
                "Only *_live vs *_frozen isolates unfreezing. task_id and "
                "question_only are frozen reference rows, not matched controls.",
                "encoder_drift ~0 alongside a gain means the encoder did not move "
                "and the effect is in the head.",
                "Tasks are held out; peptide sequences can occur across splits. "
                "This run does not establish generalization to unseen peptide families.",
            ],
            "held_out_tasks": held_out_tasks,
            "ceilings": ceilings,
            "mechanism_swap_map": swap,
            "prompt_families": {t: meta[t].get("prompt_family") for t in meta},
            "deltas": run_meta,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
