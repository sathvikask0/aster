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
    LiveEntityAster, encoder_drift, load_protein_encoder, param_groups,
    snapshot_encoder, tokenize_sequences, trainable_report,
)
from aster.real.models import CrossAttentionAster, RealAster

MIN_SAMPLES = 400
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def train(model, tr, va, device, epochs, bs, head_lr, encoder_lr, clip=1.0,
          seed=42, patience=2, label=""):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(param_groups(model, head_lr=head_lr, encoder_lr=encoder_lr))
    n = tr["y"].shape[0]
    steps = epochs * ((n + bs - 1) // bs) + 10
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps
    )

    best_loss, best_weights, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        t0, seen, running = time.time(), 0, 0.0
        for idx in batches(n, bs, shuffle=True, generator=gen):
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

        va_logits = forward_all(model, va, bs, device)
        val_loss = F.cross_entropy(
            torch.from_numpy(va_logits), va["y"].cpu()
        ).item()
        print(f"    epoch {epoch + 1}/{epochs}  train {running / max(seen, 1):.4f} "
              f"| val {val_loss:.4f} | {time.time() - t0:.0f}s", flush=True)

        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"    early stop: val loss has not improved in {patience} epochs")
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
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
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--max-peptides", type=int, default=None)
    p.add_argument("--negative-policy", default="matched",
                   choices=["random", "covered", "matched"])
    p.add_argument("--no-balance-tasks", action="store_true",
                   help="Leave the training label prior skewed (not recommended)")
    p.add_argument("--skip-frozen-reference", action="store_true",
                   help="Skip the cached-embedding runs. They are minutes on a warm "
                        "cache and they are the only thing that makes the fine-tuned "
                        "number attributable, so skipping is rarely worth it.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="reports/amp_finetune_results.json")
    args = p.parse_args()

    device = pick_device(args.device)
    balance = not args.no_balance_tasks
    print(f"AMP fine-tune | ESM-2 {args.esm} | last {args.trainable_blocks} blocks "
          f"trainable | policy={args.negative_policy} | balance={balance} | {device}")

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
    q_mat = embed_texts([questions[t] for t in meta], model=TEXT_MODEL, device=device)
    q_map = {t: q_mat[i] for i, t in enumerate(meta)}
    ans_texts = sorted({o for t in meta for o in meta[t]["options"][:2]})
    ans_mat = embed_texts(ans_texts, model=TEXT_MODEL, device=device)
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

    live_models = [
        ("cross_attention_live", CrossAttentionAster, "dual"),
        ("dual_live", RealAster, "dual"),
        ("entity_only_live", CrossAttentionAster, "entity_only"),
    ]
    print("\n--- Fine-tuned (live encoder) ---")
    for name, cls, mode in live_models:
        print(f"\n  training {name} (mode={mode})")
        torch.manual_seed(args.seed)
        head = cls(d_entity, d_text, len(train_tasks), mode=mode)
        enc, _ = load_protein_encoder(args.esm, args.trainable_blocks, device)
        model = LiveEntityAster(head, enc).to(device)
        before = snapshot_encoder(model)
        model = train(model, live_tensors["train"], live_tensors["val"], device,
                      epochs=args.epochs, bs=args.batch_size, head_lr=args.head_lr,
                      encoder_lr=args.encoder_lr, seed=args.seed, label=name)
        drift = encoder_drift(model, before)
        res = evaluate(
            forward_all(model, live_tensors["test"], args.batch_size, device),
            forward_all(model, live_tensors["val"], args.batch_size, device),
            y_test, y_val, test_ex, held_out_tasks, ceilings,
        )
        res["encoder_drift"] = drift
        res["encoder"] = {"live": True, "arch_version": getattr(cls, "ARCH_VERSION", 1)}
        if mode != "entity_only":
            res["mechanism_swap"] = mechanism_swap(
                model, live_tensors["test"],
                ablation.swapped_question_tensor(
                    live_tensors["test"], test_ex, q_map, swap, device),
                test_ex, y_test, held_out_tasks,
                lambda m, t: forward_all(m, t, args.batch_size, device), ablation,
            )
        results[name] = res
        print_result(name, res, held_out_tasks, extra=f" | drift={drift:.2e}")
        if "mechanism_swap" in res:
            ms = res["mechanism_swap"]
            print(f"    mechanism swap: {ms['own_prompt_accuracy']:.3f} -> "
                  f"{ms['swapped_prompt_accuracy']:.3f} (drop {ms['drop']:+.3f}"
                  f"{'' if ms['drop_is_significant'] else ', inside intervals'})")
            print(f"      {ms['reading']}")
        if drift < 1e-8:
            print("    WARN encoder did not move. This is not a fine-tuning result.")
        del model, enc
        if device == "mps":
            torch.mps.empty_cache()

    # ---------- Frozen reference: cached vectors, identical heads ----------
    if not args.skip_frozen_reference:
        print("\n--- Frozen reference (cached embeddings, same heads) ---")
        esm_mat = embed_sequences(unique_seqs, model=args.esm, device=device,
                                  max_len=args.max_len)
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
                         epochs=max(args.epochs, 20), bs=256, head_lr=args.head_lr,
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
                "trainable_blocks": args.trainable_blocks,
                "encoder_trainable": report,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "head_lr": args.head_lr,
                "encoder_lr": args.encoder_lr,
                "max_len": args.max_len,
                "max_peptides": args.max_peptides,
                "min_samples": MIN_SAMPLES,
                "seed": args.seed,
                "negative_policy": args.negative_policy,
                "balance_tasks": balance,
                "text_encoder": TEXT_MODEL,
                "text_encoder_trainable": False,
                "label_semantics_version": LABEL_SEMANTICS_VERSION,
                "frozen_reference_included": not args.skip_frozen_reference,
                "n_train": len(train_ex),
                "n_val": len(val_ex),
                "n_test": len(test_ex),
            },
            "reporting_rule": REPORTING_RULE,
            "reading_notes": [
                "entity_only_live has a live encoder too; if it moves like the "
                "hypothesis model, the gain is peptide classification, not typed decisions.",
                "Only *_live vs *_frozen isolates unfreezing. task_id and "
                "question_only are frozen reference rows, not matched controls.",
                "encoder_drift ~0 alongside a gain means the encoder did not move "
                "and the effect is in the head.",
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
