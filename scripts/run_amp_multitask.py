"""Run multi-task zero-shot transfer benchmark on 21 diverse antimicrobial pathogen assays.

Tests whether:
  1. Mechanistic prompts + 16 training tasks break the Rank-3 unidentifiability trap.
  2. Question-conditioned Cross-Attention outperforms the rigid pooled dot-product.
  3. Models achieve genuine positive transfer over task_id and composition ceilings on
     held-out Gram-negative, Gram-positive, and fungal targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aster.control import metrics as M
from aster.real.amp import load_amp_benchmark
from aster.real.embed import embed_sequences, embed_texts, pick_device
from aster.real.ceilings import binomial_ci95, composition_ceiling, composition_matrix
from aster.real.models import CrossAttentionAster, RealAster

MIN_SAMPLES = 400


def compute_composition(seqs: list[str]) -> np.ndarray:
    """20-dim amino-acid frequency vector (shared with the ceiling estimator)."""
    return composition_matrix(seqs).astype(np.float32)


def build_tensor_dict(examples, X, Q, A, task_to_id, device):
    """Convert dataset to device tensors."""
    seq_to_idx = {e.sequence: i for i, e in enumerate(examples)}
    task_ids = [task_to_id.get(e.task, 0) for e in examples]
    a = np.stack([np.stack([A[(e.task, 0)], A[(e.task, 1)]]) for e in examples])
    
    x_tensor = torch.from_numpy(X).float().to(device)
    q_tensor = torch.from_numpy(np.stack([Q[e.task] for e in examples])).float().to(device)
    a_tensor = torch.from_numpy(a).float().to(device)
    task_id_tensor = torch.tensor(task_ids, dtype=torch.long, device=device)
    y_tensor = torch.tensor([e.label for e in examples], dtype=torch.long, device=device)

    return {
        "x": x_tensor,
        "q_emb": q_tensor,
        "a_emb": a_tensor,
        "task_id": task_id_tensor,
        "y": y_tensor,
    }


def train_model(model_cls, tr, va, mode, d_in, d_text, n_tasks, device,
                epochs=20, bs=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = model_cls(d_in, d_text, n_tasks, mode=mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    n = tr["y"].shape[0]
    steps_per_epoch = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch + 10
    )

    best_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            batch = {k: v[idx] for k, v in tr.items()}
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["y"])
            opt.backward(loss) if hasattr(opt, "backward") else loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(va)
            val_loss = F.cross_entropy(v_logits, va["y"]).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm", default="8M", choices=["none", "8M", "35M"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-peptides", type=int, default=None, help="Cap unique peptides for rapid ESM caching")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="reports/amp_multitask_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-policy",
        default="random",
        choices=["random", "covered"],
        help=(
            "How presumed-negative rows are drawn. 'random' reproduces the "
            "published v0.2 run; 'covered' restricts them to peptides assayed "
            "against several other pathogens, so absence is less likely to be "
            "database coverage rather than measured inactivity."
        ),
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Loading 56-pathogen AMP multi-task dataset on {device}...")
    examples, meta = load_amp_benchmark(
        min_samples=MIN_SAMPLES, seed=args.seed, negative_policy=args.negative_policy
    )

    unique_seqs = sorted({e.sequence for e in examples})
    if args.max_peptides and len(unique_seqs) > args.max_peptides:
        rng = np.random.default_rng(42)
        kept_set = set(rng.choice(unique_seqs, size=args.max_peptides, replace=False))
        examples = [e for e in examples if e.sequence in kept_set]
        unique_seqs = sorted(kept_set)
        print(f"Subsampled to {len(unique_seqs):,} unique peptides ({len(examples):,} total rows)")

    train_ex = [e for e in examples if e.split == "train"]
    val_ex = [e for e in examples if e.split == "val"]
    test_ex = [e for e in examples if e.split == "test"]
    held_out_tasks = sorted({e.task for e in test_ex})
    train_tasks = sorted({e.task for e in train_ex})

    print(f"Train samples: {len(train_ex):,} across {len(train_tasks)} tasks")
    print(f"Val samples:   {len(val_ex):,}")
    print(f"Held-out test: {len(test_ex):,} across {len(held_out_tasks)} tasks: {held_out_tasks}")

    # Task to integer ID (unseen tasks mapped to 0)
    task_to_id = {t: i + 1 for i, t in enumerate(train_tasks)}

    # Encode Questions
    questions_dict = {t: meta[t]["question"] for t in meta}
    all_q_texts = [questions_dict[t] for t in meta]
    print(f"Embedding {len(all_q_texts)} mechanistic question prompts with MiniLM...")
    q_embs_mat = embed_texts(all_q_texts, model="sentence-transformers/all-MiniLM-L6-v2", device=device)
    q_emb_map = {t: q_embs_mat[i] for i, t in enumerate(meta)}

    # Encode Answer choices
    all_ans = []
    for t in meta:
        all_ans.extend([meta[t]["options"][0], meta[t]["options"][1]])
    all_ans_uniq = sorted(set(all_ans))
    ans_embs_mat = embed_texts(all_ans_uniq, model="sentence-transformers/all-MiniLM-L6-v2", device=device)
    ans_emb_dict = {txt: ans_embs_mat[i] for i, txt in enumerate(all_ans_uniq)}
    A_map = {}
    for t in meta:
        A_map[(t, 0)] = ans_emb_dict[meta[t]["options"][0]]
        A_map[(t, 1)] = ans_emb_dict[meta[t]["options"][1]]

    # Encode Sequences
    print(f"Embedding {len(unique_seqs):,} unique peptide sequences...")
    
    if args.esm != "none":
        print(f"Running ESM-2 {args.esm} embeddings...")
        esm_seq_mat = embed_sequences(unique_seqs, model=args.esm, device=device, max_len=64)
        seq_emb_map = {s: esm_seq_mat[i] for i, s in enumerate(unique_seqs)}
        d_entity = esm_seq_mat.shape[1]
    else:
        print("Using 20-dim amino acid composition vectors...")
        comp_seq_mat = compute_composition(unique_seqs)
        seq_emb_map = {s: comp_seq_mat[i] for i, s in enumerate(unique_seqs)}
        d_entity = 20

    # Build Tensors
    X_train = np.stack([seq_emb_map[e.sequence] for e in train_ex])
    X_val = np.stack([seq_emb_map[e.sequence] for e in val_ex])
    X_test = np.stack([seq_emb_map[e.sequence] for e in test_ex])

    tr_tensor = build_tensor_dict(train_ex, X_train, q_emb_map, A_map, task_to_id, device)
    va_tensor = build_tensor_dict(val_ex, X_val, q_emb_map, A_map, task_to_id, device)
    te_tensor = build_tensor_dict(test_ex, X_test, q_emb_map, A_map, task_to_id, device)

    d_text = q_embs_mat.shape[1]
    n_tasks = len(train_tasks)

    # Shortcut ceilings on the held-out tasks. The ceiling, not chance and not
    # the lookup floor, is the zero every claim below is measured against.
    print("\n--- Measuring Baselines & Ceilings on Held-Out Tasks ---")
    ceilings = {}
    for task in held_out_tasks:
        sub = [e for e in test_ex if e.task == task]
        labels = np.array([e.label for e in sub])
        ceilings[task] = composition_ceiling(
            [e.sequence for e in sub], labels, folds=5, seed=args.seed
        )
        c = ceilings[task]
        print(
            f"  {task:16s} | n={c['n']:5d} | Majority: {c['majority']:.3f} "
            f"| Comp (out-of-fold): {c['composition_oof']:.3f} "
            f"(in-sample {c['composition_insample']:.3f}) "
            f"| Ceiling: {c['ceiling']:.3f} +/-{c['ceiling_ci95']:.3f}"
        )

    results = {}
    models_to_test = [
        ("cross_attention", CrossAttentionAster, "dual"),
        ("dual_dot", RealAster, "dual"),
        ("task_id", CrossAttentionAster, "task_id"),
        ("question_only", CrossAttentionAster, "question_only"),
        ("entity_only", CrossAttentionAster, "entity_only"),
    ]

    print("\n--- Training and Evaluating Models on Multi-Task Zero-Shot Transfer ---")
    arch_versions = {}
    for name, m_cls, mode in models_to_test:
        print(f"\nTraining [{name}] (mode={mode})...")
        arch_versions[name] = getattr(m_cls, "ARCH_VERSION", 1)
        model = train_model(
            m_cls, tr_tensor, va_tensor, mode, d_entity, d_text, n_tasks, device,
            epochs=args.epochs, bs=256
        )

        with torch.no_grad():
            te_logits = model(te_tensor).cpu().numpy()
            va_logits = model(va_tensor).cpu().numpy()

        y_true = te_tensor["y"].cpu().numpy()
        y_val = va_tensor["y"].cpu().numpy()

        # Temperature is fit on validation and applied to test. It cannot make a
        # wrong model right; it only stops a model lying about how sure it is.
        temperature = M.fit_temperature(va_logits, y_val)
        probs = M.softmax(te_logits, temperature)
        preds = probs.argmax(1)

        raw = M.compute(M.softmax(te_logits), y_true).as_dict()
        calibrated = M.compute(probs, y_true).as_dict()
        overall_acc = calibrated["accuracy"]
        
        # Per-task accuracy
        per_task = {}
        for task in held_out_tasks:
            idx = np.array([i for i, e in enumerate(test_ex) if e.task == task])
            task_acc = float(np.mean(preds[idx] == y_true[idx]))
            task_lift = task_acc - ceilings[task]["ceiling"]
            per_task[task] = {
                "acc": task_acc,
                "acc_ci95": binomial_ci95(task_acc, len(idx)),
                "lift": task_lift,
                "lift_is_significant": bool(
                    abs(task_lift) > binomial_ci95(task_acc, len(idx))
                    + ceilings[task]["ceiling_ci95"]
                ),
            }

        results[name] = {
            "overall_accuracy": overall_acc,
            "temperature": float(temperature),
            "calibration": {"raw": raw, "calibrated": calibrated},
            "per_task": per_task,
        }
        print(
            f"  [{name:16s}] Acc: {overall_acc:.3f} | ECE: {calibrated['ece']:.3f} "
            f"| overconfidence: {calibrated['overconfidence']:+.3f} (T={temperature:.2f})"
        )
        if calibrated["overconfidence"] > 0.05:
            print(
                f"    WARN {name}: confident {calibrated['mean_confidence']:.3f} but right "
                f"{calibrated['accuracy']:.3f}. Do not spend bench time on this."
            )
        for t in held_out_tasks:
            flag = "" if per_task[t]["lift_is_significant"] else "  (inside intervals)"
            print(f"    {t:14s}: {per_task[t]['acc']:.3f} (lift vs ceiling: {per_task[t]['lift']:+.3f}){flag}")

    # Output report JSON
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "command": " ".join(sys.argv),
                "esm": args.esm,
                "epochs": args.epochs,
                "max_peptides": args.max_peptides,
                "min_samples": MIN_SAMPLES,
                "seed": args.seed,
                "negative_policy": args.negative_policy,
                "arch_versions": arch_versions,
                "n_train": len(train_ex),
                "n_val": len(val_ex),
                "n_test": len(test_ex),
            },
            "reporting_rule": (
                "Every claim is a margin over 'ceiling' (max of majority and the "
                "out-of-fold composition probe), never over task_id. A lift smaller "
                "than the combined 95% intervals is not a result."
            ),
            "held_out_tasks": held_out_tasks,
            "ceilings": ceilings,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
