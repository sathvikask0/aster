"""Leave-one-task-out transfer test on real PeptiVerse data. Apple Silicon ready.

    python scripts/run_peptiverse.py --root ~/peptiverse --esm 35M

The experiment: train on three properties, hold out the fourth entirely, and
ask whether the model answers a question it has never been trained on. The
`task_id` control cannot -- it has no row for an unseen question -- so any
honest transfer claim is `dual` beating `task_id`, both measured against the
composition shortcut ceiling rather than against chance.

First run embeds with ESM-2 and caches; later runs reuse the cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from aster.control import metrics as M
from aster.real import peptiverse as PV
from aster.real.embed import embed_sequences, embed_texts, pick_device
from aster.real.models import MODE_NOTES, MODES, RealAster


def build_tensors(examples, X, Q, A, task_index, device):
    n = len(examples)
    a = np.stack([np.stack([A[(e.task, 0)], A[(e.task, 1)]]) for e in examples])
    d = {
        "x": torch.from_numpy(X).float(),
        "q_emb": torch.from_numpy(np.stack([Q[e.task] for e in examples])).float(),
        "a_emb": torch.from_numpy(a).float(),
        "task_id": torch.tensor([task_index.get(e.task, 0) for e in examples]),
        "y": torch.tensor([e.label for e in examples]),
    }
    return {k: v.to(device) for k, v in d.items()}


def subset(t, idx):
    return {k: v[idx] for k, v in t.items()}


def train(tr, va, mode, d_entity, d_text, n_tasks, device, epochs=25, bs=256,
          lr=2e-3, seed=0, verbose=False, class_weight=True):
    torch.manual_seed(seed)
    model = RealAster(d_entity, d_text, n_tasks, mode=mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    n = tr["y"].shape[0]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * (n // bs + 1)))

    # Class weights: hemolysis and nephrotoxicity are ~79% one class, and an
    # unweighted model will happily collapse onto the majority and call it 79%.
    y = tr["y"]
    if class_weight:
        w = torch.tensor([1.0 / max((y == k).sum().item(), 1) for k in (0, 1)],
                         device=device, dtype=torch.float)
        w = w / w.sum() * 2
    else:
        # Off: lets us check whether below-chance transfer accuracy is the
        # weighting pushing toward the minority class rather than a real
        # inverted mapping.
        w = None

    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = subset(tr, perm[i:i + bs])
            loss = F.cross_entropy(model(b), b["y"], weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            vnll = F.cross_entropy(model(va), va["y"], weight=w).item()
        if vnll < best:
            best, best_state = vnll, {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}
        if verbose:
            print(f"      ep{ep:02d} val={vnll:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def logits_of(model, t, bs=4096):
    model.eval()
    out = [model(subset(t, slice(i, i + bs))).cpu().numpy()
           for i in range(0, t["y"].shape[0], bs)]
    return np.concatenate(out), t["y"].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/peptiverse")
    ap.add_argument("--esm", default="35M", choices=["8M", "35M", "150M", "650M"])
    ap.add_argument("--text-model", default="prajjwal1/bert-tiny")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--keep-shared", action="store_true",
                    help="keep the solubility/nephrotoxicity overlap (shows "
                         "the inflated result the audit warns about)")
    ap.add_argument("--tasks", nargs="*", default=list(PV.TASKS))
    ap.add_argument("--no-class-weight", action="store_true",
                    help="disable class weighting (diagnostic for below-chance "
                         "accuracy on the imbalanced tasks)")
    ap.add_argument("--out", default="reports/peptiverse_results.json")
    a = ap.parse_args()

    dev = pick_device(a.device)
    print(f"device: {dev}   esm: {a.esm}   text: {a.text_model}\n")

    print("loading PeptiVerse")
    ex = PV.load(a.root, a.tasks, drop_shared=not a.keep_shared,
                 max_len=a.max_len)

    print("\nshortcut ceilings (composition only -- the real zero):")
    ceil = PV.composition_ceiling(ex)
    for t, c in ceil.items():
        print(f"  {t:16s} majority={c['majority']:.3f} "
              f"composition={c['composition']:.3f} -> ceiling={c['ceiling']:.3f}")

    print("\nembedding")
    seqs = [e.sequence for e in ex]
    X = embed_sequences(seqs, a.esm, dev, a.batch_size, max_len=a.max_len)

    qs = sorted({e.question for e in ex})
    ans = sorted({o for e in ex for o in e.options})
    QE = dict(zip(qs, embed_texts(qs, a.text_model, dev)))
    AE = dict(zip(ans, embed_texts(ans, a.text_model, dev)))
    Q = {e.task: QE[e.question] for e in ex}
    A = {(e.task, i): AE[e.options[i]] for e in ex for i in (0, 1)}

    tasks = sorted({e.task for e in ex})
    tindex = {t: i + 1 for i, t in enumerate(tasks)}   # 0 reserved for <unk>
    T = build_tensors(ex, X, Q, A, tindex, dev)
    task_of = np.array([e.task for e in ex])
    d_entity, d_text = X.shape[1], len(next(iter(QE.values())))

    # ---------------- in-distribution control ----------------------
    # THE prerequisite. Train on every task with a random split and test on
    # questions the model has seen. If dual cannot clear the ceiling HERE, the
    # model is broken and the leave-one-out numbers say nothing about transfer.
    # task_id should do well too -- on seen questions a lookup table is enough,
    # which is exactly why this split cannot test the hypothesis.
    print(f"\n{'=' * 74}\nIN-DISTRIBUTION CONTROL (random split, seen questions)"
          f"\n{'=' * 74}")
    n_all = len(ex)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_all)
    n_te = int(n_all * 0.2); n_va = int(n_all * 0.1)
    te_i, va_i, tr_i = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]

    weighted_ceiling = float(np.mean([ceil[t]["ceiling"] for t in task_of[te_i]]))
    print(f"  weighted shortcut ceiling on this test set: {weighted_ceiling:.3f}")

    indist = {}
    for mode in MODES:
        din = 21 if mode == "composition" else d_entity
        src = T
        if mode == "composition":
            src = dict(T)
            src["x"] = torch.from_numpy(PV.composition_matrix(seqs)).float().to(dev)
        trm = subset(src, torch.from_numpy(tr_i).to(dev))
        vam = subset(src, torch.from_numpy(va_i).to(dev))
        tem = subset(src, torch.from_numpy(te_i).to(dev))   # task_id NOT zeroed

        m = train(trm, vam, mode, din, d_text, len(tasks), dev, epochs=a.epochs,
                  class_weight=not a.no_class_weight)
        vl, vy = logits_of(m, vam)
        t_ = M.fit_temperature(vl, vy)
        tl, ty = logits_of(m, tem)
        mm = M.compute(M.softmax(tl, t_), ty)
        per_task = {}
        for t in tasks:
            sel = np.where(task_of[te_i] == t)[0]
            if len(sel):
                per_task[t] = float((M.softmax(tl, t_)[sel].argmax(1)
                                     == ty[sel]).mean())
        indist[mode] = {"accuracy": mm.accuracy, "ece": mm.ece,
                        "over_ceiling": mm.accuracy - weighted_ceiling,
                        "per_task": per_task}
        print(f"  {mode:15s} acc={mm.accuracy:.3f}  "
              f"vs ceiling {mm.accuracy - weighted_ceiling:+.3f}  "
              f"ece={mm.ece:.3f}   " +
              "  ".join(f"{t[:4]}={v:.3f}" for t, v in per_task.items()))

    d_in = indist["dual"]["accuracy"]
    print()
    if d_in <= weighted_ceiling + 0.02:
        print("  BROKEN: dual cannot beat the shortcut even on questions it was")
        print("  trained on. Debug the model before reading any transfer number.")
    else:
        print(f"  OK: dual clears the ceiling by "
              f"{d_in - weighted_ceiling:+.3f} on seen questions, so the model")
        print("  learns. Any leave-one-out failure below is about transfer, not "
              "capacity.")

    results, rows = {}, []
    for held in tasks:
        te_idx = np.where(task_of == held)[0]
        tr_idx = np.where(task_of != held)[0]
        rng = np.random.default_rng(0); rng.shuffle(tr_idx)
        k = int(len(tr_idx) * 0.1)
        va_idx, tr_idx = tr_idx[:k], tr_idx[k:]

        tr = subset(T, torch.from_numpy(tr_idx).to(dev))
        va = subset(T, torch.from_numpy(va_idx).to(dev))
        te = subset(T, torch.from_numpy(te_idx).to(dev))

        # An unseen task must have no trained id: map it to <unk>.
        te = dict(te); te["task_id"] = torch.zeros_like(te["task_id"])

        c = ceil[held]
        print(f"\n{'=' * 74}\nHELD OUT: {held}   "
              f"(train {len(tr_idx)}, test {len(te_idx)})")
        print(f"  shortcut ceiling {c['ceiling']:.3f} "
              f"(majority {c['majority']:.3f}, composition {c['composition']:.3f})")
        print("=" * 74)

        per = {}
        for mode in MODES:
            din = 21 if mode == "composition" else d_entity
            trm, vam, tem = tr, va, te
            if mode == "composition":
                Xc = PV.composition_matrix(seqs)
                Tc = dict(T); Tc["x"] = torch.from_numpy(Xc).float().to(dev)
                trm = subset(Tc, torch.from_numpy(tr_idx).to(dev))
                vam = subset(Tc, torch.from_numpy(va_idx).to(dev))
                tem = subset(Tc, torch.from_numpy(te_idx).to(dev))
                tem = dict(tem); tem["task_id"] = torch.zeros_like(tem["task_id"])

            m = train(trm, vam, mode, din, d_text, len(tasks), dev,
                      epochs=a.epochs, class_weight=not a.no_class_weight)
            vl, vy = logits_of(m, vam)
            t_ = M.fit_temperature(vl, vy)
            tl, ty = logits_of(m, tem)
            mm = M.compute(M.softmax(tl, t_), ty)

            over = mm.accuracy - c["ceiling"]
            per[mode] = {"accuracy": mm.accuracy, "over_ceiling": over,
                         "ece": mm.ece, "overconfidence": mm.overconfidence,
                         "temperature": t_, "n": mm.n}
            print(f"  {mode:15s} acc={mm.accuracy:.3f}  "
                  f"vs ceiling {over:+.3f}  ece={mm.ece:.3f}  "
                  f"overconf={mm.overconfidence:+.3f}")
            rows.append({"held_out": held, "mode": mode, **per[mode]})

        d_, t_id = per["dual"]["accuracy"], per["task_id"]["accuracy"]
        print()
        print(f"  VERDICT  dual {d_:.3f} vs task_id {t_id:.3f} "
              f"vs ceiling {c['ceiling']:.3f}")
        if d_ <= c["ceiling"] + 0.02:
            print("           dual does not clear the composition shortcut. "
                  "No transfer demonstrated.")
        elif d_ <= t_id + 0.02:
            print("           dual does not beat the lookup control. "
                  "The text encoder is not earning its place.")
        else:
            print(f"           dual clears both by {d_ - max(t_id, c['ceiling']):+.3f}. "
                  "Worth a closer look.")
        results[held] = {"ceiling": c, "modes": per}

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "ceilings": ceil,
                               "in_distribution": indist,
                               "results": results, "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    print(f"\n{'=' * 74}\nSUMMARY (accuracy minus shortcut ceiling)\n{'=' * 74}")
    print(f"  {'held out':<16}" + "".join(f"{m:>16}" for m in MODES))
    for held in tasks:
        r = results[held]["modes"]
        print(f"  {held:<16}" + "".join(
            f"{r[m]['over_ceiling']:>+16.3f}" for m in MODES))
    print(f"  {'IN-DIST (seen)':<16}" + "".join(
        f"{indist[m]['over_ceiling']:>+16.3f}" for m in MODES))
    print("\n  A positive `dual` column that `task_id` does not match is the "
          "only cell\n  that supports the transfer claim. The IN-DIST row must "
          "be positive\n  first -- if it is not, the model is broken and the "
          "rest is noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
