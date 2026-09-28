"""Train/evaluate every mode on every split, and judge the result.

`verdict()` is the part that matters. A table of numbers invites the reader to
find the story they wanted; an explicit pass/fail against an expectation
written down in advance does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from aster.control.tensors import TensorSet
from aster.control.models import MODES, ControlScorer, build_vocabs
from aster.control import metrics as M


@dataclass
class RunResult:
    mode: str
    split: str
    test: M.Metrics
    test_calibrated: M.Metrics
    temperature: float
    reliability: list


@torch.no_grad()
def _logits(model, ts, bs=1024):
    model.eval()
    out = []
    for i in range(0, ts.n, bs):
        out.append(model(ts.batch(slice(i, i + bs))).cpu().numpy())
    return np.concatenate(out), ts.d["y"].cpu().numpy()


def train_one(sets, mode, seed=0, epochs=12, bs=256, lr=4e-3, verbose=False):
    torch.manual_seed(seed)
    tr, va = sets["train"], sets["val"]
    model = ControlScorer(sets["vq"], sets["vopt"], sets["vqid"], mode=mode)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    steps = max(1, epochs * (tr.n // bs + 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    best_state, best_val = None, np.inf
    for ep in range(epochs):
        model.train()
        for b in tr.iter_batches(bs, seed=seed * 100 + ep):
            loss = F.cross_entropy(model(b), b["y"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

        vl, vy = _logits(model, va)
        vnll = float(F.cross_entropy(
            torch.from_numpy(vl), torch.from_numpy(vy)).item())
        if vnll < best_val:
            best_val = vnll
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if verbose:
            print(f"    ep{ep:02d} val_nll={vnll:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def make_sets(split, device="cpu"):
    # Vocabularies from TRAIN ONLY -- letting test tokens in would hand unseen
    # questions a trained embedding and manufacture transfer that is not there.
    vq, vopt, vqid = build_vocabs(split.train)
    return {
        "vq": vq, "vopt": vopt, "vqid": vqid,
        "train": TensorSet(split.train, vq, vopt, vqid, device=device),
        "val": TensorSet(split.val, vq, vopt, vqid, device=device),
        "test": TensorSet(split.test, vq, vopt, vqid, device=device),
    }


def evaluate(model, sets, split_name) -> RunResult:
    vl, vy = _logits(model, sets["val"])
    t = M.fit_temperature(vl, vy)          # fit on val, never on test
    tl, ty = _logits(model, sets["test"])
    raw = M.compute(M.softmax(tl, 1.0), ty)
    cal = M.compute(M.softmax(tl, t), ty)
    rel = M.reliability_table(M.softmax(tl, t), ty)
    return RunResult(model.mode, split_name, raw, cal, t, rel)


def run_grid(split, modes=MODES, seed=0, epochs=12, device="cpu"):
    sets = make_sets(split, device)
    results = []
    for mode in modes:
        model = train_one(sets, mode, seed=seed, epochs=epochs)
        results.append(evaluate(model, sets, split.name))
    return results


# ---------------------------------------------------------------- the verdict

TOL = 0.08  # lift above which we call something "above chance"


WARN_SHORTCUT = 0.15  # above this, part of the split is answerable for free


def shortcut_ceiling(by) -> float:
    """How well you can do WITHOUT using both inputs.

    Originally this rig asserted that question_only and entity_only must both
    sit at chance. That assertion was wrong, and the arbitrary /
    held_out_entity_family cell is what exposed it: when a held-out family sits
    far from the origin in latent space, sign(w . z) is dominated by the family
    offset rather than by w, so most questions share an answer for those
    entities and "ignore the question, predict this entity's majority label"
    scores ~0.60 legitimately.

    That is not an artifact to tune away -- real peptide families with extreme
    composition will behave the same way. So the shortcut level is treated as a
    measured property of the split, and every claim is made RELATIVE to it.
    """
    return max(0.0, by.get("question_only", 0.0), by.get("entity_only", 0.0))


def verdict(regime: str, split_name: str, results: list[RunResult]) -> list[str]:
    """Expectations registered in advance, not read off the output.

    These are assertions about the HARNESS, not about biology. If they fail on
    synthetic data whose ground truth we control, nothing the harness later
    says about real data can be believed.
    """
    by = {r.mode: r for r in results}
    lines = []

    def lift(m):
        return by[m].test_calibrated.lift_over_chance if m in by else float("nan")

    lifts = {m: lift(m) for m in by}
    floor = shortcut_ceiling(lifts)

    def check(ok, msg):
        lines.append(("PASS  " if ok else "FAIL  ") + msg)

    lines.append(
        f"INFO  shortcut ceiling={floor:+.3f} "
        f"(question_only={lifts.get('question_only', float('nan')):+.3f}, "
        f"entity_only={lifts.get('entity_only', float('nan')):+.3f})"
    )
    if floor > WARN_SHORTCUT:
        lines.append(
            f"WARN  {floor:.0%} of this split is answerable without using both "
            "inputs -- read every number below as a margin over that, not as "
            "an absolute"
        )

    if split_name in ("random", "held_out_entity_family"):
        check(lift("dual") > floor + 0.30,
              f"dual beats the shortcut on seen questions "
              f"(lift={lift('dual'):+.3f} vs floor {floor:+.3f})")
        check(lift("task_id") > floor + 0.30,
              f"task_id beats it too (lift={lift('task_id'):+.3f}) "
              "-- so this split cannot discriminate the hypothesis")

    if split_name in ("held_out_question", "held_out_both"):
        check(lift("task_id") <= floor + TOL,
              f"task_id collapses to the shortcut floor on unseen questions "
              f"(lift={lift('task_id'):+.3f} vs floor {floor:+.3f}) "
              "-- the transfer floor")
        if regime == "semantic":
            check(lift("dual") > floor + 0.25,
                  f"dual transfers where text is informative "
                  f"(lift={lift('dual'):+.3f} vs floor {floor:+.3f})")
        else:
            check(lift("dual") <= floor + TOL,
                  f"dual does NOT transfer where text is uninformative "
                  f"(lift={lift('dual'):+.3f} vs floor {floor:+.3f}) "
                  "-- rig is not fooling itself")

    return lines
