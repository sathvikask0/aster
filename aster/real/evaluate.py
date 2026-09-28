"""One definition of how an AMP run is read.

Both AMP runners import from here. The reading rule -- temperature fitted on
validation and applied to test, every claim a margin over the shortcut ceiling,
a margin inside the combined intervals reported as no result -- is the part most
likely to drift between two scripts and produce two numbers that look
comparable and are not. It lives in one place for that reason.
"""

from __future__ import annotations

import numpy as np
import torch

from aster.control import metrics as M
from aster.real.ceilings import binomial_ci95


def build_tensor_dict(examples, X, Q, A, task_to_id, device, tokens=None):
    """Device tensors for a split.

    `X` is the cached entity matrix; pass None on the fine-tuning path, where
    `x` is computed in the forward pass. `tokens` is (input_ids, attention_mask)
    already aligned to `examples`, kept on CPU so a batch can be moved as it is
    consumed.
    """
    task_ids = [task_to_id.get(e.task, 0) for e in examples]
    a = np.stack([np.stack([A[(e.task, 0)], A[(e.task, 1)]]) for e in examples])

    batch = {
        "q_emb": torch.from_numpy(np.stack([Q[e.task] for e in examples])).float().to(device),
        "a_emb": torch.from_numpy(a).float().to(device),
        "task_id": torch.tensor(task_ids, dtype=torch.long, device=device),
        "y": torch.tensor([e.label for e in examples], dtype=torch.long, device=device),
    }
    if X is not None:
        batch["x"] = torch.from_numpy(X).float().to(device)
    if tokens is not None:
        batch["input_ids"], batch["attention_mask"] = tokens
    return batch


def evaluate(te_logits, va_logits, y_true, y_val, test_examples, held_out_tasks, ceilings):
    """Calibrate on validation, then score the test split against the ceilings."""
    temperature = M.fit_temperature(va_logits, y_val)
    probs = M.softmax(te_logits, temperature)
    preds = probs.argmax(1)

    raw = M.compute(M.softmax(te_logits), y_true).as_dict()
    calibrated = M.compute(probs, y_true).as_dict()

    per_task = {}
    for task in held_out_tasks:
        idx = np.array([i for i, e in enumerate(test_examples) if e.task == task])
        if idx.size == 0:
            continue
        acc = float(np.mean(preds[idx] == y_true[idx]))
        ci = binomial_ci95(acc, len(idx))
        lift = acc - ceilings[task]["ceiling"]
        per_task[task] = {
            "acc": acc,
            "acc_ci95": ci,
            "lift": lift,
            "lift_is_significant": bool(abs(lift) > ci + ceilings[task]["ceiling_ci95"]),
        }

    return {
        "overall_accuracy": calibrated["accuracy"],
        "temperature": float(temperature),
        "calibration": {"raw": raw, "calibrated": calibrated},
        "per_task": per_task,
    }


def print_result(name, result, held_out_tasks, extra=""):
    c = result["calibration"]["calibrated"]
    print(
        f"  [{name:18s}] Acc: {result['overall_accuracy']:.3f} | ECE: {c['ece']:.3f} "
        f"| overconfidence: {c['overconfidence']:+.3f} (T={result['temperature']:.2f}){extra}"
    )
    if c["overconfidence"] > 0.05:
        print(
            f"    WARN {name}: confident {c['mean_confidence']:.3f} but right "
            f"{c['accuracy']:.3f}. Do not spend bench time on this."
        )
    for t in held_out_tasks:
        pt = result["per_task"].get(t)
        if pt is None:
            continue
        flag = "" if pt["lift_is_significant"] else "  (inside intervals)"
        print(f"    {t:14s}: {pt['acc']:.3f} (lift vs ceiling: {pt['lift']:+.3f}){flag}")


REPORTING_RULE = (
    "Every claim is a margin over 'ceiling' (max of majority and the "
    "out-of-fold composition probe), never over task_id. A lift smaller "
    "than the combined 95% intervals is not a result."
)
