"""Mechanism-swap: does the prompt work as biology, or as a name?

The AMP prompts are written as mechanism descriptions -- LPS outer membrane for
Gram-negatives, thick peptidoglycan for Gram-positives, chitin and ergosterol
for fungi -- on the theory that an unseen target is a recombination of seen
mechanism terms rather than a new symbol. That theory has never been tested. A
model can score exactly as well by treating each prompt as an opaque identifier
that happens to be spelled in English.

The test: at evaluation time, hand each held-out target a prompt from a
*different* mechanism family. If accuracy is unchanged, the mechanism language
bought nothing and the prompt is a task id. This is pre-registered as outcome 5
of v0.3, and it is eval-only -- no retraining, so it costs one forward pass.

A note on what a drop does and does not prove. Losing accuracy under a swapped
prompt shows the model is sensitive to prompt content, which is weaker than
showing it uses the mechanism *correctly*: a model keyed to surface features
that happen to correlate with the family would also drop. Read it as a
necessary condition, not a sufficient one.
"""

from __future__ import annotations

import numpy as np


def swap_map(meta: dict, held_out_tasks, seed: int = 42) -> dict[str, str]:
    """Assign each held-out target a donor task from a different prompt family.

    The donor is drawn from the *training* tasks where possible, so the swapped
    prompt is one the model has actually seen and cannot be dismissed as
    out-of-distribution text. Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    families = {t: meta[t].get("prompt_family", "generic") for t in meta}
    train_tasks = sorted(t for t in meta if not meta[t].get("is_held_out"))

    mapping = {}
    for task in sorted(held_out_tasks):
        own = families.get(task)
        donors = [t for t in train_tasks if families.get(t) != own]
        if not donors:
            donors = [t for t in sorted(meta) if families.get(t) != own and t != task]
        if not donors:
            continue
        mapping[task] = donors[int(rng.integers(len(donors)))]
    return mapping


def swapped_question_tensor(tensors, examples, q_map, mapping, device):
    """A copy of the test batch whose question embeddings come from the donors."""
    import torch

    swapped = dict(tensors)
    rows = np.stack([
        q_map[mapping.get(e.task, e.task)] for e in examples
    ])
    swapped["q_emb"] = torch.from_numpy(rows).float().to(device)
    return swapped


def describe(mapping: dict[str, str], meta: dict) -> list[str]:
    return [
        f"{task} ({meta[task].get('prompt_family')}) <- {donor} "
        f"({meta[donor].get('prompt_family')})"
        for task, donor in sorted(mapping.items())
    ]


def read_swap(own_acc: float, swapped_acc: float, ci: float) -> str:
    """The pre-registered reading, so it is not re-decided after seeing numbers."""
    drop = own_acc - swapped_acc
    if abs(drop) <= ci:
        return (
            "prompt content is not being used: accuracy survives a prompt from "
            "another mechanism family, so the prompt is functioning as a task "
            "identifier spelled in English"
        )
    if drop > ci:
        return (
            "prompt content is being used: a wrong-family prompt costs accuracy. "
            "Necessary but not sufficient for the mechanism language mattering -- "
            "sensitivity to prompt text is not correct use of mechanism"
        )
    return (
        "accuracy IMPROVED with a wrong-family prompt, which no version of the "
        "hypothesis predicts; suspect the prompt is acting as a label-prior knob"
    )


def swapped_index_tensor(tensors, examples, question_order, mapping, device):
    """Swap by question *index*, for a live text encoder.

    With the text tower in the graph the batch carries q_idx rather than a
    precomputed q_emb, so the swap has to move the index. Same assignment, same
    reading; only the plumbing differs.
    """
    import torch

    if "q_idx" not in tensors:
        raise KeyError("swapped_index_tensor needs q_idx; use swapped_question_tensor")
    position = {task: i for i, task in enumerate(question_order)}
    rows = [position[mapping.get(e.task, e.task)] for e in examples]
    swapped = dict(tensors)
    swapped["q_idx"] = torch.tensor(rows, dtype=torch.long, device=device)
    return swapped
