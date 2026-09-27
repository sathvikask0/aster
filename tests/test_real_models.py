"""What each mode is allowed to see, pinned so it cannot regress silently.

A control that cannot express the shortcut it is named after measures nothing,
and reads as a clean result. These tests are the guard against that.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aster.real.models import CrossAttentionAster, RealAster

B, D_E, D_T, N_TASKS, H = 6, 32, 16, 5, 32


def batch(k=2, seed=0, task_id=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(B, D_E, generator=g),
        "q_emb": torch.randn(B, D_T, generator=g),
        "a_emb": torch.randn(B, k, D_T, generator=g),
        "task_id": torch.full((B,), task_id, dtype=torch.long),
        "y": torch.randint(0, k, (B,), generator=g),
    }


def model(mode="dual", **kw):
    m = CrossAttentionAster(D_E, D_T, N_TASKS, h=H, n_heads=2, mode=mode, **kw)
    return m.eval()


def logits(m, b):
    with torch.no_grad():
        return m(b)


# --- the option set is free ------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 5])
def test_scores_every_candidate_answer(k):
    m = model()
    assert logits(m, batch(k=k)).shape == (B, k)


def test_rejects_more_options_than_it_was_built_for():
    m = model(max_options=3)
    with pytest.raises(ValueError):
        logits(m, batch(k=4))


def test_answer_text_changes_the_score():
    """Architecture v1 ignored a_emb entirely. This is that bug, pinned."""
    m = model()
    b = batch()
    other = dict(b, a_emb=torch.randn_like(b["a_emb"]))
    assert not torch.allclose(logits(m, b), logits(m, other))


def test_masked_options_are_unreachable():
    m = model()
    b = batch()
    b["option_mask"] = torch.tensor([[True, False]] * B)
    out = logits(m, b)
    assert torch.isinf(out[:, 1]).all() and torch.isfinite(out[:, 0]).all()


def test_a_row_with_no_valid_option_is_rejected():
    m = model()
    b = batch()
    b["option_mask"] = torch.zeros(B, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        logits(m, b)


# --- the lookup floor is structural, not empirical -------------------------

def test_task_id_is_exactly_chance_on_an_unseen_task():
    m = model("task_id")
    out = logits(m, batch(task_id=0))
    assert torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)


def test_task_id_can_still_separate_a_seen_task():
    m = model("task_id")
    out = logits(m, batch(task_id=3))
    assert not torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)


# --- the shortcut detectors detect their own shortcut ----------------------

def test_question_only_never_sees_the_entity():
    m = model("question_only")
    b = batch()
    assert torch.allclose(logits(m, b), logits(m, dict(b, x=torch.randn_like(b["x"]))))


def test_question_only_does_see_the_question():
    m = model("question_only")
    b = batch()
    assert not torch.allclose(
        logits(m, b), logits(m, dict(b, q_emb=torch.randn_like(b["q_emb"])))
    )


def test_entity_only_never_sees_the_question():
    m = model("entity_only")
    b = batch()
    same_entity = dict(
        b, q_emb=torch.randn_like(b["q_emb"]), a_emb=torch.randn_like(b["a_emb"])
    )
    assert torch.allclose(logits(m, b), logits(m, same_entity))


def test_entity_only_can_express_an_entity_prior():
    """It must vary with the entity AND across options, or it measures nothing."""
    m = model("entity_only")
    b = batch()
    out = logits(m, b)
    assert not torch.allclose(out, logits(m, dict(b, x=torch.randn_like(b["x"]))))
    assert not torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)


def test_real_aster_entity_only_is_not_a_constant_prediction():
    m = RealAster(D_E, D_T, N_TASKS, h=H, mode="entity_only").eval()
    b = batch()
    with torch.no_grad():
        out, moved = m(b), m(dict(b, x=torch.randn_like(b["x"])))
    assert not torch.allclose(out, moved)
    assert not torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)


def test_real_aster_task_id_is_chance_on_an_unseen_task():
    m = RealAster(D_E, D_T, N_TASKS, h=H, mode="task_id").eval()
    with torch.no_grad():
        out = m(batch(task_id=0))
    assert torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)


def test_gradients_reach_every_trainable_parameter():
    m = CrossAttentionAster(D_E, D_T, N_TASKS, h=H, n_heads=2, mode="dual")
    b = batch()
    torch.nn.functional.cross_entropy(m(b), b["y"]).backward()
    dead = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"no gradient reached: {dead}"
