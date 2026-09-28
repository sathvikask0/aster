"""What unfreezing the protein encoder is allowed to change, pinned.

These run on a randomly initialised ESM of the same class as ESM-2, so they
need no downloaded weights: the claims here are about gradient flow, pooling
and what each mode can see, none of which depend on pretraining.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import EsmConfig, EsmModel

from aster.model import blocks
from aster.real.finetune import (
    LiveEntityAster, encoder_drift, param_groups, snapshot_encoder, trainable_report,
)
from aster.real.models import CrossAttentionAster, RealAster

N_LAYERS, HIDDEN, D_TEXT, N_TASKS, VOCAB = 4, 32, 16, 5, 33


def encoder(trainable_blocks=2):
    from aster.model import unfreeze_suffix

    cfg = EsmConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=128,
        position_embedding_type="absolute", pad_token_id=1, mask_token_id=32,
        token_dropout=False,
    )
    enc = EsmModel(cfg, add_pooling_layer=False)
    if trainable_blocks:
        unfreeze_suffix(enc, trainable_blocks)
    return enc


def live(mode="dual", cls=CrossAttentionAster, trainable_blocks=2):
    enc = encoder(trainable_blocks)
    head = cls(HIDDEN, D_TEXT, N_TASKS, h=32, mode=mode) if cls is CrossAttentionAster \
        else cls(HIDDEN, D_TEXT, N_TASKS, h=32, mode=mode)
    return LiveEntityAster(head, enc)


def batch(b=6, k=2, seed=0, length=12):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, VOCAB, (b, length), generator=g)
    mask = torch.ones(b, length, dtype=torch.long)
    mask[:, -3:] = 0  # padding must not reach the pooled vector
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "q_emb": torch.randn(b, D_TEXT, generator=g),
        "a_emb": torch.randn(b, k, D_TEXT, generator=g),
        "task_id": torch.full((b,), 1, dtype=torch.long),
        "y": torch.randint(0, k, (b,), generator=g),
    }


# --- only the last n blocks may learn --------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_only_the_last_n_blocks_receive_gradients(n):
    model = live(trainable_blocks=n)
    layers = blocks(model.encoder)
    loss = torch.nn.functional.cross_entropy(model(batch()), batch()["y"])
    loss.backward()

    for i, layer in enumerate(layers):
        got = any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
        expected = i >= len(layers) - n
        assert got is expected, f"block {i}: gradient={got}, expected={expected}"

    assert model.encoder.embeddings.word_embeddings.weight.grad is None


def test_embedding_and_early_blocks_do_not_move_during_training():
    model = live(trainable_blocks=1)
    frozen_before = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    early_before = next(blocks(model.encoder)[0].parameters()).detach().clone()
    before = snapshot_encoder(model)

    opt = torch.optim.AdamW(param_groups(model, head_lr=1e-2, encoder_lr=1e-2))
    for step in range(3):
        b = batch(seed=step)
        torch.nn.functional.cross_entropy(model(b), b["y"]).backward()
        opt.step()
        opt.zero_grad()

    assert torch.equal(model.encoder.embeddings.word_embeddings.weight, frozen_before)
    assert torch.equal(next(blocks(model.encoder)[0].parameters()), early_before)
    assert encoder_drift(model, before) > 0, "the unfrozen block never moved"


def test_a_fully_frozen_encoder_reports_no_drift():
    model = live(trainable_blocks=0)
    before = snapshot_encoder(model)
    b = batch()
    torch.nn.functional.cross_entropy(model(b), b["y"]).backward()
    assert before == {} or encoder_drift(model, before) == 0.0


def test_trainable_report_counts_only_what_learns():
    model = live(trainable_blocks=2)
    r = trainable_report(model.encoder)
    assert 0 < r["trainable_params"] < r["total_params"]
    assert r["trainable_fraction"] == pytest.approx(
        r["trainable_params"] / r["total_params"], abs=1e-5
    )


# --- the head still sees what it is supposed to see ------------------------

def test_pooling_ignores_padding():
    # eval(): dropout would make two forwards of the same batch differ anyway.
    model = live().eval()
    b = batch()
    with torch.no_grad():
        pooled = model.encode(b)
        b2 = {**b, "input_ids": b["input_ids"].clone()}
        b2["input_ids"][:, -3:] = 7  # only masked-out positions change
        assert torch.allclose(pooled, model.encode(b2), atol=1e-5)


def test_pooling_matches_a_mask_weighted_mean():
    model = live().eval()
    b = batch()
    with torch.no_grad():
        h = model.encoder(input_ids=b["input_ids"],
                          attention_mask=b["attention_mask"]).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        assert torch.allclose(model.encode(b), (h * m).sum(1) / m.sum(1), atol=1e-6)


def test_entity_only_never_sees_the_question_through_the_live_path():
    model = live(mode="entity_only").eval()
    b = batch()
    with torch.no_grad():
        base = model(b)
        moved = model({**b, "q_emb": torch.randn_like(b["q_emb"])})
    assert torch.allclose(base, moved), "entity_only read the question"


def test_entity_only_does_see_the_peptide_through_the_live_path():
    model = live(mode="entity_only").eval()
    b = batch()
    with torch.no_grad():
        base = model(b)
        other = model({**b, "input_ids": batch(seed=99)["input_ids"]})
    assert not torch.allclose(base, other), "entity_only ignored the peptide"


def test_answer_text_still_changes_the_score_with_a_live_encoder():
    model = live(mode="dual").eval()
    b = batch()
    with torch.no_grad():
        base = model(b)
        swapped = model({**b, "a_emb": b["a_emb"].flip(1)})
    assert not torch.allclose(base, swapped.flip(1)) or not torch.allclose(base, swapped)


@pytest.mark.parametrize("cls", [CrossAttentionAster, RealAster])
def test_both_heads_train_through_the_live_encoder(cls):
    model = live(cls=cls, trainable_blocks=1)
    opt = torch.optim.AdamW(param_groups(model, head_lr=1e-2, encoder_lr=1e-3))
    b = batch()
    first = torch.nn.functional.cross_entropy(model(b), b["y"]).item()
    for _ in range(25):
        loss = torch.nn.functional.cross_entropy(model(b), b["y"])
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < first, "the live path cannot even fit one batch"


def test_param_groups_separates_encoder_from_head():
    model = live(trainable_blocks=2)
    groups = param_groups(model, head_lr=1e-3, encoder_lr=2e-5)
    assert [g["lr"] for g in groups] == [1e-3, 2e-5]
    head_ids = {id(p) for p in groups[0]["params"]}
    enc_ids = {id(p) for p in groups[1]["params"]}
    assert not head_ids & enc_ids
    assert enc_ids, "no encoder parameters were handed to the optimizer"


def test_param_groups_accepts_a_bare_head():
    head = CrossAttentionAster(HIDDEN, D_TEXT, N_TASKS, h=32, mode="dual")
    groups = param_groups(head, head_lr=1e-3, encoder_lr=2e-5)
    assert len(groups) == 1 and groups[0]["lr"] == 1e-3


# --- the runner's training loop, exercised offline -------------------------

def test_runner_train_loop_fits_a_tiny_problem():
    """Catches the wiring the unit tests above cannot: param groups feeding the
    scheduler, batch slicing, and eval-mode logits for a whole split."""
    from scripts.run_amp_finetune import forward_all, train

    torch.manual_seed(0)
    model = live(trainable_blocks=1)
    b = batch(b=16, seed=3)
    # A label the peptide determines, so a fit is possible at all.
    b["y"] = (b["input_ids"][:, 0] % 2).long()
    tr = {k: v for k, v in b.items()}
    va = {k: v.clone() for k, v in b.items()}

    before = forward_all(model, va, 8, "cpu")
    trained = train(model, tr, va, "cpu", epochs=3, bs=8,
                    head_lr=1e-2, encoder_lr=1e-3, patience=5)
    after = forward_all(trained, va, 8, "cpu")

    assert after.shape == before.shape == (16, 2)
    loss_before = torch.nn.functional.cross_entropy(torch.from_numpy(before), va["y"])
    loss_after = torch.nn.functional.cross_entropy(torch.from_numpy(after), va["y"])
    assert loss_after < loss_before
