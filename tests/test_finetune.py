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
    LiveAster, encoder_drift, param_groups, snapshot_encoder, trainable_report,
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
    return LiveAster(head, enc)


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


def test_cached_frozen_prefix_preserves_outputs_and_suffix_gradients():
    """Caching must not change the learned computation, including rotary ESM."""
    import copy
    from aster.model import unfreeze_suffix
    from aster.real.finetune import cache_frozen_prefix

    torch.manual_seed(123)
    cfg = EsmConfig(
        vocab_size=33, hidden_size=32, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=64, pad_token_id=1, mask_token_id=32,
        token_dropout=True, position_embedding_type="rotary",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    enc = EsmModel(cfg, add_pooling_layer=False)
    unfreeze_suffix(enc, 2)
    original = LiveAster(RealAster(32, D_TEXT, N_TASKS, h=32, p=0), enc)
    cached = copy.deepcopy(original)
    b = batch(b=5)
    b["input_ids"][b["attention_mask"] == 0] = 1
    states = cache_frozen_prefix(cached.encoder, b["input_ids"],
                                 b["attention_mask"], 2, batch_size=2)
    assert states.device.type == "cpu" and not states.requires_grad
    cached.prefix_states, cached.trainable_blocks = states, 2
    # Repeated, reordered sequences must gather the correct cache rows.
    idx = torch.tensor([3, 0, 3, 4])
    selected = {k: v[idx] for k, v in b.items()}
    cached_batch = {**selected, "prefix_idx": idx}
    original.train()
    cached.train()
    direct = original(selected)
    indirect = cached(cached_batch)
    torch.testing.assert_close(direct, indirect, atol=1e-6, rtol=1e-5)
    for model, logits in ((original, direct), (cached, indirect)):
        torch.nn.functional.cross_entropy(logits, selected["y"]).backward()
    for (name, p), (_, q) in zip(original.named_parameters(), cached.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
        else:
            assert p.grad is None and q.grad is None


def test_prefix_cache_rejects_trainable_prefix():
    from aster.real.finetune import cache_frozen_prefix

    enc = encoder(trainable_blocks=3)
    b = batch()
    with pytest.raises(ValueError, match="trainable parameters"):
        cache_frozen_prefix(enc, b["input_ids"], b["attention_mask"], 2)


def test_small_checkpoint_restores_trainable_weights_and_preserves_frozen_weights():
    from aster.real.finetune import snapshot_trainable_state

    model = live()
    saved = snapshot_trainable_state(model)
    full = {n: p.detach().clone() for n, p in model.named_parameters()}
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert not frozen_names.intersection(saved)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(1)
    model.load_state_dict(saved, strict=False)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, full[n], rtol=0, atol=0)


# --- the text tower, when it is also live ----------------------------------

def text_encoder(trainable_blocks=1):
    from transformers import BertConfig, BertModel

    from aster.model import unfreeze_suffix

    cfg = BertConfig(vocab_size=40, hidden_size=D_TEXT, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=64)
    enc = BertModel(cfg, add_pooling_layer=False)
    if trainable_blocks:
        unfreeze_suffix(enc, trainable_blocks)
    return enc


def text_tables(n_questions=4, n_answers=2, length=6):
    from aster.real.finetune import TextTable

    g = torch.Generator().manual_seed(11)
    def table(n):
        ids = torch.randint(4, 40, (n, length), generator=g)
        mask = torch.ones(n, length, dtype=torch.long)
        return TextTable(ids, mask)
    return table(n_questions), table(n_answers)


def live_text(mode="dual", trainable_blocks=1, text_blocks=1):
    from aster.real.finetune import LiveAster

    q, a = text_tables()
    head = CrossAttentionAster(HIDDEN, D_TEXT, N_TASKS, h=32, mode=mode)
    return LiveAster(head, encoder(trainable_blocks),
                     text_encoder=text_encoder(text_blocks), questions=q, answers=a)


def index_batch(b=6, k=2, seed=0, length=12, n_questions=4, n_answers=2):
    base = batch(b=b, k=k, seed=seed, length=length)
    g = torch.Generator().manual_seed(seed + 1)
    base["q_idx"] = torch.randint(0, n_questions, (b,), generator=g)
    base["a_idx"] = torch.stack([torch.arange(k) % n_answers for _ in range(b)])
    return base


def test_a_live_text_encoder_requires_its_tables():
    from aster.real.finetune import LiveAster

    with pytest.raises(ValueError, match="question and answer tables"):
        LiveAster(CrossAttentionAster(HIDDEN, D_TEXT, N_TASKS, h=32),
                  encoder(1), text_encoder=text_encoder(1))


def test_live_text_refuses_cached_embeddings():
    model = live_text()
    b = batch()  # carries q_emb/a_emb but no q_idx/a_idx
    with pytest.raises(KeyError, match="cannot carry gradients"):
        model(b)


def test_cached_question_embeddings_are_ignored_once_text_is_live():
    model = live_text().eval()
    b = index_batch()
    with torch.no_grad():
        base = model(b)
        # Poisoning the cached vectors must change nothing: they are unused now.
        other = model({**b, "q_emb": torch.randn_like(b["q_emb"]) * 50,
                       "a_emb": torch.randn_like(b["a_emb"]) * 50})
    assert torch.allclose(base, other)


def test_the_question_index_does_change_the_score():
    model = live_text().eval()
    b = index_batch()
    with torch.no_grad():
        base = model(b)
        moved = model({**b, "q_idx": (b["q_idx"] + 1) % 4})
    assert not torch.allclose(base, moved)


def test_only_the_last_text_blocks_receive_gradients():
    model = live_text(text_blocks=1)
    layers = blocks(model.text_encoder)
    b = index_batch()
    torch.nn.functional.cross_entropy(model(b), b["y"]).backward()

    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in layers[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in layers[-1].parameters())
    assert model.text_encoder.embeddings.word_embeddings.weight.grad is None


def test_both_towers_move_and_are_reported_separately():
    model = live_text(trainable_blocks=1, text_blocks=1)
    protein_before = snapshot_encoder(model)
    text_before = snapshot_encoder(model, "text_encoder")
    opt = torch.optim.AdamW(param_groups(model, head_lr=1e-2, encoder_lr=1e-3,
                                         text_lr=1e-3))
    for step in range(3):
        b = index_batch(seed=step)
        torch.nn.functional.cross_entropy(model(b), b["y"]).backward()
        opt.step()
        opt.zero_grad()

    assert encoder_drift(model, protein_before) > 0
    assert encoder_drift(model, text_before, "text_encoder") > 0


def test_param_groups_gives_the_text_tower_its_own_rate():
    model = live_text()
    groups = param_groups(model, head_lr=1e-3, encoder_lr=2e-5, text_lr=5e-6)
    assert [g["lr"] for g in groups] == [1e-3, 2e-5, 5e-6]
    ids = [{id(p) for p in g["params"]} for g in groups]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def test_text_lr_defaults_to_the_encoder_rate():
    model = live_text()
    groups = param_groups(model, head_lr=1e-3, encoder_lr=2e-5)
    assert [g["lr"] for g in groups] == [1e-3, 2e-5, 2e-5]


def test_protein_only_model_reports_no_text_drift():
    model = live()
    assert model.text_is_live is False
    assert snapshot_encoder(model, "text_encoder") == {}
    assert encoder_drift(model, {}, "text_encoder") == 0.0
