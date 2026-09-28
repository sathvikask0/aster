"""Both AMP runners, end to end, offline.

The unit tests cover the pieces; this covers the wiring between them -- argument
handling, tensor construction, the ceiling pass, the mechanism-swap call and the
report it writes. It runs with composition vectors and a stub text encoder, so it
needs no downloaded weights, which is also why it can run anywhere.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

D_TEXT = 24


def fake_embed_texts(texts, model=None, device=None, cache_dir=None, verbose=False):
    """Deterministic pseudo-embeddings: distinct per text, stable across calls."""
    out = np.zeros((len(texts), D_TEXT), dtype=np.float32)
    for i, t in enumerate(texts):
        digest = hashlib.sha256(t.encode()).digest()
        out[i] = np.frombuffer(digest[:D_TEXT], dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return out


@pytest.fixture
def amp_data(tmp_path):
    """A small benchmark with three prompt families, so a swap has a donor."""
    rng = np.random.default_rng(0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    peptides = ["".join(rng.choice(list(alphabet), size=int(rng.integers(8, 20))))
                for _ in range(120)]
    tasks = {
        "e_coli": ("Gram-negative", "A rod-shaped bacterium."),
        "s_aureus": ("Gram-positive", "A cocci."),
        "a_flavus": ("Fungus", "A mold."),
        "k_pneumoniae": ("Gram-negative", "Another rod."),
        "c_albicans": ("Fungus", "A yeast."),
    }
    rows = ["sequence,pathogen"]
    for i, (task, _) in enumerate(tasks.items()):
        # Overlapping but distinct actives per task, so negatives exist everywhere.
        for s in peptides[i * 12: i * 12 + 40]:
            rows.append(f"{s},{task}")
    (tmp_path / "peptide_pathogen_triple.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "pathogen_description.csv").write_text(
        "pathogen,types,description\n"
        + "".join(f"{t},{ty},{d}\n" for t, (ty, d) in tasks.items())
    )
    return tmp_path


def _patch(monkeypatch, module, amp_data):
    from aster.real.amp import load_amp_benchmark

    def loader(**kwargs):
        kwargs.pop("min_samples", None)
        kwargs.pop("data_dir", None)
        return load_amp_benchmark(
            data_dir=amp_data, min_samples=1, min_coverage=1,
            test_tasks=("k_pneumoniae", "c_albicans"), **kwargs
        )

    monkeypatch.setattr(module, "load_amp_benchmark", loader)
    monkeypatch.setattr(module, "embed_texts", fake_embed_texts)


def test_frozen_runner_writes_a_readable_report(monkeypatch, amp_data, tmp_path):
    import scripts.run_amp_multitask as runner

    _patch(monkeypatch, runner, amp_data)
    out = tmp_path / "frozen.json"
    monkeypatch.setattr(sys, "argv", [
        "run_amp_multitask.py", "--esm", "none", "--epochs", "2",
        "--negative-policy", "matched", "--balance-tasks",
        "--device", "cpu", "--out", str(out),
    ])
    runner.main()

    report = json.loads(out.read_text())
    assert report["config"]["label_semantics_version"] == 2
    assert report["config"]["negative_policy"] == "matched"
    assert set(report["held_out_tasks"]) == {"k_pneumoniae", "c_albicans"}
    assert report["mechanism_swap_map"], "no swap assignment was recorded"

    for task, donor in report["mechanism_swap_map"].items():
        assert report["prompt_families"][donor] != report["prompt_families"][task]

    # task_id must stay structurally at chance on an unseen task.
    assert report["results"]["task_id"]["overall_accuracy"] == pytest.approx(0.5, abs=0.02)

    # The shortcut-blind controls get no swap; the question-reading ones do.
    assert "mechanism_swap" not in report["results"]["entity_only"]
    assert "mechanism_swap" not in report["results"]["task_id"]
    for name in ("cross_attention", "dual_dot", "question_only"):
        ms = report["results"][name]["mechanism_swap"]
        assert 0.0 <= ms["own_prompt_accuracy"] <= 1.0
        assert ms["drop"] == pytest.approx(
            ms["own_prompt_accuracy"] - ms["swapped_prompt_accuracy"], abs=1e-9
        )
        assert ms["reading"]

    # question_only ignores the peptide, so its swap is a pure prompt effect.
    assert "per_task_own" in report["results"]["question_only"]["mechanism_swap"]


def _stubs(monkeypatch, runner):
    """Tiny random encoders of the right classes, so no weights are downloaded."""
    import torch
    from transformers import BertConfig, BertModel, EsmConfig, EsmModel

    from aster.model import unfreeze_suffix

    class StubTokenizer:
        def __call__(self, seqs, return_tensors=None, padding=None, truncation=None,
                     max_length=32):
            ids = torch.ones(len(seqs), max_length, dtype=torch.long)
            mask = torch.zeros(len(seqs), max_length, dtype=torch.long)
            for i, s in enumerate(seqs):
                n = max(1, min(len(s), max_length))
                ids[i, :n] = torch.tensor([4 + (ord(c) % 20) for c in s[:n]])
                mask[i, :n] = 1
            return {"input_ids": ids, "attention_mask": mask}

    def stub_protein(model="8M", trainable_blocks=2, device="cpu", revision=None):
        cfg = EsmConfig(
            vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, max_position_embeddings=64, pad_token_id=1,
            mask_token_id=32, token_dropout=False, position_embedding_type="absolute",
        )
        enc = EsmModel(cfg, add_pooling_layer=False)
        if trainable_blocks:
            unfreeze_suffix(enc, min(trainable_blocks, 2))
        return enc.to(device), StubTokenizer()

    def stub_text(model, trainable_blocks=1, device="cpu", revision=None):
        cfg = BertConfig(vocab_size=40, hidden_size=D_TEXT, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=32,
                         max_position_embeddings=128)
        enc = BertModel(cfg, add_pooling_layer=False)
        if trainable_blocks:
            unfreeze_suffix(enc, min(trainable_blocks, 2))
        return enc.to(device), StubTokenizer()

    monkeypatch.setattr(runner, "load_protein_encoder", stub_protein)
    monkeypatch.setattr(runner, "load_text_encoder", stub_text)


@pytest.mark.parametrize("cached", [False, True])
def test_finetune_runner_runs_with_a_stub_encoder(monkeypatch, amp_data, tmp_path, cached):
    """The live path, minus the download: a tiny random ESM of the same class."""
    import torch
    from transformers import EsmConfig, EsmModel

    import scripts.run_amp_finetune as runner
    from aster.model import unfreeze_suffix

    _patch(monkeypatch, runner, amp_data)
    _stubs(monkeypatch, runner)

    out = tmp_path / "finetune.json"
    argv = [
        "run_amp_finetune.py", "--epochs", "1", "--batch-size", "16",
        "--trainable-blocks", "1", "--max-len", "32", "--device", "cpu",
        "--out", str(out),
    ]
    argv += ["--cache-frozen-prefix"] if cached else ["--skip-frozen-reference"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()

    report = json.loads(out.read_text())
    cfg = report["config"]
    assert cfg["label_semantics_version"] == 2
    assert cfg["negative_policy"] == "matched" and cfg["balance_tasks"] is True
    assert cfg["text_encoder_trainable"] is False
    assert cfg["frozen_reference_included"] is cached
    assert cfg["training_protocol_version"] == 2
    if cached:
        assert cfg["frozen_reference_epochs"] == cfg["epochs"] == 1
        assert cfg["frozen_reference_batch_size"] == cfg["batch_size"] == 16
        assert "dual_frozen" in report["results"]
    assert cfg["encoder_trainable"]["trainable_params"] > 0

    for name in ("cross_attention_live", "dual_live", "entity_only_live"):
        res = report["results"][name]
        assert res["encoder"]["live"] is True
        assert "encoder_drift" in res
    # The control must not be handed the question, even here.
    assert "mechanism_swap" not in report["results"]["entity_only_live"]
    assert report["results"]["cross_attention_live"]["mechanism_swap"]["reading"]


def test_unfreeze_text_puts_both_towers_in_the_graph(monkeypatch, amp_data, tmp_path):
    """Both-towers-live: the control on the text side must appear, and the
    mechanism swap must still work when the question is an index, not a vector."""
    import scripts.run_amp_finetune as runner

    _patch(monkeypatch, runner, amp_data)
    _stubs(monkeypatch, runner)

    out = tmp_path / "both.json"
    monkeypatch.setattr(sys, "argv", [
        "run_amp_finetune.py", "--epochs", "1", "--batch-size", "16",
        "--trainable-blocks", "1", "--max-len", "32", "--device", "cpu",
        "--unfreeze-text", "--text-trainable-blocks", "1", "--text-lr", "1e-4",
        "--skip-frozen-reference", "--out", str(out),
    ])
    runner.main()

    report = json.loads(out.read_text())
    cfg = report["config"]
    assert cfg["text_encoder_trainable"] is True
    assert cfg["text_trainable_blocks"] == 1
    assert cfg["text_lr"] == pytest.approx(1e-4)
    assert cfg["text_encoder_trainable_report"]["trainable_params"] > 0
    assert set(cfg["live_text_modes"]) == {"dual", "question_only"}

    # The text-side control must exist, or the hypothesis model is compared
    # against a text tower that was denied the same capacity.
    assert "question_only_live" in report["results"]

    text_live = {n: r["encoder"]["text_live"] for n, r in report["results"].items()}
    assert text_live["cross_attention_live"] is True
    assert text_live["dual_live"] is True
    assert text_live["question_only_live"] is True
    assert text_live["entity_only_live"] is False, "entity_only reads no text"

    # Both towers must have moved, and be reported apart.
    hyp = report["results"]["cross_attention_live"]
    assert hyp["encoder_drift"] > 0
    assert hyp["text_encoder_drift"] > 0
    assert report["results"]["entity_only_live"]["text_encoder_drift"] == 0

    # The swap goes through q_idx on this path; it must still produce a reading.
    ms = hyp["mechanism_swap"]
    assert ms["reading"]
    assert 0.0 <= ms["swapped_prompt_accuracy"] <= 1.0
