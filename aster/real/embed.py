"""ESM-2 and text embeddings, cached to disk. Tuned for Apple Silicon (MPS).

Embeddings are computed once and cached, because on an M3 the encoder pass is
the expensive part and every model variant in the experiment reuses the exact
same vectors. Caching also keeps the comparison honest -- every model sees
identical inputs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

# Apple Silicon: let unsupported ops fall back to CPU instead of hard-failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ESM_SIZES = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}


def pick_device(pref="auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _key(items, model, extra="") -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(extra.encode())
    for s in items:
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def embed_sequences(seqs, model="35M", device="auto", batch_size=16,
                    cache_dir="~/.cache/aster", max_len=256, verbose=True):
    """Mean-pooled ESM-2 embeddings, one row per sequence. Cached by content."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = ESM_SIZES.get(model, model)
    uniq = sorted(set(seqs))
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"esm_{_key(uniq, name, str(max_len))}.npz"

    if f.exists():
        z = np.load(f, allow_pickle=True)
        table = {s: v for s, v in zip(z["seqs"], z["emb"])}
        if verbose:
            print(f"  esm cache hit: {f.name} ({len(table)} sequences)")
        return np.stack([table[s] for s in seqs])

    dev = pick_device(device)
    if verbose:
        print(f"  embedding {len(uniq)} unique sequences with {name} on {dev}")
        if dev == "mps":
            print("  (Apple Silicon GPU; first run is slow, then cached)")

    tok = AutoTokenizer.from_pretrained(name)
    net = AutoModel.from_pretrained(name).to(dev).eval()

    # Sort by length so each batch pads to a similar size -- a large win on MPS.
    order = sorted(range(len(uniq)), key=lambda i: len(uniq[i]))
    out = [None] * len(uniq)

    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            batch = [uniq[i] for i in idx]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = net(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            for j, i in enumerate(idx):
                out[i] = pooled[j].float().cpu().numpy()
            if verbose and (b // batch_size) % 40 == 0:
                print(f"    {min(b + batch_size, len(order))}/{len(order)}",
                      flush=True)

    emb = np.stack(out).astype(np.float32)
    np.savez_compressed(f, seqs=np.array(uniq, dtype=object), emb=emb)
    if verbose:
        print(f"  cached -> {f}")
    table = dict(zip(uniq, emb))
    return np.stack([table[s] for s in seqs])


def embed_texts(texts, model="prajjwal1/bert-tiny", device="auto",
                cache_dir="~/.cache/aster", verbose=True):
    """Mean-pooled sentence embeddings for questions and answer phrases.

    Frozen and computed offline. Cached output vectors cannot carry gradients
    back to the encoder; fine-tuning needs a separate training-time forward pass.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    uniq = sorted(set(texts))
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"txt_{_key(uniq, model)}.npz"

    if f.exists():
        z = np.load(f, allow_pickle=True)
        table = {s: v for s, v in zip(z["seqs"], z["emb"])}
        if verbose:
            print(f"  text cache hit: {f.name} ({len(table)} strings)")
        return np.stack([table[t] for t in texts])

    dev = pick_device(device)
    if verbose:
        print(f"  embedding {len(uniq)} strings with {model} on {dev}")
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModel.from_pretrained(model).to(dev).eval()

    with torch.no_grad():
        enc = tok(uniq, return_tensors="pt", padding=True, truncation=True,
                  max_length=64)
        enc = {k: v.to(dev) for k, v in enc.items()}
        h = net(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        emb = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()

    emb = emb.astype(np.float32)
    np.savez_compressed(f, seqs=np.array(uniq, dtype=object), emb=emb)
    table = dict(zip(uniq, emb))
    return np.stack([table[t] for t in texts])
