"""Joint protein/text encoders with a small, trainable cell-state baseline.

No calibrated-probability or pretrained-cell-encoder claims are made here.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from transformers import AutoModel


@dataclass
class AsterConfig:
    protein_model: str = 'facebook/esm2_t6_8M_UR50D'
    text_model: str = 'prajjwal1/bert-tiny'
    protein_revision: str | None = None
    text_revision: str | None = None
    protein_last_layers: int = 2
    text_last_layers: int = 1
    cell_features: int = 1000
    latent_dim: int = 128


def blocks(model):
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    if hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f'Unsupported encoder layout: {type(model).__name__}')


def unfreeze_suffix(model, n):
    layers = blocks(model)
    if not 0 < n <= len(layers):
        raise ValueError(f'Expected 1..{len(layers)} trainable layers, got {n}')
    model.requires_grad_(False)
    for layer in layers[-n:]:
        layer.requires_grad_(True)
    norm = getattr(getattr(model, 'encoder', None), 'emb_layer_norm_after', None)
    if norm is None:
        norm = getattr(model, 'final_norm', None)
    if norm is not None:
        norm.requires_grad_(True)


def pooled(model, tokens):
    result = model(**tokens).last_hidden_state
    # Callers provide mask without CLS/EOS for protein pooling if desired.
    mask = tokens['attention_mask'].unsqueeze(-1).to(result.dtype)
    return (result * mask).sum(1) / mask.sum(1).clamp_min(1)


class AsterModel(nn.Module):
    def __init__(self, config, protein=None, text=None):
        super().__init__()
        self.config = config
        self.protein = protein if protein is not None else AutoModel.from_pretrained(
            config.protein_model, revision=config.protein_revision, attn_implementation='eager')
        self.text = text if text is not None else AutoModel.from_pretrained(
            config.text_model, revision=config.text_revision, attn_implementation='eager')
        unfreeze_suffix(self.protein, config.protein_last_layers)
        unfreeze_suffix(self.text, config.text_last_layers)
        d = config.latent_dim
        ph = self.protein.config.hidden_size
        th = self.text.config.hidden_size
        # A learned cell-state baseline, not Arc SE or scGPT.
        self.cell = nn.Sequential(nn.Linear(config.cell_features, d), nn.GELU(), nn.LayerNorm(d))
        self.state = nn.Sequential(nn.Linear(2 * ph + d, d), nn.GELU(), nn.LayerNorm(d))
        self.option = nn.Sequential(nn.Linear(th, d), nn.GELU(), nn.LayerNorm(d))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.2)))

    def train(self, mode=True):
        super().train(mode)
        # Deterministic frozen prefix; eval does not turn off gradients in the suffix.
        self.protein.eval()
        self.text.eval()
        return self

    def forward(self, intervention, readout, cell_state, options, option_mask=None):
        """Options are token tensors shaped [batch, choices, tokens].

        Each option includes the question, assay context and candidate answer.
        Proteins are token tensors [batch, tokens]. No final embeddings are cached.
        """
        if cell_state.ndim != 2 or cell_state.shape[1] != self.config.cell_features:
            raise ValueError('cell_state has incompatible feature shape')
        b, k, length = options['input_ids'].shape
        if b != cell_state.shape[0] or k < 2:
            raise ValueError('Expected at least two choices per input')
        z = torch.cat([pooled(self.protein, intervention), pooled(self.protein, readout), self.cell(cell_state)], -1)
        q = nn.functional.normalize(self.state(z), dim=-1)
        flat = {key: value.reshape(b * k, length) for key, value in options.items()}
        v = nn.functional.normalize(self.option(pooled(self.text, flat)), dim=-1).reshape(b, k, -1)
        temperature = self.log_temperature.exp().clamp(0.02, 2.0)
        logits = torch.einsum('bd,bkd->bk', q, v) / temperature
        if option_mask is not None:
            if option_mask.shape != (b, k) or not option_mask.bool().any(1).all():
                raise ValueError('Every input needs at least one valid option')
            logits = logits.masked_fill(~option_mask.bool(), -torch.inf)
        return logits

    @torch.no_grad()
    def decide(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        try:
            p = self(*args, **kwargs).softmax(-1)
            return {'choice': p.argmax(-1).cpu().tolist(), 'probabilities': p.cpu().tolist(),
                    'calibrated': False, 'status': 'experimental'}
        finally:
            self.train(was_training)

    def save(self, path):
        torch.save({'config': asdict(self.config), 'state_dict': self.state_dict()}, path)
