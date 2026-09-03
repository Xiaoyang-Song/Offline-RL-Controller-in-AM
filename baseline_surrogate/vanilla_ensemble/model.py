"""
baseline_surrogate/vanilla_ensemble/model.py
--------------------------------------------------
Baseline 4: vanilla deep ensemble. Same learned-latent skeleton as the main
surrogate (shared Encoder/Decoder, imported directly from
surrogate_model_latent_uncertainty_v2.model — no need to reimplement), but:
  - single-stage (one transition per layer, conditioned on [a_t, cool_t]
    together — see baseline_surrogate/common/data.py's module docstring for
    why every external baseline here collapses the heat/cool split)
  - K=5 members, each independently initialised (different random init,
    NOT bootstrap-resampled — every member trains on the exact same full
    dataset) and MSE-only (no Gaussian NLL, no per-member sigma head)
  - prediction is the decoded ensemble MEAN, uncertainty is not reported
    (this repo's uncertainty quantification IS the main surrogate's
    contribution, not something a baseline needs to also provide)

This isolates exactly one variable against the main surrogate: does
bootstrap resampling + a Gaussian-NLL-calibrated sigma head buy anything
over plain ensembling (independent init + MSE) at matched architecture?
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.model import Encoder, Decoder, ResidualBlock


class DeterministicTransitionMLP(nn.Module):
    """Point-estimate transition head: (z, cond, layer_embed) -> Delta-z.
    Same trunk shape as GaussianTransitionMLP but with only a mu head —
    no log-sigma head, no PETS soft clamp (nothing to calibrate)."""

    def __init__(
        self, latent_dim: int, cond_dim: int = 2, hidden: int = 128, depth: int = 3,
        n_layers: int = 12, layer_embed_dim: int = 8, dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)
        in_dim = latent_dim + cond_dim + layer_embed_dim
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.trunk = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head  = nn.Linear(hidden, latent_dim)
        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, layer_idx: torch.Tensor) -> torch.Tensor:
        e = self.layer_embed(layer_idx)
        x = torch.cat([z, cond, e], dim=-1)
        return self.head(self.trunk(self.input_proj(x)))


class VanillaDeepEnsembleSurrogate(nn.Module):
    def __init__(
        self,
        state_dim:        int   = 1053,
        latent_dim:       int   = 64,
        n_ensemble:       int   = 5,
        n_layers:         int   = 12,
        layer_embed_dim:  int   = 8,
        enc_hidden:       int   = 256,
        enc_depth:        int   = 3,
        trans_hidden:     int   = 128,
        trans_depth:      int   = 3,
        dec_hidden:       int   = 256,
        dec_depth:        int   = 3,
        dropout:          float = 0.0,
        member_init_seed: int   = None,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.latent_dim = latent_dim
        self.n_ensemble = n_ensemble

        self.encoder = Encoder(state_dim, latent_dim, enc_hidden, enc_depth, dropout)
        self.decoder = Decoder(latent_dim, state_dim, dec_hidden, dec_depth, dropout)

        rng_state = torch.get_rng_state() if member_init_seed is not None else None
        members = []
        for k in range(n_ensemble):
            if member_init_seed is not None:
                torch.manual_seed(member_init_seed + k)
            members.append(DeterministicTransitionMLP(
                latent_dim, cond_dim=2, hidden=trans_hidden, depth=trans_depth,
                n_layers=n_layers, layer_embed_dim=layer_embed_dim, dropout=dropout,
            ))
        self.transitions = nn.ModuleList(members)
        if rng_state is not None:
            torch.set_rng_state(rng_state)

    def forward_all_members(self, s, a, c, layer_idx):
        """s,a,c normalised. Returns (preds (K, B, state_dim) decoded, z_t (B, latent_dim))."""
        z_t  = self.encoder(s)
        cond = torch.cat([a, c], dim=-1)
        preds = [self.decoder(z_t + m(z_t, cond, layer_idx)) for m in self.transitions]
        return torch.stack(preds, dim=0), z_t

    @torch.no_grad()
    def predict_mean(self, s, a, c, layer_idx) -> torch.Tensor:
        preds, _ = self.forward_all_members(s, a, c, layer_idx)
        return preds.mean(dim=0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
