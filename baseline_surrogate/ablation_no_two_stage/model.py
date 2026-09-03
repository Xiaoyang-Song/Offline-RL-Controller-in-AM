"""
baseline_surrogate/ablation_no_two_stage/model.py
--------------------------------------------------------
Ablation 1 of the MAIN surrogate (surrogate_model_latent_uncertainty_v2):
same learned latent space + bootstrap-trained Gaussian ensemble machinery
(GaussianTransitionMLP, moment matching — imported directly, not
reimplemented), but collapsed to ONE single-stage transition per layer
(a_t and cool_t concatenated into one 2-dim conditioning vector) instead of
the explicit heating-then-cooling split. Holding latent space + ensemble/
NLL fixed and only removing the two-stage decomposition isolates exactly
what that decomposition buys.

Unlike every OTHER baseline in this package, this one keeps the full
bootstrap + Gaussian-NLL machinery on purpose — the point of this specific
ablation is "the main method minus two-stage," not "the main method minus
uncertainty," so uncertainty stays in (even though, like every baseline
here, only the point-estimate prediction is reported/compared).
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.model import Encoder, Decoder, GaussianTransitionMLP, _moment_match


class NoTwoStageSurrogate(nn.Module):
    def __init__(
        self,
        state_dim:        int   = 1053,
        cond_dim:         int   = 2,     # [a_t, cool_t] concatenated
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
        mu_init_scale:    float = 1e-3,
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
            members.append(GaussianTransitionMLP(
                latent_dim, cond_dim, trans_hidden, trans_depth, dropout,
                n_layers, layer_embed_dim, mu_init_scale=mu_init_scale,
            ))
        self.transitions = nn.ModuleList(members)
        if rng_state is not None:
            torch.set_rng_state(rng_state)

    def _run(self, z, cond, layer_idx):
        mus, log_sigmas = [], []
        for t in self.transitions:
            mu, log_sigma = t(z, cond, layer_idx)
            mus.append(mu)
            log_sigmas.append(log_sigma)
        return torch.stack(mus, dim=0), torch.stack(log_sigmas, dim=0)

    def forward(self, s, cond, layer_idx):
        """s: (B, state_dim) normalised. cond: (B, 2) = [a_t, cool_t] normalised."""
        z_t = self.encoder(s)
        s_recon = self.decoder(z_t)
        mu, log_sigma = self._run(z_t, cond, layer_idx)
        return dict(z_t=z_t, s_recon=s_recon, mu=mu, log_sigma=log_sigma)

    @torch.no_grad()
    def predict_mean(self, s, a, c, layer_idx) -> torch.Tensor:
        cond = torch.cat([a, c], dim=-1)
        z_t  = self.encoder(s)
        mu, log_sigma = self._run(z_t, cond, layer_idx)
        mu_mean, *_ = _moment_match(mu, log_sigma)
        return self.decoder(z_t + mu_mean)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
