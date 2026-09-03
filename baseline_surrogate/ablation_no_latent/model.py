"""
baseline_surrogate/ablation_no_latent/model.py
--------------------------------------------------
Ablation 2 of the MAIN surrogate (surrogate_model_latent_uncertainty_v2):
the same two-stage (heating -> cooling) bootstrap-trained Gaussian ensemble
(GaussianTransitionMLP + moment matching — imported, not reimplemented),
but the encoder/decoder are the IDENTITY — transitions act directly on the
raw (normalised) 1053-dim state instead of a learned latent. Holding the
two-stage decomposition + ensemble/NLL machinery fixed and only removing
the latent bottleneck isolates exactly what that bottleneck buys.

Consequently `encode`/`decode` are the identity, so the main model's AE
reconstruction losses (L_recon_s, L_recon_heat_ae) are trivially zero here
and are dropped from this ablation's loss (see train.py) — nothing to
learn, decoder(encoder(x)) == x by construction.

This needs the TRUE two-stage target (u_heat_t) since "keep two-stage,
remove latent" is exactly what it tests — so, unlike every other baseline
in this package, its train.py imports TwoStageLatentSurrogateDataset /
build_normalizers directly from surrogate_model_latent_uncertainty_v2
rather than anything in baseline_surrogate/common/data.py.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.model import GaussianTransitionMLP, _moment_match


class NoLatentTwoStageSurrogate(nn.Module):
    def __init__(
        self,
        state_dim:        int   = 1053,
        lp_dim:            int  = 1,
        cool_dim:          int  = 1,
        n_ensemble:        int  = 5,
        n_layers:          int  = 12,
        layer_embed_dim:   int  = 8,
        trans_hidden:      int  = 128,
        trans_depth:       int  = 3,
        dropout:           float = 0.0,
        mu_init_scale:     float = 1e-3,
        member_init_seed:  int   = None,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.n_ensemble = n_ensemble

        rng_state = torch.get_rng_state() if member_init_seed is not None else None

        def _build(cond_dim, seed_offset):
            members = []
            for k in range(n_ensemble):
                if member_init_seed is not None:
                    torch.manual_seed(member_init_seed + seed_offset + k)
                members.append(GaussianTransitionMLP(
                    state_dim, cond_dim, trans_hidden, trans_depth, dropout,
                    n_layers, layer_embed_dim, mu_init_scale=mu_init_scale,
                ))
            return nn.ModuleList(members)

        self.heating_transitions = _build(lp_dim, 0)
        self.cooling_transitions = _build(cool_dim, n_ensemble)
        if rng_state is not None:
            torch.set_rng_state(rng_state)

    @staticmethod
    def encode(s: torch.Tensor) -> torch.Tensor:
        return s

    @staticmethod
    def decode(z: torch.Tensor) -> torch.Tensor:
        return z

    def _run(self, transitions, z, c, layer_idx):
        mus, log_sigmas = [], []
        for t in transitions:
            mu, log_sigma = t(z, c, layer_idx)
            mus.append(mu)
            log_sigmas.append(log_sigma)
        return torch.stack(mus, dim=0), torch.stack(log_sigmas, dim=0)

    def forward(self, s_t, a_t, cool_t, u_heat_t, layer_idx):
        """Mirrors TwoStageEnsembleGaussianLatentDynamicsModel.forward, minus
        the (here-trivial) AE reconstruction terms."""
        z_t = s_t
        mu_heat, log_sigma_heat = self._run(self.heating_transitions, z_t, a_t, layer_idx)
        z_heat_enc = u_heat_t
        mu_cool, log_sigma_cool = self._run(self.cooling_transitions, z_heat_enc, cool_t, layer_idx)
        return dict(
            z_t=z_t, mu_heat=mu_heat, log_sigma_heat=log_sigma_heat,
            z_heat_enc=z_heat_enc, mu_cool=mu_cool, log_sigma_cool=log_sigma_cool,
        )

    @torch.no_grad()
    def predict_mean(self, s, a, c, layer_idx) -> torch.Tensor:
        """Single-stage-compatible predict: chains heating then cooling,
        returns the final s_{t+1} point estimate (for common/eval.py)."""
        z_t = s
        mu_heat, log_sigma_heat = self._run(self.heating_transitions, z_t, a, layer_idx)
        mu_heat_mean, *_ = _moment_match(mu_heat, log_sigma_heat)
        z_heat = z_t + mu_heat_mean
        mu_cool, log_sigma_cool = self._run(self.cooling_transitions, z_heat, c, layer_idx)
        mu_cool_mean, *_ = _moment_match(mu_cool, log_sigma_cool)
        return z_heat + mu_cool_mean

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
