"""
baseline_surrogate/vanilla_ensemble/model.py
--------------------------------------------------
Baseline 4: vanilla deep ensemble. K=5 independently initialised copies of
the SAME plain-MLP architecture as baseline_surrogate/mlp/model.py — no
learned latent space, no layer embedding, no residual/delta prediction (see
that module's docstring for why: none of these are baseline concerns, they
are techniques surrogate_model_v3 itself proposes, and a baseline shouldn't
borrow them). The only thing this baseline adds over the plain MLP is the
ensemble itself:
  - K members, each independently initialised (different random init, NOT
    bootstrap-resampled — every member trains on the exact same full dataset)
  - MSE-only (no Gaussian NLL, no per-member sigma head)
  - prediction is the ensemble MEAN; uncertainty is not reported (this
    repo's uncertainty quantification IS the main surrogate's contribution,
    not something a baseline needs to also provide)

This isolates exactly one variable against the main surrogate: does
bootstrap resampling + a Gaussian-NLL-calibrated sigma head buy anything
over plain ensembling (independent init + MSE), with every other axis
(latent space, layer embedding, residual prediction) already stripped from
both sides?
"""

import torch
import torch.nn as nn


class DeterministicMLP(nn.Module):
    """One ensemble member: plain feedforward MLP, same shape as
    baseline_surrogate.mlp.model.PlainMLPSurrogate's trunk. Direct
    s_{t+1} prediction, no layer embedding, no residual add."""

    def __init__(self, state_dim: int, hidden: int = 512, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        in_dim = state_dim + 1 + 1
        blocks = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            blocks += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*blocks)
        self.head  = nn.Linear(hidden, state_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a, c], dim=-1)
        return self.head(self.trunk(x))


class VanillaDeepEnsembleSurrogate(nn.Module):
    def __init__(
        self,
        state_dim:        int   = 1053,
        n_ensemble:        int  = 5,
        hidden:            int  = 512,
        depth:             int  = 4,
        dropout:           float = 0.0,
        member_init_seed:  int   = None,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.n_ensemble = n_ensemble

        rng_state = torch.get_rng_state() if member_init_seed is not None else None
        members = []
        for k in range(n_ensemble):
            if member_init_seed is not None:
                torch.manual_seed(member_init_seed + k)
            members.append(DeterministicMLP(state_dim, hidden=hidden, depth=depth, dropout=dropout))
        self.members = nn.ModuleList(members)
        if rng_state is not None:
            torch.set_rng_state(rng_state)

    def forward_all_members(self, s, a, c) -> torch.Tensor:
        """s,a,c normalised. Returns preds (K, B, state_dim)."""
        return torch.stack([m(s, a, c) for m in self.members], dim=0)

    @torch.no_grad()
    def predict_mean(self, s, a, c) -> torch.Tensor:
        return self.forward_all_members(s, a, c).mean(dim=0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
