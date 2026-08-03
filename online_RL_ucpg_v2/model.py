"""
online_RL_ucpg_v2/model.py
------------------------------
Latent-space CONTINUOUS stochastic policy network for Uncertainty-Constrained
Policy Gradient: pi_theta: Z -> Gaussian(A), A = R (laser power, Watts).

Differs from online_RL_ucpg/model.py only in the action representation
--------------------------------------------------------------------------
v1's LatentPolicyNet produced categorical logits over a discrete laser-power
grid. Laser power is physically continuous, so this version instead outputs
the (mean, std) of a diagonal Gaussian and lets nabla_theta log pi_theta use
Normal.log_prob — everything else (trunk architecture, layer embedding,
latent input, the surrogate this drives) is unchanged.

Architecture
------------
  Input  : obs (latent_dim+2,) = latent state z_t (latent_dim) + float layer
           token in [0,1] + float cool_time token in [0,1] (see env.py's
           _make_obs — cool_time is exogenous per-episode context, not an
           action, but IS observed: a proper contextual-MDP setup, since in
           the real process cooling duration is a known build parameter).
  Embed  : float layer token -> integer index -> nn.Embedding(n_layers, layer_embed_dim)
  Stem   : Linear(latent_dim + layer_embed_dim + 1 -> H) -> LayerNorm(H) -> SiLU
           (+1 = the raw cool_time token, concatenated directly — it is
           already a normalised scalar, so no embedding table is needed)
  Trunk  : depth x ResidualBlock(H)
  mu head    : Linear(H -> 1) -> tanh -> rescaled to [action_min, action_max]
  sigma      : a single, state-INDEPENDENT learnable scale (standard practice
               for Gaussian policy-gradient methods, e.g. Spinning Up / TRPO's
               continuous-control policies — avoids the network learning a
               degenerate state-dependent exploration collapse early in
               training), expressed as a PETS-style soft-clamped log-fraction
               of the action half-range so the bound is scale-invariant no
               matter what --action_min/--action_max are set to.

Why the mean is squashed but the SAMPLE is not
------------------------------------------------
mu = action_center + action_half_range * tanh(mu_raw) keeps the Gaussian's
MEAN inside [action_min, action_max] (a sensible, bounded place for the
policy to aim), but the actual sampled action a ~ Normal(mu, sigma) is left
UNCLIPPED — no tanh-squash-then-Jacobian-correct on the sample itself, no
clipping at the environment boundary. This is a deliberate simplification
matching v1's philosophy ("the UCPG policy may pick actions the surrogate
was never trained on; the uncertainty constraint, not a hard action bound,
is what should discourage it") — see the package README. It also keeps
nabla_theta log pi_theta(a_t|s_t) exactly Normal.log_prob(a_t), no
change-of-variables term to derive or debug.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block  (identical to online_RL_ucpg/model.py's ResidualBlock)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """x → Linear → LayerNorm → SiLU → Dropout → Linear → LayerNorm → (+ skip) → SiLU"""
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class ContinuousLatentPolicyNet(nn.Module):
    """
    Diagonal Gaussian policy over continuous laser power, conditioned on the
    surrogate's latent state and a learned per-layer embedding.

    Parameters
    ----------
    latent_dim      : surrogate latent space dimension
    action_min      : lower end of the "aim point" range for the policy mean [W]
    action_max      : upper end of the "aim point" range for the policy mean [W]
    hidden          : width of every hidden layer
    depth           : number of ResidualBlocks in the trunk
    dropout         : dropout probability inside each block (0 = off)
    n_layers        : number of LPBF build layers (sets embedding table size)
    layer_embed_dim : dimension of the learned layer embedding
    log_sigma_init  : initial value of the (log-fraction-of-half-range) std
    min_log_sigma   : lower soft bound on log-fraction std (prevents collapse to 0)
    max_log_sigma   : upper soft bound on log-fraction std (prevents runaway exploration)
    """

    def __init__(
        self,
        latent_dim:      int,
        action_min:      float,
        action_max:      float,
        hidden:          int   = 128,
        depth:           int   = 3,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
        log_sigma_init:  float = -1.0,
        min_log_sigma:   float = -3.0,
        max_log_sigma:   float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim      = latent_dim
        self.action_min      = float(action_min)
        self.action_max      = float(action_max)
        self.action_center    = (self.action_min + self.action_max) / 2.0
        self.action_half_range = (self.action_max - self.action_min) / 2.0
        self.hidden          = hidden
        self.depth           = depth
        self.n_layers        = n_layers
        self.layer_embed_dim = layer_embed_dim

        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        stem_in = latent_dim + layer_embed_dim + 1   # +1 = raw cool_time token
        self.stem = nn.Sequential(
            nn.Linear(stem_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout=dropout) for _ in range(depth)]
        )
        self.mu_head = nn.Linear(hidden, 1)
        # Near-zero init → tanh(mu_raw) ≈ 0 → mu ≈ action_center at the start
        # of training (maximum-uncertainty starting aim point, analogous to
        # v1's near-uniform categorical init).
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mu_head.bias)

        # State-INDEPENDENT log-std, expressed as a PETS-style soft-clamped
        # log-fraction of the action half-range (see module docstring).
        self.raw_log_sigma = nn.Parameter(torch.tensor(float(log_sigma_init)))
        self.min_log_sigma = nn.Parameter(torch.tensor(float(min_log_sigma)))
        self.max_log_sigma = nn.Parameter(torch.tensor(float(max_log_sigma)))

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor):
        """
        obs     : (B, latent_dim+2) — second-to-last column = normalised float
                  layer token in [0,1]; last column = normalised cool_time
                  token in [0,1] (exogenous per-episode context, see env.py)
        returns : (mu, sigma), each (B,) — Gaussian policy parameters [W]
        """
        z             = obs[:, :-2]                                          # (B, latent_dim)
        layer_float   = obs[:, -2]                                           # (B,)
        cool_time_tok = obs[:, -1]                                           # (B,)
        layer_idx   = (layer_float * (self.n_layers - 1)).round().long()
        layer_idx   = layer_idx.clamp(0, self.n_layers - 1)
        e = self.layer_embed(layer_idx)                                      # (B, embed_dim)
        x = torch.cat([z, e, cool_time_tok.unsqueeze(-1)], dim=-1)
        x = self.stem(x)
        x = self.trunk(x)

        mu_raw = self.mu_head(x).squeeze(-1)                                 # (B,)
        mu     = self.action_center + self.action_half_range * torch.tanh(mu_raw)

        log_sigma = self.max_log_sigma - F.softplus(self.max_log_sigma - self.raw_log_sigma)
        log_sigma = self.min_log_sigma + F.softplus(log_sigma - self.min_log_sigma)
        sigma = self.action_half_range * log_sigma.exp()
        sigma = sigma.expand_as(mu)

        return mu, sigma

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ContinuousLatentPolicyNet(latent_dim={self.latent_dim}, "
            f"action_range=[{self.action_min:.0f}, {self.action_max:.0f}]W, "
            f"hidden={self.hidden}, depth={self.depth}, "
            f"n_layers={self.n_layers}, layer_embed_dim={self.layer_embed_dim}, "
            f"params={self.count_parameters():,})"
        )
