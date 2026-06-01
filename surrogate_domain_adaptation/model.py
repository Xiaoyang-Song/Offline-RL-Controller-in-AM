"""
surrogate_domain_adaptation/model.py
--------------------------------------
Domain adapter modules for few-shot cooling-time adaptation.

Architecture
------------
  The pretrained EnsembleLatentDynamicsModel (from surrogate_model_latent)
  is frozen.  Lightweight DomainAdapter modules are inserted at three points:

    s_t  →[Encoder]→  z_t  →[EncAdapter]→  z̃_t
    (z̃_t, a_t, l) →[Transition_k]→  Δz_k  →[TransAdapter]→  Δ̃z_k
    z̃_t + mean_k(Δ̃z_k)  →[DecAdapter]→  ẑ  →[Decoder]→  ŝ_{t+1}

  Each DomainAdapter is a two-layer bottleneck MLP initialised to the
  identity (zero-init on the up-projection), so the adapted model starts
  as an exact copy of the base and only diverges as the adapter is trained.

Why adapters for cooling time?
-------------------------------
  Cooling time changes the inter-layer temperature state systematically:
    ct_050 → mean field at layer 12 ≈ 2274 K
    ct_150 → mean field at layer 12 ≈  953 K  (2.4× cooler)
  Three effects need adaptation:
    1. Encoder: states lie in a different region → enc adapter
    2. Transition: thermal response differs (cooler start → different ΔT) → trans adapter
    3. Decoder: latent → state mapping shifts with the temperature scale → dec adapter

Parameter count  (bottleneck=32, latent_dim=32)
    enc_adapter:   32→32→32  ≈  2.1 K params
    trans_adapter: 32→32→32  ≈  2.1 K params  (shared across K members)
    dec_adapter:   32→32→32  ≈  2.1 K params
    Total ≈ 6.3 K  (< 0.4 % of the 1.9 M base model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from surrogate_model_latent.model import EnsembleLatentDynamicsModel


# ---------------------------------------------------------------------------
# Adapter building block
# ---------------------------------------------------------------------------

class DomainAdapter(nn.Module):
    """
    Bottleneck residual adapter:  x  →  x + W_up(σ(W_down(x)))

    Initialised to identity via zero-init on W_up so the adapted model
    starts identical to the base at K=0.

    Parameters
    ----------
    dim        : input / output dimension
    bottleneck : inner bottleneck width (default 32)
    """
    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up   = nn.Linear(bottleneck, dim)
        # Zero init → identity at initialisation
        nn.init.kaiming_uniform_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.silu(self.down(x)))


# ---------------------------------------------------------------------------
# Adapted ensemble dynamics model
# ---------------------------------------------------------------------------

class AdaptedEnsembleLatentDynamicsModel(nn.Module):
    """
    Drop-in replacement for EnsembleLatentDynamicsModel with domain adapters.

    The base model is frozen; only the three DomainAdapter modules are trained
    during few-shot adaptation.  The forward API is identical to the base so
    the existing loss functions in surrogate_model_latent/train.py work unchanged.

    Parameters
    ----------
    base_model      : pretrained EnsembleLatentDynamicsModel (will be frozen)
    bottleneck      : adapter bottleneck width (default 32)
    use_enc_adapter : whether to adapt the encoder output
    use_trans_adapter: whether to adapt the transition output
    use_dec_adapter : whether to adapt the decoder input
    """

    def __init__(
        self,
        base_model:        EnsembleLatentDynamicsModel,
        bottleneck:        int  = 32,
        use_enc_adapter:   bool = True,
        use_trans_adapter: bool = True,
        use_dec_adapter:   bool = True,
    ):
        super().__init__()
        self.base        = base_model
        self.latent_dim  = base_model.latent_dim
        self.state_dim   = base_model.state_dim
        self.action_dim  = base_model.action_dim
        self.n_ensemble  = base_model.n_ensemble
        self.n_layers    = base_model.n_layers

        self.enc_adapter   = DomainAdapter(self.latent_dim, bottleneck) if use_enc_adapter   else None
        self.trans_adapter = DomainAdapter(self.latent_dim, bottleneck) if use_trans_adapter else None
        self.dec_adapter   = DomainAdapter(self.latent_dim, bottleneck) if use_dec_adapter   else None

    # ------------------------------------------------------------------
    def freeze_base(self) -> None:
        """Freeze all base model parameters (call before adaptation)."""
        for p in self.base.parameters():
            p.requires_grad_(False)

    def unfreeze_adapters(self) -> None:
        """Ensure adapter parameters are trainable."""
        for m in [self.enc_adapter, self.trans_adapter, self.dec_adapter]:
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)

    def adapter_parameters(self):
        """Return only the adapter parameters for the optimizer."""
        params = []
        for m in [self.enc_adapter, self.trans_adapter, self.dec_adapter]:
            if m is not None:
                params.extend(m.parameters())
        return params

    def count_adapter_parameters(self) -> int:
        return sum(p.numel() for p in self.adapter_parameters())

    # ------------------------------------------------------------------
    def encode(self, s: torch.Tensor) -> torch.Tensor:
        z = self.base.encoder(s)
        if self.enc_adapter is not None:
            z = self.enc_adapter(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.dec_adapter is not None:
            z = self.dec_adapter(z)
        return self.base.decoder(z)

    def predict_ensemble(
        self,
        z:         torch.Tensor,   # (B, latent_dim)  — already through enc_adapter
        a:         torch.Tensor,   # (B, action_dim)
        layer_idx: torch.Tensor,   # (B,) int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean_delta, std_delta).
        The transition adapter is applied to every member's output before aggregation,
        preserving ensemble diversity (std is unaffected by the shared shift).
        """
        deltas = torch.stack(
            [t(z, a, layer_idx) for t in self.base.transitions], dim=0
        )  # (K, B, latent_dim)

        if self.trans_adapter is not None:
            K, B, D = deltas.shape
            deltas = deltas + self.trans_adapter(
                deltas.reshape(K * B, D)
            ).reshape(K, B, D)

        return deltas.mean(0), deltas.std(0)

    # ------------------------------------------------------------------
    def forward(
        self,
        s_t:       torch.Tensor,
        a_t:       torch.Tensor,
        layer_idx: torch.Tensor,
        s_t1:      Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Identical return signature to EnsembleLatentDynamicsModel.forward:
            (s_t_recon, s_t1_pred, mu_deltas, z_t)
        """
        z_t       = self.encode(s_t)
        s_t_recon = self.decode(z_t)

        mu_deltas = torch.stack(
            [t(z_t, a_t, layer_idx) for t in self.base.transitions], dim=0
        )  # (K, B, latent_dim)

        if self.trans_adapter is not None:
            K, B, D = mu_deltas.shape
            mu_deltas = mu_deltas + self.trans_adapter(
                mu_deltas.reshape(K * B, D)
            ).reshape(K, B, D)

        z_t1_pred = z_t + mu_deltas.mean(0)
        s_t1_pred = self.decode(z_t1_pred)

        return s_t_recon, s_t1_pred, mu_deltas, z_t

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        s_0:     torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Auto-regressive rollout using ensemble mean. Same API as base."""
        B, T, _ = actions.shape
        device   = s_0.device
        z_t      = self.encode(s_0)
        pred_states, epistemic_stds = [], []

        for t in range(T):
            a_t       = actions[:, t, :]
            layer_idx = torch.full((B,), t, dtype=torch.long, device=device)
            mu_mean, mu_std = self.predict_ensemble(z_t, a_t, layer_idx)
            z_t       = z_t + mu_mean
            pred_states.append(self.decode(z_t))
            epistemic_stds.append(mu_std.mean(dim=-1))

        return torch.stack(pred_states, dim=1), torch.stack(epistemic_stds, dim=1)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_base    = sum(p.numel() for p in self.base.parameters())
        n_adapter = self.count_adapter_parameters()
        adapters  = "+".join(
            name for name, m in [
                ("enc",   self.enc_adapter),
                ("trans", self.trans_adapter),
                ("dec",   self.dec_adapter),
            ] if m is not None
        )
        return (
            f"AdaptedEnsembleLatentDynamicsModel("
            f"adapters=[{adapters}], "
            f"base={n_base:,} params (frozen), "
            f"adapter={n_adapter:,} params (trainable))"
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_adapted(
    model:        AdaptedEnsembleLatentDynamicsModel,
    state_mean:   torch.Tensor,
    state_std:    torch.Tensor,
    action_mean:  float,
    action_std:   float,
    K:            int,
    path:         str,
    extra:        dict = None,
) -> None:
    """Save adapted model checkpoint (adapter weights + normalizers)."""
    data = {
        "enc_adapter_state":   model.enc_adapter.state_dict()   if model.enc_adapter   else None,
        "trans_adapter_state": model.trans_adapter.state_dict() if model.trans_adapter else None,
        "dec_adapter_state":   model.dec_adapter.state_dict()   if model.dec_adapter   else None,
        "state_mean":          state_mean.cpu(),
        "state_std":           state_std.cpu(),
        "action_mean":         action_mean,
        "action_std":          action_std,
        "K":                   K,
        "adapter_config": {
            "bottleneck":        model.enc_adapter.down.out_features
                                 if model.enc_adapter else None,
            "use_enc_adapter":   model.enc_adapter   is not None,
            "use_trans_adapter": model.trans_adapter is not None,
            "use_dec_adapter":   model.dec_adapter   is not None,
        },
    }
    if extra:
        data.update(extra)
    torch.save(data, path)
    print(f"[model] Adapted checkpoint (K={K}) → {path}")


def load_adapted(
    base_model: EnsembleLatentDynamicsModel,
    path:       str,
    device:     str = "cpu",
) -> AdaptedEnsembleLatentDynamicsModel:
    """Load adapter weights onto a base model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["adapter_config"]
    adapted = AdaptedEnsembleLatentDynamicsModel(
        base_model,
        bottleneck        = cfg["bottleneck"] or 32,
        use_enc_adapter   = cfg["use_enc_adapter"],
        use_trans_adapter = cfg["use_trans_adapter"],
        use_dec_adapter   = cfg["use_dec_adapter"],
    ).to(device)

    if adapted.enc_adapter   and ckpt["enc_adapter_state"]:
        adapted.enc_adapter.load_state_dict(ckpt["enc_adapter_state"])
    if adapted.trans_adapter and ckpt["trans_adapter_state"]:
        adapted.trans_adapter.load_state_dict(ckpt["trans_adapter_state"])
    if adapted.dec_adapter   and ckpt["dec_adapter_state"]:
        adapted.dec_adapter.load_state_dict(ckpt["dec_adapter_state"])

    adapted.eval()
    return adapted
