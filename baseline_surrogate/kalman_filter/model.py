"""
baseline_surrogate/kalman_filter/model.py
----------------------------------------------
Baseline 3: linear-Gaussian Kalman-filter dynamics model. The classical
control-theory counterpart to the main surrogate's learned nonlinear
latent transition:

    z_t      = PCA.transform(s_t)                       fixed, non-learned "latent"
    z_{t+1}  = A z_t + B a_t + C cool_t + b_l[layer]     fit by ordinary least
                                                          squares (closed form,
                                                          no gradient descent)
    s_{t+1}  = PCA.inverse_transform(z_{t+1})

Why no filtering UPDATE step (this is a "Kalman filter dynamics model,"
not a running filter): a surrogate model is always queried with the true
current state s_t, never a noisy sensor reading of it — unlike
baselines/kalman_particle (a CONTROLLER, which only gets a synthetically
noised scalar reading and must filter it). There is nothing to fuse an
observation against here, so only the linear-Gaussian PREDICT/process
model is exercised. This is still the textbook Kalman filter's process
model (Kalman, 1960) — a real deployed filter reduces to exactly this
prediction equation whenever its own state estimate is already exact.

Fit once via `fit_linear_gaussian_kalman` (closed form); no torch.nn.Module
anywhere in this baseline — see train.py for the fit CLI.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA


class LinearGaussianKalmanSurrogate:
    """PCA transform/inverse-transform implemented directly as linear
    algebra on `components`/`mean` (rather than calling a live sklearn PCA
    object) so this class is a plain, self-contained data holder — no
    dependency on sklearn's internal fitted-estimator state (some sklearn
    versions' PCA.transform reads OTHER fitted attributes too, e.g.
    explained_variance_, which a hand-reconstructed PCA object built from
    only components_/mean_ wouldn't have — see the AttributeError this
    replaced). components/mean come from a one-time sklearn PCA().fit() in
    fit_linear_gaussian_kalman; nothing after that needs sklearn at all."""

    def __init__(self, components: np.ndarray, mean: np.ndarray, coeffs: np.ndarray,
                n_layers: int, latent_dim: int):
        self.components = components   # (n_components, state_dim)
        self.mean       = mean         # (state_dim,)
        self.coeffs     = coeffs       # (latent_dim + 2 + n_layers, latent_dim)
        self.n_layers   = n_layers
        self.latent_dim = latent_dim

    def _pca_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components.T

    def _pca_inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components + self.mean

    def _design(self, z: np.ndarray, a: np.ndarray, c: np.ndarray, layer_idx: np.ndarray) -> np.ndarray:
        B = z.shape[0]
        onehot = np.zeros((B, self.n_layers), dtype=np.float32)
        onehot[np.arange(B), np.clip(layer_idx, 0, self.n_layers - 1)] = 1.0
        return np.concatenate([z, a[:, None], c[:, None], onehot], axis=1).astype(np.float32)

    def predict_raw(
        self, s_raw: np.ndarray, a_raw: np.ndarray, c_raw: np.ndarray, layer_idx: np.ndarray,
    ) -> np.ndarray:
        """All args raw numpy: s_raw (B, state_dim) [K], a_raw (B,) [W], c_raw (B,) [s],
        layer_idx (B,) int. Returns predicted s_{t+1}, raw Kelvin (B, state_dim)."""
        z = self._pca_transform(s_raw)
        design = self._design(z, a_raw, c_raw, layer_idx)
        z_next = design @ self.coeffs
        return self._pca_inverse_transform(z_next).astype(np.float32)


def fit_linear_gaussian_kalman(
    s: np.ndarray, a: np.ndarray, c: np.ndarray, layer_idx: np.ndarray, s2: np.ndarray,
    n_components: int = 64, n_layers: int = 12,
) -> LinearGaussianKalmanSurrogate:
    """Fit PCA on pooled {s, s2} (so the same basis represents both sides of
    the transition), then fit the linear process model on the (filtered)
    training transitions via np.linalg.lstsq (closed form)."""
    pooled = np.concatenate([s, s2], axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(pooled)
    print(f"[kalman_filter] PCA({n_components}) explained variance ratio "
          f"sum = {pca.explained_variance_ratio_.sum():.4f}")

    model = LinearGaussianKalmanSurrogate(
        components=pca.components_.astype(np.float32), mean=pca.mean_.astype(np.float32),
        coeffs=np.zeros((1, 1), dtype=np.float32), n_layers=n_layers, latent_dim=n_components,
    )
    z_t    = model._pca_transform(s)
    z_next = model._pca_transform(s2)

    design = model._design(z_t, a, c, layer_idx)
    coeffs, *_ = np.linalg.lstsq(design, z_next, rcond=None)
    model.coeffs = coeffs.astype(np.float32)

    resid = design @ coeffs - z_next
    print(f"[kalman_filter] Fit on {len(s):,} transitions. "
          f"Latent-space residual RMSE = {np.sqrt((resid ** 2).mean()):.4f}")
    return model


def load_kalman_surrogate(path: str) -> tuple:
    """Reconstruct a LinearGaussianKalmanSurrogate (and its normalisers)
    from a checkpoint saved by kalman_filter/train.py."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LinearGaussianKalmanSurrogate(
        components=ckpt["pca_components"], mean=ckpt["pca_mean"],
        coeffs=ckpt["coeffs"], n_layers=ckpt["n_layers"], latent_dim=ckpt["latent_dim"],
    )
    return (model, ckpt["state_mean"], ckpt["state_std"],
            ckpt["lp_mean"], ckpt["lp_std"], ckpt["cool_mean"], ckpt["cool_std"])


class KalmanPredictor:
    """Adapts LinearGaussianKalmanSurrogate.predict_raw (raw numpy) to
    common/eval.py's predict_fn(s_norm, a_norm, c_norm, layer_idx) ->
    s2_pred_norm interface (normalised torch tensors) — denormalises,
    predicts in raw space, renormalises back."""

    def __init__(self, kf: LinearGaussianKalmanSurrogate, state_mean, state_std,
                 lp_mean, lp_std, cool_mean, cool_std, device: str):
        self.kf = kf
        self.state_mean = state_mean.to(device)
        self.state_std  = state_std.to(device)
        self.lp_mean, self.lp_std   = lp_mean, lp_std
        self.cool_mean, self.cool_std = cool_mean, cool_std
        self.device = device

    def predict(self, s_norm, a_norm, c_norm, layer_idx) -> torch.Tensor:
        s_raw = (s_norm * self.state_std + self.state_mean).cpu().numpy()
        a_raw = (a_norm[:, 0] * self.lp_std   + self.lp_mean).cpu().numpy()
        c_raw = (c_norm[:, 0] * self.cool_std + self.cool_mean).cpu().numpy()
        layer_np = layer_idx.cpu().numpy()

        s2_raw = self.kf.predict_raw(s_raw, a_raw, c_raw, layer_np)
        s2_raw_t = torch.tensor(s2_raw, dtype=torch.float32, device=self.device)
        return (s2_raw_t - self.state_mean) / self.state_std
