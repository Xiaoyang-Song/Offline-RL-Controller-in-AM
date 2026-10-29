"""Per-member outputs of the two-stage surrogate for one test transition, for fig_architecture.
    python Document/figure_data/compute_architecture_fields.py  -> Document/figure_data/architecture_fields.npz
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from surrogate_model_v3.model import load_surrogate
from surrogate_model_v3.dataset import load_trajectories, split_trajectories, TwoStageTrajectoryDataset

LAYER = 5                                    # 0-based -> layer 6


@torch.no_grad()
def members(model, ss, x, a, c, layer):
    li = torch.tensor([layer])
    out = {}
    for name, trans, cond in (("heat", model.heating_transitions, a), ("cool", model.cooling_transitions, c)):
        mus, sig = [], []
        for t in trans:
            mu, log_sigma, _ = t(x, cond, li)
            mus.append(mu[0] * ss); sig.append(log_sigma[0].exp() * ss)
        out[name + "_mu"] = torch.stack(mus).numpy(); out[name + "_sig"] = torch.stack(sig).numpy()
        if name == "heat":                   # cooling input = ensemble-mean heated field
            x = x + torch.stack(mus).mean(0)[None] / ss
    return out


def run(ckpt, power=None):
    model, sm, ss, lm, ls, cm, cs = load_surrogate(ckpt, "cpu")
    _, _, te = split_trajectories(load_trajectories(os.path.join(ROOT, "Data/DatasetV2_layer_12_samples_5000.pkl")),
                                  0.10, 0.10, 42)
    ds = TwoStageTrajectoryDataset(te[:1], state_mean=sm, state_std=ss, lp_mean=lm, lp_std=ls, cool_mean=cm,
                                   cool_std=cs, initial_temp=300.0, n_ensemble=model.n_ensemble, bootstrap_seed=42)
    S, H, A, C, _ = ds[0]
    ss_, sm_ = ss.reshape(-1), sm.reshape(-1)
    a = A[LAYER][None] if power is None else torch.tensor([[(power - lm) / ls]], dtype=torch.float32)
    o = members(model, ss_, S[LAYER][None], a, C[LAYER][None], LAYER)
    o["s_prev"] = (S[LAYER] * ss_ + sm_).numpy()
    o["u_true"] = (H[LAYER] * ss_ + sm_).numpy()
    o["s_true"] = (S[LAYER + 1] * ss_ + sm_).numpy()
    o["power"] = float(A[LAYER, 0] * ls + lm) if power is None else power
    o["cool"] = float(C[LAYER, 0] * cs + cm)
    return o


if __name__ == "__main__":
    res = {}
    for tag, ck, p in (("id", "full_range", None), ("ood", "gap_100-150_350-400_perturb0.0_bootstrap1.0", 380.0)):
        o = run(os.path.join(ROOT, "surrogate_model_v3/runs", ck, "surrogate_best.pt"), p)
        res.update({f"{tag}_{k}": v for k, v in o.items()})
        print(tag, "power", o["power"], "cool", o["cool"],
              "heat epi max", o["heat_mu"].std(0).max(), "ale max", np.sqrt((o["heat_sig"] ** 2).mean(0)).max(),
              "|mu| max", np.abs(o["heat_mu"].mean(0)).max())
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_fields.npz"), **res)
