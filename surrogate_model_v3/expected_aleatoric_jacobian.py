"""
surrogate_model_v3/expected_aleatoric_jacobian.py
--------------------------------------------------------
Expected full-transition ALEATORIC σ for a surrogate trained with injected
target noise (train.py --perturb_frac ε), computed with that model's EXACT
cooling Jacobian instead of assuming I + J ≈ I.

With independent per-node noise of std ε on the stored end-of-heating field and
on the next state, the heating stage's aleatoric covariance is ε² I and the
cooling stage's (its target is next-state minus the noisy heating field) is
2ε² I. Propagating heating through A = I + ∂μ_cool/∂s_heat:

    Σ_full = ε² (A Aᵀ + 2 I)    →    σ_full,i = ε √(2 + ‖a_i‖²)

(a_i = row i of A). The node-averaged expected σ — the same aggregation as the
learned σ in evaluate_ood.py — is averaged over a random sample of test
transitions (teacher-forced s_t, as in evaluate_ood.py's combined pass).
Output: a JSON with the expected value per checkpoint (and the I + J ≈ I value
√3 ε for comparison).

Usage
-----
    python -m surrogate_model_v3.expected_aleatoric_jacobian \\
        --checkpoint <ckpt1> <ckpt2> ... --perturb 0.05 0.1 0.2 \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --out surrogate_model_v3/results_aleatoric_calibration/expected_full_aleatoric.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.dataset import load_trajectories, split_trajectories, TwoStageSurrogateDataset
from surrogate_model_v3.model import load_surrogate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--perturb", nargs="+", type=float, required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--n_transitions", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="")
    a = p.parse_args()
    assert len(a.checkpoint) == len(a.perturb)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test = split_trajectories(load_trajectories(a.data_path), 0.10, 0.10, a.seed)
    results = []
    for ckpt, eps in zip(a.checkpoint, a.perturb):
        model, sm, ss, lm, ls, cm, cs = load_surrogate(ckpt, device)
        # clean (unperturbed) test transitions, same normalisation as the model
        ds = TwoStageSurrogateDataset(test, state_mean=sm.cpu(), state_std=ss.cpu(), lp_mean=lm, lp_std=ls,
                                      cool_mean=cm, cool_std=cs, initial_temp=300.0,
                                      n_ensemble=model.n_ensemble, bootstrap_seed=0)
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(len(ds), size=min(a.n_transitions, len(ds)), replace=False)
        per_tr, diag_only = [], []
        for i in idx:
            s, act, c, _h, _s2, layer, _b = ds[int(i)]
            s, act, c = s[None].to(device), act[None].to(device), c[None].to(device)
            layer = torch.as_tensor([int(layer)], device=device)
            with torch.no_grad():
                s_heat = s + model.predict_heating_ensemble(s, act, layer)[0]
            J = torch.autograd.functional.jacobian(
                lambda x: model._cooling_mean_fn(x.unsqueeze(0), c, layer)[0], s_heat[0], vectorize=True)
            A = torch.eye(J.shape[0], device=device) + J
            row_sq = (A ** 2).sum(1)                                     # ‖a_i‖²
            per_tr.append(float((eps * torch.sqrt(2.0 + row_sq)).mean()))
            diag_only.append(float((eps * torch.sqrt(2.0 + A.diagonal() ** 2)).mean()))
        r = dict(checkpoint=os.path.abspath(ckpt), perturb=eps, n_transitions=len(idx),
                 expected_full=float(np.mean(per_tr)), expected_full_sd=float(np.std(per_tr)),
                 expected_full_diag_only=float(np.mean(diag_only)), sqrt3_eps=float(np.sqrt(3) * eps))
        results.append(r)
        print(f"[expected_aleatoric] ε={eps:g}: Jacobian-based {r['expected_full']:.4f} "
              f"(sd over transitions {r['expected_full_sd']:.4f}; diagonal-only {r['expected_full_diag_only']:.4f}; "
              f"√3ε {r['sqrt3_eps']:.4f})  [{ckpt}]")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[expected_aleatoric] Saved → {a.out}")


if __name__ == "__main__":
    main()
