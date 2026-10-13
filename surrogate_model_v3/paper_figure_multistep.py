"""
surrogate_model_v3/paper_figure_multistep.py
--------------------------------------------------
Figure: does uncertainty propagated BETWEEN layers track the error of
AUTOREGRESSIVE rollouts? For each checkpoint, on the held-out test split, per
layer t = 1..12:
  * autoregressive RMSE [K] of the predicted next state (mean chained through the
    ensemble mean, only s_0 and the action / cool-time schedule given)
  * one-step sigma: within-layer propagation, state assumed known at each layer
    (reset_each_layer=True)
  * between-layer propagated sigma (multistep_propagation.rollout_propagated)
sigma is aggregated like the RMSE: sqrt(mean over trajectories and nodes of
sigma_i^2 * state_std_i^2), total = epistemic + aleatoric.

    python -m surrogate_model_v3.paper_figure_multistep \\
        --checkpoint "Full range (100–400 W)=surrogate_model_v3/runs/full_range/surrogate_best.pt" \\
        --checkpoint "Narrow (200–300 W)=surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt" \\
        --out surrogate_model_v3/results_multistep/paper_fig_multistep
Re-plot from the saved npz without a GPU: add --from_npz <out>.npz.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute(name, ckpt, data_path, chunk, n_probes, device):
    from surrogate_model_v3.model import load_surrogate
    from surrogate_model_v3.dataset import load_trajectories, split_trajectories, TwoStageTrajectoryDataset
    from surrogate_model_v3.multistep_propagation import rollout_propagated
    torch.manual_seed(0)
    model, sm, ss, lm, ls, cm, cs = load_surrogate(ckpt, device)
    _, _, te = split_trajectories(load_trajectories(data_path), 0.10, 0.10, 42)
    ds = TwoStageTrajectoryDataset(te, state_mean=sm.cpu(), state_std=ss.cpu(), lp_mean=lm, lp_std=ls,
                                   cool_mean=cm, cool_std=cs, initial_temp=300.0, n_ensemble=model.n_ensemble,
                                   bootstrap_seed=42)
    std2 = (ss.to(device).reshape(-1) ** 2)
    acc = {k: 0.0 for k in ("se", "one_epi", "one_ale", "prop_epi", "prop_ale")}
    T = None; n = 0
    for i0 in range(0, len(ds), chunk):
        S, _H, A, C = [torch.stack(x).to(device) for x in zip(*[ds[i][:4] for i in range(i0, min(i0 + chunk, len(ds)))])]
        T = A.shape[1]
        P = rollout_propagated(model, S[:, 0], A, C, n_probes=n_probes)
        O = rollout_propagated(model, S[:, 0], A, C, reset_each_layer=True, n_probes=n_probes)
        red = lambda v: (v * std2).sum(dim=(0, 2)).cpu().numpy()          # sum over traj & nodes, per layer
        acc["se"] += red((P["next_mean"] - S[:, 1:T + 1]) ** 2)
        acc["one_epi"] += red(O["next_epi"]); acc["one_ale"] += red(O["next_ale"])
        acc["prop_epi"] += red(P["next_epi"]); acc["prop_ale"] += red(P["next_ale"])
        n += S.shape[0] * S.shape[-1]
        print(f"  [{name}] {min(i0 + chunk, len(ds))}/{len(ds)} trajectories", flush=True)
    r = {k: np.sqrt(v / n) for k, v in acc.items()}
    return dict(rmse=r["se"], one=np.sqrt(r["one_epi"] ** 2 + r["one_ale"] ** 2),
                prop=np.sqrt(r["prop_epi"] ** 2 + r["prop_ale"] ** 2),
                prop_epi=r["prop_epi"], one_epi=r["one_epi"], layers=np.arange(1, T + 1))


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK, INK_2, GRID, _panel_label
    plt.rcParams.update(PAPER_RC)
    ERR, PROP, ONE = "#1f1f1f", "#2a6f97", "#c8553d"
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.2, 2.6), squeeze=False, gridspec_kw=dict(wspace=0.28))
    for j, (ax, (name, r)) in enumerate(zip(axes[0], results)):
        L = r["layers"]
        ax.plot(L, r["rmse"], color=ERR, marker="D", markersize=3, linewidth=1.6, zorder=4)
        ax.plot(L, r["prop"], color=PROP, marker="o", markersize=3.2, linewidth=1.6, zorder=3)
        ax.plot(L, r["one"], color=ONE, marker="s", markersize=3, linewidth=1.4, linestyle=(0, (3.2, 1.6)), zorder=3)
        ax.set_xticks([1, 3, 6, 9, 12]); ax.set_xlim(0.5, L[-1] + 0.5); ax.set_ylim(bottom=0)
        ax.set_xlabel("Layer (autoregressive step)")
        if j == 0:
            ax.set_ylabel("Next-state error / σ [K] (RMS over nodes)")
        ax.set_title(name, pad=3, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        _panel_label(ax, "abcd"[j], x=-0.16 if j == 0 else -0.08)
    handles = [Line2D([], [], color=ERR, marker="D", markersize=3, linewidth=1.6, label="Autoregressive RMSE"),
               Line2D([], [], color=PROP, marker="o", markersize=3.2, linewidth=1.6, label="σ, propagated between layers"),
               Line2D([], [], color=ONE, marker="s", markersize=3, linewidth=1.4, linestyle=(0, (3.2, 1.6)),
                      label="σ, one-step")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncols=3, handlelength=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    print(f"[multistep] Saved → {out}.{{pdf,png}}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=CKPT")
    p.add_argument("--data_path", default="Data/DatasetV2_layer_12_samples_5000.pkl")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--n_probes", type=int, default=16)
    p.add_argument("--from_npz", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.from_npz:
        z = np.load(a.from_npz, allow_pickle=True)
        results = list(z["results"])
    else:
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        results = []
        for spec in a.checkpoint:
            name, ckpt = [x.strip() for x in spec.rsplit("=", 1)]
            results.append((name, compute(name, ckpt, a.data_path, a.chunk, a.n_probes, device)))
        np.savez(a.out + ".npz", results=np.array(results, dtype=object))
    for name, r in results:
        print(f"  {name}: layer   1 / 6 / 12   RMSE {r['rmse'][0]:.2f} / {r['rmse'][5]:.2f} / {r['rmse'][-1]:.2f} K | "
              f"σ propagated {r['prop'][0]:.2f} / {r['prop'][5]:.2f} / {r['prop'][-1]:.2f} | "
              f"σ one-step {r['one'][0]:.2f} / {r['one'][5]:.2f} / {r['one'][-1]:.2f}")
    plot(results, a.out)


if __name__ == "__main__":
    main()
