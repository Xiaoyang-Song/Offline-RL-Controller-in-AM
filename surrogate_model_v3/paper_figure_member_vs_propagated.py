"""
surrogate_model_v3/paper_figure_member_vs_propagated.py
--------------------------------------------------------------
Figure: EPISTEMIC uncertainty of autoregressive rollouts, computed two ways, per
layer t = 1..12 on the held-out test split:
  * Jacobian propagation between layers (scheme A: mean chained through the
    ensemble mean, epistemic covariance propagated through both stages'
    Jacobians — multistep_propagation.rollout_propagated, epistemic part)
  * per-member rollouts (scheme B: each of the K members rolled out as its own
    12-layer trajectory; sigma = std across the K trajectories) — the nonlinear,
    sampling-based reference the propagation approximates to first order
  * one-step epistemic sigma (state assumed known at every layer)
plus the autoregressive RMSE of the ensemble-mean rollout for scale.
All aggregated as RMS over trajectories and nodes, in K. Panel row 2: ratio of
the two multi-step estimates (propagated / per-member).

    python -m surrogate_model_v3.paper_figure_member_vs_propagated \\
        --checkpoint "Full range (100–400 W)=surrogate_model_v3/runs/full_range/surrogate_best.pt" \\
        --checkpoint "Narrow (200–300 W)=surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt" \\
        --out surrogate_model_v3/results_multistep/paper_fig_member_vs_propagated
Re-plot without a GPU: --from_npz <out>.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@torch.no_grad()
def member_rollouts(model, s0, actions, cools):
    """-> (K, N, T, D) next states of K independent member trajectories."""
    N, T, _ = actions.shape
    out = []
    for k in range(model.n_ensemble):
        s, traj = s0, []
        for t in range(T):
            li = torch.full((N,), t, dtype=torch.long, device=s0.device)
            h = s + model.heating_transitions[k](s, actions[:, t], li)[0]
            s = h + model.cooling_transitions[k](h, cools[:, t], li)[0]
            traj.append(s)
        out.append(torch.stack(traj, 1))
    return torch.stack(out, 0)


def compute(name, ckpt, data_path, chunk, device):
    from surrogate_model_v3.model import load_surrogate
    from surrogate_model_v3.dataset import load_trajectories, split_trajectories, TwoStageTrajectoryDataset
    from surrogate_model_v3.multistep_propagation import rollout_propagated
    torch.manual_seed(0)
    model, sm, ss, lm, ls, cm, cs = load_surrogate(ckpt, device)
    _, _, te = split_trajectories(load_trajectories(data_path), 0.10, 0.10, 42)
    ds = TwoStageTrajectoryDataset(te, state_mean=sm.cpu(), state_std=ss.cpu(), lp_mean=lm, lp_std=ls,
                                   cool_mean=cm, cool_std=cs, initial_temp=300.0, n_ensemble=model.n_ensemble,
                                   bootstrap_seed=42)
    std2 = ss.to(device).reshape(-1) ** 2
    acc = {k: 0.0 for k in ("se", "prop", "member", "one")}
    n = 0
    for i0 in range(0, len(ds), chunk):
        S, _H, A, C = [torch.stack(x).to(device) for x in zip(*[ds[i][:4] for i in range(i0, min(i0 + chunk, len(ds)))])]
        T = A.shape[1]
        P = rollout_propagated(model, S[:, 0], A, C, track_aleatoric=False)
        O = rollout_propagated(model, S[:, 0], A, C, reset_each_layer=True, track_aleatoric=False)
        Mb = member_rollouts(model, S[:, 0], A, C)                       # (K,N,T,D)
        red = lambda v: (v * std2).sum(dim=(0, 2)).cpu().numpy()
        acc["se"] += red((P["next_mean"] - S[:, 1:T + 1]) ** 2)
        acc["prop"] += red(P["next_epi"])
        acc["one"] += red(O["next_epi"])
        acc["member"] += red(Mb.var(0, unbiased=False))
        n += S.shape[0] * S.shape[-1]
        print(f"  [{name}] {min(i0 + chunk, len(ds))}/{len(ds)}", flush=True)
    r = {k: np.sqrt(v / n) for k, v in acc.items()}
    return dict(rmse=r["se"], prop=r["prop"], member=r["member"], one=r["one"], layers=np.arange(1, T + 1))


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK_2, INK_3, GRID, _panel_label
    plt.rcParams.update(PAPER_RC)
    ERR, PROP, MEM, ONE = "#1f1f1f", "#2a6f97", "#e09f3e", "#c8553d"
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n + 0.2, 4.0), squeeze=False,
                             gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.12, wspace=0.28), sharex=True)
    for j, (name, r) in enumerate(results):
        L = r["layers"]
        ax = axes[0, j]
        ax.plot(L, r["rmse"], color=ERR, marker="D", markersize=3, linewidth=1.6, zorder=4)
        ax.plot(L, r["member"], color=MEM, marker="^", markersize=3.4, linewidth=1.6, zorder=3)
        ax.plot(L, r["prop"], color=PROP, marker="o", markersize=3.2, linewidth=1.6, zorder=3)
        ax.plot(L, r["one"], color=ONE, marker="s", markersize=3, linewidth=1.3, linestyle=(0, (3.2, 1.6)), zorder=2)
        ax.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel("Error / epistemic σ [K]")
        ax.set_title(name, pad=3, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        _panel_label(ax, "abcd"[j], x=-0.16 if j == 0 else -0.08)
        ax2 = axes[1, j]
        ratio = r["prop"] / np.maximum(r["member"], 1e-12)
        ax2.axhline(1.0, color=INK_3, linewidth=0.7)
        ax2.plot(L, ratio, color=PROP, marker="o", markersize=3, linewidth=1.4)
        ax2.set_ylim(0, max(1.6, ratio.max() * 1.15))
        ax2.set_xticks([1, 3, 6, 9, 12]); ax2.set_xlim(0.5, L[-1] + 0.5)
        ax2.set_xlabel("Layer (autoregressive step)")
        if j == 0:
            ax2.set_ylabel("Propagated /\nper-member")
        ax2.grid(axis="y", color=GRID, linewidth=0.6); ax2.set_axisbelow(True)
    handles = [Line2D([], [], color=ERR, marker="D", markersize=3, linewidth=1.6, label="Autoregressive RMSE"),
               Line2D([], [], color=MEM, marker="^", markersize=3.4, linewidth=1.6, label="σ epi, per-member rollouts"),
               Line2D([], [], color=PROP, marker="o", markersize=3.2, linewidth=1.6, label="σ epi, Jacobian-propagated"),
               Line2D([], [], color=ONE, marker="s", markersize=3, linewidth=1.3, linestyle=(0, (3.2, 1.6)),
                      label="σ epi, one-step")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncols=4, handlelength=2.0,
               columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    print(f"[member_vs_propagated] Saved → {out}.{{pdf,png}}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=CKPT")
    p.add_argument("--data_path", default="Data/DatasetV2_layer_12_samples_5000.pkl")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--from_npz", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.from_npz:
        results = list(np.load(a.from_npz, allow_pickle=True)["results"])
    else:
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        results = [(n.strip(), compute(n.strip(), c.strip(), a.data_path, a.chunk, device))
                   for n, c in (spec.rsplit("=", 1) for spec in a.checkpoint)]
        np.savez(a.out + ".npz", results=np.array(results, dtype=object))
    for name, r in results:
        print(f"  {name}: layer 1/6/12  RMSE {r['rmse'][0]:.2f}/{r['rmse'][5]:.2f}/{r['rmse'][-1]:.2f} | "
              f"per-member {r['member'][0]:.2f}/{r['member'][5]:.2f}/{r['member'][-1]:.2f} | "
              f"propagated {r['prop'][0]:.2f}/{r['prop'][5]:.2f}/{r['prop'][-1]:.2f} | "
              f"one-step {r['one'][0]:.2f}/{r['one'][5]:.2f}/{r['one'][-1]:.2f}")
    plot(results, a.out)


if __name__ == "__main__":
    main()
