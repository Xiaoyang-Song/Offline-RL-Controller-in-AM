"""
rl_narrow/plot_maps_by_method.py
--------------------------------
Transposed variant of rl_narrow/plot_maps.py: columns = methods (offline baselines left, digital
twin-enabled online RL right, each group under a coloured header), rows = layers (default 4, 8, 12).
Same data, method selection, colours and deviation definition as plot_maps.py.

    python -m rl_narrow.plot_maps_by_method --results_dir rl_narrow/results_real \
        --out rl_narrow/results_real/figures/fig_rl_maps_by_method
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_model_v3.paper_figure_ood import PAPER_RC
from baselines.common.data_utils import load_mesh_nodes
from matplotlib.colors import LinearSegmentedColormap
from rl_narrow.plot_maps import DEV, FAMILIES, GROUPS, find

# jet-based diverging map (jet's blue end -> white at 0 -> jet's red end), matching the jet temperature
# fields of Document/make_traj_figure*.py so the paper's field figures share one colour language
JET_DIV = LinearSegmentedColormap.from_list(
    "jet_div", ["#00007f", "#0000ff", "#0080ff", "#00d4ff", "#ffffff", "#ffd400", "#ff8000", "#ff0000", "#7f0000"])
CMAPS = {"dev": DEV, "jet": JET_DIV}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="rl_narrow/results_real")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--cool_time", type=float, default=0.11)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T_l", type=float, default=2000.0)
    ap.add_argument("--T_h", type=float, default=2800.0)
    ap.add_argument("--vlim", type=float, default=1500.0)
    ap.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
    ap.add_argument("--cmap", choices=list(CMAPS), default="dev",
                    help="dev = original blue/red palette; jet = jet-based diverging (matches the trajectory figure).")
    ap.add_argument("--out", default="rl_narrow/results_real/figures/fig_rl_maps_by_method")
    args = ap.parse_args()

    nodes = load_mesh_nodes(args.mesh_path)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    W_DOM, H_DOM, L = 12.0, 3.0, 12
    side = lambda l: min(W_DOM, H_DOM) * (0.4 + 0.1 * (l - 1) / (L - 1))

    found = []
    for fam, label, grp in FAMILIES:
        x = find(args.results_dir, fam, args.seed)
        if x is not None:
            found.append((label or x[0], x[1], x[2], grp))
    cols = []                                       # offline left -> online right; worst -> best within a group
    for grp, _t, _c in GROUPS:
        cols += sorted([r for r in found if r[3] == grp], key=lambda r: r[2])
    nC, nR = len(cols), len(args.layers)
    n_off = sum(r[3] == "offline" for r in cols)

    plt.rcParams.update(PAPER_RC)
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})
    gap = 0.18
    wr = [1.0] * n_off + ([gap] if 0 < n_off < nC else []) + [1.0] * (nC - n_off)
    fig = plt.figure(figsize=(7.2, 0.9 + 0.62 * nR))
    gs = fig.add_gridspec(nR, len(wr), width_ratios=wr, hspace=0.10, wspace=0.07)
    gcol = lambda j: j if j < n_off else j + (1 if 0 < n_off < nC else 0)
    axes = np.empty((nR, nC), dtype=object)
    for i in range(nR):
        for j in range(nC):
            axes[i, j] = fig.add_subplot(gs[i, gcol(j)])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-args.vlim, vmax=args.vlim)
    saved = dict(layers=np.array(args.layers), labels=np.array([c[0] for c in cols]),
                 groups=np.array([c[3] for c in cols]), avg_reward=np.array([c[2] for c in cols]),
                 cool_time=args.cool_time, seed=args.seed, vlim=args.vlim, nodes=nodes, triangles=tri.triangles)
    colors = dict((g, c) for g, _t, c in GROUPS)
    for j, (label, path, _s, grp) in enumerate(cols):
        z = np.load(path)
        e = int(np.argmin(np.abs(z["cool_times"] - args.cool_time)))
        online = grp == "online"
        for i, l in enumerate(args.layers):
            ax = axes[i, j]
            u = z["u_heat"][e, l - 1]
            dlt = np.where(u > args.T_h, u - args.T_h, np.where(u < args.T_l, u - args.T_l, 0.0))
            h = ax.tripcolor(tri, dlt, cmap=CMAPS[args.cmap], norm=norm, shading="gouraud")
            sd = side(l)
            ax.add_patch(Rectangle((W_DOM / 2 - sd / 2, H_DOM / 2 - sd / 2), sd, sd, fill=False, ec="k",
                                   lw=0.5, ls=(0, (2.0, 1.4))))
            m = ((np.abs(nodes[:, 0] - W_DOM / 2) <= sd / 2) & (np.abs(nodes[:, 1] - H_DOM / 2) <= sd / 2))
            saved[f"delta_c{j}_l{l}"] = dlt.astype(np.float32)
            saved[f"dbar_c{j}_l{l}"] = float(np.abs(dlt[m]).mean())
            ax.text(0.985, 0.035, f"$\\bar d$ = {np.abs(dlt[m]).mean():.0f} K", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.0, color="#1f2329",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.35))
            ax.set_xlim(0, W_DOM); ax.set_ylim(0, H_DOM); ax.set_aspect("auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(label, fontsize=7.6, pad=4, linespacing=1.05,
                             color=colors[grp] if online else "#1f2329", fontweight="bold" if online else "normal")
            if j == 0:
                ax.set_ylabel(f"Layer {l}", fontsize=8.5, fontweight="bold", labelpad=4)
    cb = fig.colorbar(h, ax=list(axes.ravel()), orientation="horizontal", pad=0.05, shrink=0.55, aspect=40)
    cb.set_ticks([-args.vlim, -args.vlim / 2, 0, args.vlim / 2, args.vlim])
    cb.set_label("Deviation from target window [K]")
    for grp, title, col in GROUPS:                  # group header bars above the column titles
        idx = [j for j, c in enumerate(cols) if c[3] == grp]
        if not idx:
            continue
        x0, x1 = axes[0, idx[0]].get_position().x0, axes[0, idx[-1]].get_position().x1
        yb = axes[0, 0].get_position().y1 + 0.40 / fig.get_figheight()
        fig.add_artist(plt.Line2D([x0 + 0.004, x1 - 0.004], [yb, yb], transform=fig.transFigure, color=col, lw=2.2))
        fig.text((x0 + x1) / 2, yb + 0.07 / fig.get_figheight(), title, ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", fontstyle="italic", color=col)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    np.savez_compressed(f"{args.out}_data.npz", **saved)
    print("saved", args.out, "(+ _data.npz) | columns:", [c[0].replace(chr(10), " ") for c in cols])


if __name__ == "__main__":
    main()
