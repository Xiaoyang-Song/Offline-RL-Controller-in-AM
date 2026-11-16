"""
rl_narrow/plot_maps.py
----------------------
Paper figure: real end-of-heating (EoH) deviation maps under each control scheme. Rows = methods, offline
baselines on top and digital twin-enabled online RL at the bottom (Kalman filter not shown), columns = layers (default 4, 8, 12). Each map shows the signed deviation of the real
EoH field from the target window [T_L, T_H]:
    delta = u - T_H if u > T_H (overheating, red),  u - T_L if u < T_L (insufficient heating, blue),  0 inside,
over the full 12 x 3 mm domain (style of Document/make_traj_figure.py); the dashed square is the scan region of that layer. Above each map:
the ROI mean |delta| (= the EoH deviation d of the reward). One episode (cooling time
--cool_time) and, for multi-seed methods, seed --seed, from rl_narrow/results_real/<slug>_fields.npz.

    python -m rl_narrow.plot_maps --results_dir rl_narrow/results_real --out Document/figures/fig_rl_maps
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_model_v3.paper_figure_ood import PAPER_RC
from baselines.common.data_utils import load_mesh_nodes

DEV = LinearSegmentedColormap.from_list("dev", ["#1d3f8f", "#5f86c9", "#c9d7ef", "#ffffff", "#f4c2b8", "#d9634f", "#8e1d1b"])
# (family, label, group) -- Kalman filter deliberately not shown (as in plot_results.py)
FAMILIES = [("PG shaped (ours)", "Online PG", "online"), ("Online Q shaped", "Online\nQ-learning", "online"),
            ("Offline FQI", "Offline\nQ-learning", "offline"), ("Offline CQL", "Offline\nCQL", "offline"),
            ("Offline IQL", "Offline\nIQL", "offline"), ("Proportional", "Proportional", "offline"),
            ("Constant", None, "offline")]
GROUPS = [("offline", "Offline methods (baselines)", "#2f4b9a"),
          ("online", "Digital twin-enabled online RL", "#9b2a22")]


def find(results_dir, family, seed):
    """-> (label, fields npz path, average reward of the family) ; Constant = best power in hindsight."""
    runs = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        r = json.load(open(f))
        if r.get("rows"):
            runs.append((f, r))
    score = lambda r: np.mean([x["score_heat"] for x in r["rows"]])
    if family == "Constant":
        c = [(f, r) for f, r in runs if r["type"] == "const"]
        if not c:
            return None
        f, r = max(c, key=lambda fr: score(fr[1]))
        return f"Constant\n({float(r['path']):.0f} W)", f.replace(".json", "_fields.npz"), score(r)
    fam_runs = [(f, r) for f, r in runs if re.sub(r"\s*seed\s*\d+\s*$", "", r["method"]).strip() == family]
    if not fam_runs:
        return None
    pick = [(f, r) for f, r in fam_runs if (re.search(r"seed\s*(\d+)\s*$", r["method"]) is None
                                            or int(re.search(r"seed\s*(\d+)\s*$", r["method"]).group(1)) == seed)]
    f = (pick or fam_runs)[0][0]
    return None, f.replace(".json", "_fields.npz"), float(np.mean([score(r) for _f, r in fam_runs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="rl_narrow/results_real")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])   # layer 1 starts cold for every method
    ap.add_argument("--cool_time", type=float, default=0.11)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T_l", type=float, default=2000.0)
    ap.add_argument("--T_h", type=float, default=2800.0)
    ap.add_argument("--vlim", type=float, default=1500.0, help="Color limit of the signed deviation [K].")
    ap.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
    ap.add_argument("--out", default="Document/figures/fig_rl_maps")
    args = ap.parse_args()

    nodes = load_mesh_nodes(args.mesh_path)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    W_DOM, H_DOM, L = 12.0, 3.0, 12
    side = lambda l: min(W_DOM, H_DOM) * (0.4 + 0.1 * (l - 1) / (L - 1))

    # rows: offline group on top, online group at the bottom; within a group worst -> best (best nearest the bottom)
    found = []
    for fam, label, grp in FAMILIES:
        x = find(args.results_dir, fam, args.seed)
        if x is not None:
            found.append((label or x[0], x[1], x[2], grp))
    rows = []
    for grp, _t, _c in GROUPS:
        rows += sorted([r for r in found if r[3] == grp], key=lambda r: r[2])
    nR, nC = len(rows), len(args.layers)
    n_off = sum(r[3] == "offline" for r in rows)

    plt.rcParams.update(PAPER_RC)                                       # same fonts as the other paper figures
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})
    saved = dict(layers=np.array(args.layers), labels=np.array([r[0] for r in rows]),
                 groups=np.array([r[3] for r in rows]), avg_reward=np.array([r[2] for r in rows]),
                 cool_time=args.cool_time, seed=args.seed, T_l=args.T_l, T_h=args.T_h, vlim=args.vlim,
                 nodes=nodes, triangles=tri.triangles)
    gap = 0.16                                                          # spacer row between the two groups
    hr = [1.0] * n_off + ([gap] if 0 < n_off < nR else []) + [1.0] * (nR - n_off)
    fig = plt.figure(figsize=(7.2, 0.55 + 0.98 * nR + 0.2))
    gs = fig.add_gridspec(len(hr), nC, height_ratios=hr, hspace=0.14, wspace=0.08)
    grow = lambda i: i if i < n_off else i + (1 if 0 < n_off < nR else 0)
    axes = np.empty((nR, nC), dtype=object)
    for i in range(nR):
        for j in range(nC):
            axes[i, j] = fig.add_subplot(gs[grow(i), j], sharex=axes[0, 0] if i or j else None,
                                         sharey=axes[0, 0] if i or j else None)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-args.vlim, vmax=args.vlim)
    for i, (label, path, _score, grp) in enumerate(rows):
        z = np.load(path)
        e = int(np.argmin(np.abs(z["cool_times"] - args.cool_time)))
        online = grp == "online"
        gcol = dict((g, c) for g, _t, c in GROUPS)[grp]
        for j, l in enumerate(args.layers):
            ax = axes[i, j]
            u = z["u_heat"][e, l - 1]
            dlt = np.where(u > args.T_h, u - args.T_h, np.where(u < args.T_l, u - args.T_l, 0.0))
            h = ax.tripcolor(tri, dlt, cmap=DEV, norm=norm, shading="gouraud")
            sd = side(l)
            ax.add_patch(Rectangle((W_DOM / 2 - sd / 2, H_DOM / 2 - sd / 2), sd, sd, fill=False, ec="k",
                                   lw=0.6, ls=(0, (2.0, 1.4))))
            m = ((np.abs(nodes[:, 0] - W_DOM / 2) <= sd / 2) & (np.abs(nodes[:, 1] - H_DOM / 2) <= sd / 2))
            saved[f"delta_r{i}_l{l}"] = dlt.astype(np.float32)
            saved[f"dbar_r{i}_l{l}"] = float(np.abs(dlt[m]).mean())
            ax.text(0.97, 0.06, f"$\\bar d$ = {np.abs(dlt[m]).mean():.0f} K", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7, color=gcol if online else "#1f2329",
                    fontweight="bold" if online else "normal", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
            ax.set_aspect("auto")
            ax.set_xticks([0, 6, 12]); ax.set_yticks([0, 1.5, 3])
            if i == 0:
                ax.set_title(f"Layer {l}", fontsize=9.5, fontweight="bold", pad=8)
            if j == 0:
                ax.set_ylabel("Y [mm]")
                ax.annotate(label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-34, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8, color=gcol if online else "#1f2329",
                            fontweight="bold" if online else "normal", linespacing=1.1)
            else:
                ax.tick_params(labelleft=False)
            if i == nR - 1:
                ax.set_xlabel("X [mm]")
            else:
                ax.tick_params(labelbottom=False)
            if online:
                for sp in ax.spines.values():
                    sp.set_color(gcol); sp.set_linewidth(1.3)
    cb = fig.colorbar(h, ax=list(axes.ravel()), pad=0.02, shrink=0.55, aspect=28)
    cb.set_ticks([-args.vlim, -args.vlim / 2, 0, args.vlim / 2, args.vlim])
    cb.set_label("Deviation from target window [K]\n(< 0: insufficient heating, > 0: overheating)")
    # group headers: coloured bar + rotated title left of the row labels (placed after the colorbar layout)
    for grp, title, col in GROUPS:
        idx = [i for i, r in enumerate(rows) if r[3] == grp]
        if not idx:
            continue
        y0, y1 = axes[idx[-1], 0].get_position().y0, axes[idx[0], 0].get_position().y1
        x = axes[0, 0].get_position().x0 - 1.12 / fig.get_figwidth()      # fixed 1.12 in left of the maps
        fig.add_artist(plt.Line2D([x, x], [y0 + 0.004, y1 - 0.004], transform=fig.transFigure, color=col, lw=2.2))
        fig.text(x - 0.012, (y0 + y1) / 2, title, rotation=90, ha="right", va="center", fontsize=8.5,
                 fontweight="bold", fontstyle="italic", color=col)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    np.savez_compressed(f"{args.out}_data.npz", **saved)          # everything drawn, for restyling later
    print("saved", args.out, "(+ _data.npz) | rows:", [r[0].replace(chr(10), " ") for r in rows])


if __name__ == "__main__":
    main()
