"""
rl_narrow/plot_maps.py
----------------------
Paper figure: real end-of-heating (EoH) deviation maps under each control scheme. Rows = methods (ours
at the bottom), columns = layers (default 1, 4, 8, 12). Each map shows the signed deviation of the real
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
DEFAULT_ROWS = ["Constant", "Kalman filter", "Offline FQI", "Offline CQL", "Offline IQL", "Online Q shaped",
                "PG shaped (ours)"]
LABELS = {"Offline FQI": "Offline\nQ-learning", "Offline CQL": "Offline\nCQL", "Offline IQL": "Offline\nIQL",
          "Online Q shaped": "Online\nQ-learning", "Kalman filter": "Kalman\nfilter",
          "PG shaped (ours)": "Online PG\n(ours)"}


def find(results_dir, family, seed):
    """-> (display label, fields npz path) for a family; Constant = best power in hindsight."""
    runs = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        r = json.load(open(f))
        if r.get("rows"):
            runs.append((f, r))
    if family == "Constant":
        c = [(f, r) for f, r in runs if r["type"] == "const"]
        if not c:
            return None
        f, r = max(c, key=lambda fr: np.mean([x["score_heat"] for x in fr[1]["rows"]]))
        return f"Constant\n({float(r['path']):.0f} W)", f.replace(".json", "_fields.npz")
    for f, r in runs:
        fam = re.sub(r"\s*seed\s*\d+\s*$", "", r["method"]).strip()
        s = re.search(r"seed\s*(\d+)\s*$", r["method"])
        if fam == family and (s is None or int(s.group(1)) == seed):
            return LABELS.get(family, family), f.replace(".json", "_fields.npz")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="rl_narrow/results_real")
    ap.add_argument("--rows", nargs="+", default=DEFAULT_ROWS, help="Method families, top to bottom.")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12])
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

    rows = [x for x in (find(args.results_dir, fam, args.seed) for fam in args.rows) if x is not None]
    nR, nC = len(rows), len(args.layers)
    plt.rcParams.update({"font.family": "serif", "font.size": 9})      # as Document/make_traj_figure.py
    saved = dict(layers=np.array(args.layers), labels=np.array([r[0] for r in rows]), cool_time=args.cool_time,
                 seed=args.seed, T_l=args.T_l, T_h=args.T_h, vlim=args.vlim, nodes=nodes, triangles=tri.triangles)
    fig, axes = plt.subplots(nR, nC, figsize=(7.4, 0.35 + 1.08 * nR), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.16, wspace=0.08))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-args.vlim, vmax=args.vlim)
    for i, (label, path) in enumerate(rows):
        z = np.load(path)
        e = int(np.argmin(np.abs(z["cool_times"] - args.cool_time)))
        ours = "ours" in label
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
                    va="bottom", fontsize=7, color="#c4352d" if ours else "k",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
            ax.set_aspect("auto")
            if i == 0:
                ax.set_title(f"Layer {l}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Y [mm]", fontsize=8)
                ax.text(-0.40, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=8.5,
                        color="#c4352d" if ours else "k", fontweight="bold" if ours else "normal", linespacing=1.1)
            if i == nR - 1:
                ax.set_xlabel("X [mm]")
            if ours:
                for sp in ax.spines.values():
                    sp.set_color("#c4352d"); sp.set_linewidth(1.3)
    cb = fig.colorbar(h, ax=axes, pad=0.015, shrink=0.6, aspect=30)
    cb.set_ticks([-args.vlim, -args.vlim / 2, 0, args.vlim / 2, args.vlim])
    cb.set_label("Deviation from target window [K]\n(< 0: insufficient heating, > 0: overheating)", fontsize=8)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=250, bbox_inches="tight")
    np.savez_compressed(f"{args.out}_data.npz", **saved)          # everything drawn, for restyling later
    print("saved", args.out, "(+ _data.npz) | rows:", [r[0].replace(chr(10), " ") for r in rows])


if __name__ == "__main__":
    main()
