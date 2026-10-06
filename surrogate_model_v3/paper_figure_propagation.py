"""
surrogate_model_v3/paper_figure_propagation.py
--------------------------------------------------------
Single, paper-ready figure summarising what Jacobian propagation does to the
diagonal (rank=0) surrogate's combined two-stage uncertainty. Reads the
`propagation_maps.npz` written by plot_propagation_maps.py — no model / GPU.

Layout (double-column width, 2 rows x 3 panels):
  (a)-(c) σ_prop/σ_naive maps of the combined total σ (test-set mean) at three
          representative layers, zoomed on the scan region at true aspect,
          one shared diverging colourbar (log-symmetric around 1).
  (d)     centreline profile (y = domain centre) of naive vs. propagated total
          σ [K] at the middle layer, scan-square extent shaded.
  (e)     per-layer ratio inside vs. outside the scan square (total solid,
          epistemic dashed).
  (f)     per-trajectory in-square total ratio, median + IQR band, laser power
          inside vs. outside the training range (if the npz has an id_range).

Usage
-----
    python -m surrogate_model_v3.paper_figure_propagation \\
        --npz surrogate_model_v3/results_propagation_maps_narrow/propagation_maps.npz \\
        --out surrogate_model_v3/results_propagation_maps_narrow/paper_fig_propagation
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.plot_propagation_maps import load_mesh, roi_masks, scan_squares

# -----------------------------------------------------------------------------
# Theme: cool teal-blue (damped) <-> burnt orange (amplified), warm-gray neutral.
# Line colours reuse the two poles so "inside = damped = cool" reads the same
# in the maps and in the curves.
# -----------------------------------------------------------------------------
COOL, COOL_DARK = "#2c7fb8", "#0c3c66"
WARM, WARM_DARK = "#d9712b", "#7f2f05"
NEUTRAL = "#f2f1ee"
INK, INK_2, INK_3 = "#1f1f1f", "#5a5955", "#9a9893"
GRID = "#e7e5e0"
DIVERGING = LinearSegmentedColormap.from_list(
    "paper_div", [COOL_DARK, COOL, "#a6cbe3", NEUTRAL, "#f1c19b", WARM, WARM_DARK])

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
}
RATIO = r"$\sigma_{\mathrm{prop}}/\sigma_{\mathrm{naive}}$"


def _panel_label(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def _region_ratio(mv, roi, q, inside):
    T = roi.shape[0]
    m = roi if inside else ~roi
    return np.array([np.sqrt(mv[f"prop_{q}"][t][m[t]].sum() / mv[f"naive_{q}"][t][m[t]].sum())
                     for t in range(T)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
    p.add_argument("--layers", type=int, nargs=3, default=[1, 6, 12], help="Layers for panels (a)-(c).")
    p.add_argument("--profile_layer", type=int, default=6, help="Layer for the centreline profile (d).")
    p.add_argument("--x_window", type=float, nargs=2, default=[2.0, 10.0], help="Zoom window in x for (a)-(c).")
    p.add_argument("--profile_window", type=float, nargs=2, default=[3.5, 8.5], help="x range of the profile (d).")
    p.add_argument("--out", required=True, help="Output path without extension (.pdf and .png written).")
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)

    z = np.load(a.npz)
    mv = {k[len("all_"):]: z[k] for k in z.files if k.startswith("all_")}
    state_std = z["state_std"]
    per_traj_total, per_traj_lp = z["per_traj_total"], z["per_traj_lp"]
    id_range = tuple(z["id_range"]) if "id_range" in z.files else None
    T = per_traj_lp.shape[0]

    shade, _edges, nodes = load_mesh(a.mesh_path)
    squares = scan_squares(nodes, T)
    roi = roi_masks(nodes, squares)
    ratio = np.sqrt(mv["prop_total"] / np.maximum(mv["naive_total"], 1e-30))
    lr = np.log2(np.clip(ratio, 1e-6, None))
    lim = max(np.percentile(np.abs(lr[[L - 1 for L in a.layers]]), 99.5), np.log2(1.05))
    norm = Normalize(-lim, lim)

    # ---------------- layout -------------------------------------------------
    W = 7.0
    y0, y1 = nodes[1].min(), nodes[1].max()
    map_w = (W - 0.75) / 3
    map_h = map_w * (y1 - y0) / (a.x_window[1] - a.x_window[0])
    fig = plt.figure(figsize=(W, map_h + 2.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[map_h, 1.75], hspace=0.42, wspace=0.28,
                          left=0.07, right=0.90, top=0.95, bottom=0.10)

    # ---------------- (a)-(c) ratio maps -----------------------------------
    for j, L in enumerate(a.layers):
        ax = fig.add_subplot(gs[0, j])
        tpc = ax.tripcolor(shade, lr[L - 1], cmap=DIVERGING, norm=norm, shading="gouraud", rasterized=True)
        cx, cy, hs = squares[L - 1]
        ax.add_patch(Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs, fill=False, edgecolor=INK,
                               linewidth=0.7, linestyle=(0, (2.5, 1.5))))
        ax.set_xlim(*a.x_window); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(INK_3); s.set_linewidth(0.5)
        ax.set_title(f"Layer {L}", pad=3, color=INK_2)
        _panel_label(ax, "abc"[j])
        inside = np.sqrt(mv["prop_total"][L - 1][roi[L - 1]].sum() / mv["naive_total"][L - 1][roi[L - 1]].sum())
        ax.text(0.98, 0.04, f"in-square {inside:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=INK_2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    # shared colourbar, aligned with the map row
    pos = fig.add_subplot(gs[0, 2]).get_position(); fig.axes[-1].remove()
    cax = fig.add_axes([0.915, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(tpc, cax=cax)
    ticks = np.array([0.7, 0.8, 1.0, 1.25, 1.4])
    ticks = ticks[np.abs(np.log2(ticks)) <= lim + 1e-9]
    cb.set_ticks(np.log2(ticks)); cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.5, length=2, labelsize=6.5)
    cb.set_label(RATIO, labelpad=2)

    # ---------------- (d) centreline profile --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    t = a.profile_layer - 1
    cx, cy, hs = squares[t]
    xs = np.linspace(*a.profile_window, 400)
    curves = {}
    for tag in ("naive", "prop"):
        sig_k = np.sqrt(mv[f"{tag}_total"][t]) * state_std
        curves[tag] = mtri.LinearTriInterpolator(shade, sig_k)(xs, np.full_like(xs, cy))
    ax.axvspan(cx - hs, cx + hs, color=GRID, zorder=0, linewidth=0)
    ax.text(cx, 0.02, "scan\nsquare", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6.5, color=INK_2, linespacing=0.9)
    ax.fill_between(xs, curves["prop"], curves["naive"], where=curves["prop"] < curves["naive"],
                    color=COOL, alpha=0.25, linewidth=0, interpolate=True)
    ax.fill_between(xs, curves["prop"], curves["naive"], where=curves["prop"] >= curves["naive"],
                    color=WARM, alpha=0.25, linewidth=0, interpolate=True)
    ax.plot(xs, curves["naive"], color=INK_2, linewidth=1.0, linestyle=(0, (3, 1.5)), label="Naive sum")
    ax.plot(xs, curves["prop"], color=INK, linewidth=1.3, label="Propagated")
    ax.set_xlim(*a.profile_window); ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$x$ (centreline $y = %.1f$)" % cy)
    ax.set_ylabel(r"Total $\sigma$ [K]")
    ax.set_title(f"Centreline profile, layer {a.profile_layer}", pad=3, color=INK_2)
    ax.legend(loc="upper left", handlelength=1.5, borderaxespad=0.2)
    _panel_label(ax, "d", x=-0.18)

    # ---------------- (e) inside vs outside across layers -------------------
    ax = fig.add_subplot(gs[1, 1])
    layers = np.arange(1, T + 1)
    for inside, col, mk in ((True, COOL, "o"), (False, WARM, "s")):
        ax.plot(layers, _region_ratio(mv, roi, "total", inside), color=col, marker=mk, markersize=3,
                linewidth=1.3)
        ax.plot(layers, _region_ratio(mv, roi, "epi", inside), color=col, marker=mk, markersize=3,
                markerfacecolor="white", markeredgewidth=0.8, linewidth=1.0, linestyle=(0, (3, 1.5)))
    handles = [Line2D([], [], color=COOL, marker="o", markersize=3, linewidth=1.3, label="Inside square"),
               Line2D([], [], color=WARM, marker="s", markersize=3, linewidth=1.3, label="Outside square"),
               Line2D([], [], color=INK_2, linewidth=1.3, label="Total"),
               Line2D([], [], color=INK_2, linewidth=1.0, linestyle=(0, (3, 1.5)), label="Epistemic")]
    ax.legend(handles=handles, loc="center", ncols=2, handlelength=2.0, columnspacing=1.0,
              bbox_to_anchor=(0.5, 0.60))
    ax.set_xlabel("Layer"); ax.set_ylabel(RATIO)
    ax.set_title("Scan region vs. rest of part", pad=3, color=INK_2)
    _panel_label(ax, "e", x=-0.18)
    e_ax = ax

    # ---------------- (f) ID vs OOD per trajectory ---------------------------
    ax = fig.add_subplot(gs[1, 2], sharey=e_ax)
    if id_range is not None:
        id_mask = (per_traj_lp >= id_range[0]) & (per_traj_lp <= id_range[1])
        groups = [(f"In training range ({id_range[0]:.0f}–{id_range[1]:.0f} W)", id_mask, INK, "-"),
                  ("Outside training range", ~id_mask, INK_3, (0, (3, 1.5)))]
    else:
        groups = [("All test trajectories", np.ones_like(per_traj_lp, dtype=bool), INK, "-")]
    for lab, mask, col, ls in groups:
        q = [np.array([np.percentile(per_traj_total[t][mask[t]], p) if mask[t].any() else np.nan
                       for t in range(T)]) for p in (25, 50, 75)]
        ax.fill_between(layers, q[0], q[2], color=col, alpha=0.12, linewidth=0)
        ax.plot(layers, q[1], color=col, linestyle=ls, linewidth=1.3, label=lab)
    ax.set_xlabel("Layer")
    ax.tick_params(labelleft=False)
    ax.legend(loc="lower right", handlelength=2.0)
    ax.set_title("Inside square, per trajectory", pad=3, color=INK_2)
    ax.text(0.98, 0.97, "median, IQR band", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=INK_2)
    _panel_label(ax, "f", x=-0.10)

    for ax in (e_ax, ax):
        ax.axhline(1.0, color=INK_3, linewidth=0.6, zorder=0)
        ax.set_xticks([1, 3, 6, 9, 12]); ax.set_xlim(0.5, T + 0.5)
        ax.grid(axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}")
    plt.close(fig)
    print(f"[paper_figure] Saved → {a.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
