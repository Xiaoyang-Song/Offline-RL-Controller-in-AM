"""
surrogate_model_v3/paper_figure_propagation.py
--------------------------------------------------------
Single, paper-ready figure summarising what Jacobian propagation does to the
diagonal (rank=0) surrogate's combined two-stage uncertainty. Reads the
`propagation_maps.npz` written by plot_propagation_maps.py — no model / GPU.

Layout (double-column width, 2 rows x 3 panels):
  (a)-(c) σ_prop/σ_naive maps of the combined total σ (test-set mean) at three
          representative layers, zoomed on the laser ROI (scan square) at true
          aspect, faint FE mesh, one shared diverging colourbar (log-symmetric
          around 1). Titles: ROI / outside-ROI aggregated ratio.
  (d)     centreline profile (y = domain centre) of naive vs. propagated total
          σ [K] at --profile_layer, ROI extent shaded. With --profile_3d: a 3D
          surface of total σ over the part instead (propagated = shaded surface,
          naive = translucent cap where it exceeds propagated), saved as
          <out>_3d.{pdf,png}.
  (e)     per-layer pooled ratio inside vs. outside the ROI (total solid, epistemic
          dashed), shaded: percentile-bootstrap 95 % CI over test trajectories.
  (f)     per-trajectory in-ROI total ratio, median + IQR band (gradient), laser power
          inside vs. outside the training range (if the npz has an id_range).

Colour semantics (one meaning per colour, colour-blind-safe hues):
  blue / red       — damped / amplified (maps), inside / outside ROI (e)
  teal / gold      — propagated / naive sum (d)
  slate / wine     — laser power in / out of training range (f)

Usage
-----
    python -m surrogate_model_v3.paper_figure_propagation \\
        --npz surrogate_model_v3/results_propagation_maps_narrow/propagation_maps.npz \\
        --out surrogate_model_v3/results_propagation_maps_narrow/paper_fig_propagation [--profile_3d]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.plot_propagation_maps import load_mesh, roi_masks, scan_squares

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
COOL_D, COOL, COOL_L = "#17215e", "#3b55c4", "#9fb0ea"            # damped / inside ROI (cooling blue)
WARM_L, WARM, WARM_D = "#efa99b", "#c4352d", "#5e0c16"            # amplified / outside ROI (heating red)
NEUTRAL = "#f0efec"
PROP, PROP_L, PROP_D = "#1f7a70", "#bfe3dc", "#0f4740"            # propagated (teal)
NAIVE, NAIVE_L = "#c9971a", "#f0d58c"                              # naive sum (gold)
ID_C, OOD_C = "#2f3e4e", "#a23b5e"                                  # laser power in / out of range
INK, INK_2, INK_3 = "#1f1f1f", "#55545a", "#9a99a0"
GRID = "#ebeaee"
DIVERGING = LinearSegmentedColormap.from_list(
    "damp_amplify", [COOL_D, COOL, COOL_L, NEUTRAL, WARM_L, WARM, WARM_D])

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
DASH = (0, (3.2, 1.6))


def _panel_label(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def _region_ratio(mv, roi, q, inside):
    T = roi.shape[0]
    m = roi if inside else ~roi
    return np.array([np.sqrt(mv[f"prop_{q}"][t][m[t]].sum() / mv[f"naive_{q}"][t][m[t]].sum())
                     for t in range(T)])


def _bootstrap_pooled_ci(z, q, reg, n_boot=2000, seed=0, level=95.0):
    """Percentile bootstrap CI of the pooled ratio sqrt(Σ prop / Σ naive) over test
    trajectories (resampled with replacement), per layer. Needs the per-trajectory
    region sums saved by plot_propagation_maps.py (per_traj_sum_*); returns None if absent."""
    kp, kn = f"per_traj_sum_prop_{q}_{reg}", f"per_traj_sum_naive_{q}_{reg}"
    if kp not in z.files:
        return None
    P, N = z[kp], z[kn]                                     # (T, n_traj)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P.shape[1], size=(n_boot, P.shape[1]))
    r = np.sqrt(P[:, idx].sum(-1) / N[:, idx].sum(-1))      # (T, n_boot)
    lo, hi = np.percentile(r, [(100 - level) / 2, 100 - (100 - level) / 2], axis=1)
    return lo, hi


def _gradient_band(ax, x, centre, lo, hi, color, n_steps=12, alpha=0.035, zorder=2):
    """Band from lo to hi drawn as a soft gradient, densest at `centre`."""
    for k in range(1, n_steps + 1):
        f = k / n_steps
        ax.fill_between(x, centre - f * (centre - lo), centre + f * (hi - centre), color=color,
                        alpha=alpha, linewidth=0, zorder=zorder)


def _sigma_fields(mv, state_std, shade, t, X, Y, cubic=True):
    """Total σ [K] of naive and propagated, interpolated onto (X, Y)."""
    out = {}
    for tag in ("naive", "prop"):
        v = np.sqrt(mv[f"{tag}_total"][t]) * state_std
        itp = mtri.CubicTriInterpolator(shade, v, kind="min_E") if cubic else mtri.LinearTriInterpolator(shade, v)
        out[tag] = np.clip(np.ma.filled(itp(X, Y), np.nan), 0, None)
    return out


def _style_line_axes(ax, T):
    ax.axhline(1.0, color=INK_3, linewidth=0.7, zorder=1)
    ax.set_xticks([1, 3, 6, 9, 12]); ax.set_xlim(0.5, T + 0.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Layer")


# -----------------------------------------------------------------------------
# (d) variants
# -----------------------------------------------------------------------------

def _profile_panel(fig, spec, mv, state_std, shade, squares, a):
    ax = fig.add_subplot(spec)
    t = a.profile_layer - 1
    cx, cy, hs = squares[t]
    xs = np.linspace(*a.profile_window, 500)
    Z = _sigma_fields(mv, state_std, shade, t, xs, np.full_like(xs, cy), cubic=False)
    ax.fill_between(xs, Z["prop"], Z["naive"], where=Z["naive"] > Z["prop"], color=NAIVE_L, alpha=0.75,
                    linewidth=0, interpolate=True, zorder=2)
    ax.plot(xs, Z["naive"], color=NAIVE, linewidth=1.3, linestyle=DASH, zorder=3)
    ax.plot(xs, Z["prop"], color=PROP, linewidth=1.5, zorder=4)
    # peak reduction annotation
    i_n, i_p = np.nanargmax(Z["naive"]), np.nanargmax(Z["prop"])
    pk_n, pk_p = Z["naive"][i_n], Z["prop"][i_p]
    ax.annotate("", xy=(xs[i_p], pk_p), xytext=(xs[i_p], pk_n),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.7, mutation_scale=6), zorder=5)
    ax.text(xs[i_p] + 0.07, pk_n * 0.99, f"$-${100 * (1 - pk_p / pk_n):.0f}%", fontsize=6.5,
            color=INK_2, va="bottom", ha="left")
    # ROI: faint dashed edges + a labelled bracket along the bottom
    ytop = pk_n * 1.12
    for xe in (cx - hs, cx + hs):
        ax.axvline(xe, color=INK_3, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    yb, tick = 0.05 * ytop, 0.018 * ytop
    ax.plot([cx - hs, cx - hs, cx + hs, cx + hs], [yb + tick, yb, yb, yb + tick], color=INK_2,
            linewidth=0.7, zorder=5, solid_joinstyle="miter")
    ax.text(cx, yb + 0.012 * ytop, "ROI", ha="center", va="bottom", fontsize=6.5, color=INK_2, zorder=5)
    ax.set_xlim(*a.profile_window); ax.set_ylim(0, ytop)
    ax.set_xlabel(r"$x$ [mm]  (centreline $y = %.1f$ mm)" % cy)
    ax.set_ylabel(r"Total $\sigma$ [K]")
    ax.set_title(f"Centreline profile, layer {a.profile_layer}", pad=3, color=INK_2)
    ax.legend(handles=[Line2D([], [], color=NAIVE, linewidth=1.3, linestyle=DASH, label="Naive sum"),
                       Line2D([], [], color=PROP, linewidth=1.5, label="Propagated"),
                       Patch(facecolor=NAIVE_L, alpha=0.75, label="Removed")],
              loc="upper left", handlelength=1.6, borderaxespad=0.2, fontsize=6.5)
    _panel_label(ax, "d", x=-0.18)


def _surface_panel(fig, spec, mv, state_std, shade, squares, a):
    """(d) as a 3D view of total σ [K] over the part at --profile_layer.
    Propagated = teal shaded surface; naive = translucent gold cap drawn only
    where it exceeds the propagated surface. Both live in ONE Poly3DCollection
    so matplotlib depth-sorts their faces together (separate surfaces would be
    drawn wholesale one over the other)."""
    t = a.profile_layer - 1
    cx, cy, hs = squares[t]
    y0, y1 = shade.y.min(), shade.y.max()
    X, Y = np.meshgrid(np.linspace(*a.profile_window, 91), np.linspace(y0, y1, 49))
    Z = _sigma_fields(mv, state_std, shade, t, X, Y)
    zmax = np.nanmax(Z["naive"])

    def quads(Zs):
        v = np.stack([X, Y, Zs], -1)
        q = np.stack([v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]], axis=2)     # (ny-1, nx-1, 4, 3)
        return q.reshape(-1, 4, 3)

    # lit colours for the propagated surface
    ls = LightSource(azdeg=315, altdeg=40)
    ramp = LinearSegmentedColormap.from_list("prop", [PROP_L, "#6fbfb2", PROP, PROP_D])
    zc = 0.25 * (Z["prop"][:-1, :-1] + Z["prop"][:-1, 1:] + Z["prop"][1:, 1:] + Z["prop"][1:, :-1])
    rgb = ramp(np.clip(zc / zmax, 0, 1))[..., :3]
    lit = ls.shade_rgb(rgb, zc, vert_exag=0.15, blend_mode="soft")
    prop_faces = quads(Z["prop"])
    prop_cols = np.concatenate([lit.reshape(-1, 3), np.full((lit.size // 3, 1), 0.78)], 1)   # slightly translucent

    gap = 0.25 * ((Z["naive"] - Z["prop"])[:-1, :-1] + (Z["naive"] - Z["prop"])[:-1, 1:]
                  + (Z["naive"] - Z["prop"])[1:, 1:] + (Z["naive"] - Z["prop"])[1:, :-1])
    show = (gap > 0.02 * zmax).reshape(-1)
    naive_faces = quads(Z["naive"])[show]
    naive_cols = np.tile(to_rgba(NAIVE, 0.42), (naive_faces.shape[0], 1))

    ax = fig.add_subplot(spec, projection="3d")
    coll = Poly3DCollection(np.concatenate([prop_faces, naive_faces]),
                            facecolors=np.concatenate([prop_cols, naive_cols]),
                            edgecolors=np.concatenate([np.tile(to_rgba(PROP_D, 0.08), (len(prop_faces), 1)),
                                                       np.tile(to_rgba(NAIVE, 0.55), (len(naive_faces), 1))]),
                            linewidths=0.15)
    # ROI: square on the floor (z = 0), drawn BEFORE the surface so it sits underneath it; the
    # propagated surface is slightly translucent, so the square shows through where it is covered.
    ax.computed_zorder = False
    x0_, x1_, y0_, y1_ = cx - hs, cx + hs, cy - hs, cy + hs
    sq_x, sq_y = [x0_, x1_, x1_, x0_, x0_], [y0_, y0_, y1_, y1_, y0_]
    ax.plot(sq_x, sq_y, np.zeros(5), color=INK, linewidth=1.3, linestyle=(0, (3, 1.6)), zorder=1)
    ax.add_collection3d(coll)
    coll.set_zorder(2)
    ax.text(x0_ - 0.12, y0_ - 0.05, 0, "ROI", fontsize=7, color=INK, ha="right", va="top", zorder=3)
    ax.set_xlim(*a.profile_window); ax.set_ylim(y0, y1); ax.set_zlim(0, zmax * 1.02)
    ax.set_box_aspect((a.profile_window[1] - a.profile_window[0], (y1 - y0), 2.6), zoom=1.12)
    ax.view_init(elev=24, azim=-122)   # z axis on the left, away from (e)
    ax.set_xlabel(r"$x$ [mm]", labelpad=-6); ax.set_ylabel(r"$y$ [mm]", labelpad=-7)
    ax.tick_params(labelsize=6, pad=-3)
    ax.set_yticks([y0, (y0 + y1) / 2, y1]); ax.set_yticklabels([f"{v:g}" for v in (y0, (y0 + y1) / 2, y1)])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.4)
    ax.legend(handles=[Patch(facecolor=PROP, label="Propagated"),
                       Patch(facecolor=NAIVE, alpha=0.5, edgecolor=NAIVE, label="Naive sum (excess)")],
              loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=6.5, handlelength=1.2)
    ax.set_title(f"Total $\\sigma$ [K], layer {a.profile_layer}", pad=-2, color=INK_2)
    ax.text2D(-0.02, 1.04, "(d)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
    p.add_argument("--layers", type=int, nargs=3, default=[1, 6, 12], help="Layers for panels (a)-(c).")
    p.add_argument("--profile_layer", type=int, default=6, help="Layer for panel (d).")
    p.add_argument("--x_window", type=float, nargs=2, default=[2.0, 10.0], help="Zoom window in x for (a)-(c).")
    p.add_argument("--profile_window", type=float, nargs=2, default=[3.5, 8.5], help="x range of panel (d).")
    p.add_argument("--profile_3d", action="store_true",
                   help="Draw (d) as a 3D surface of total σ over the part (propagated surface, naive "
                        "excess as a translucent cap) instead of the centreline profile; written to "
                        "<out>_3d.{pdf,png}.")
    p.add_argument("--out", required=True, help="Output path without extension (.pdf and .png written).")
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)

    z = np.load(a.npz)
    mv = {k[len("all_"):]: z[k] for k in z.files if k.startswith("all_")}
    state_std = z["state_std"]
    per_traj_total, per_traj_lp = z["per_traj_total"], z["per_traj_lp"]
    id_range = tuple(z["id_range"]) if "id_range" in z.files else None
    T = per_traj_lp.shape[0]

    shade, edges, nodes = load_mesh(a.mesh_path)
    squares = scan_squares(nodes, T)
    roi = roi_masks(nodes, squares)
    ratio = np.sqrt(mv["prop_total"] / np.maximum(mv["naive_total"], 1e-30))
    lr = np.log2(np.clip(ratio, 1e-6, None))
    lim = max(np.percentile(np.abs(lr[[L - 1 for L in a.layers]]), 99.5), np.log2(1.05))
    norm = Normalize(-lim, lim)
    refiner = mtri.UniformTriRefiner(shade)

    # ---------------- layout -------------------------------------------------
    W = 7.0
    y0, y1 = nodes[1].min(), nodes[1].max()
    map_w = (W - 0.75) / 3
    map_h = map_w * (y1 - y0) / (a.x_window[1] - a.x_window[0])
    fig = plt.figure(figsize=(W, map_h + 2.45))
    gs = fig.add_gridspec(2, 3, height_ratios=[map_h, 1.75], hspace=0.62, wspace=0.36,
                          left=0.07, right=0.90, top=0.95, bottom=0.10)

    # ---------------- (a)-(c) ratio maps -----------------------------------
    for j, L in enumerate(a.layers):
        ax = fig.add_subplot(gs[0, j])
        # smooth display: cubic interpolation on a 3x-refined mesh (display only)
        tri_f, z_f = refiner.refine_field(lr[L - 1], subdiv=3,
                                          triinterpolator=mtri.CubicTriInterpolator(shade, lr[L - 1], kind="min_E"))
        z_f = np.clip(z_f, -lim, lim)
        tpc = ax.tripcolor(tri_f, z_f, cmap=DIVERGING, norm=norm, shading="gouraud", rasterized=True)
        ax.triplot(edges, color=INK, linewidth=0.2, alpha=0.16)   # FE mesh (corner triangles), faint
        cx, cy, hs = squares[L - 1]
        ax.add_patch(Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs, fill=False, edgecolor=INK,
                               linewidth=0.8, linestyle=(0, (2.5, 1.5))))
        if j == 0:
            ax.text(cx - hs - 0.1, cy + hs, "ROI", ha="right", va="top", fontsize=6.5, color=INK)
        ax.set_xlim(*a.x_window); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(np.ceil(a.x_window[0]), a.x_window[1] + 1e-9, 2.0))
        ax.set_yticks([y0, (y0 + y1) / 2, y1])
        ax.set_yticklabels([f"{v:g}" for v in (y0, (y0 + y1) / 2, y1)])
        ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        ax.set_xlabel(r"$x$ [mm]", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$y$ [mm]", labelpad=1)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(INK_2); s.set_linewidth(0.5)
        r_in = _region_ratio(mv, roi, "total", True)[L - 1]
        r_out = _region_ratio(mv, roi, "total", False)[L - 1]
        ax.set_title(f"Layer {L}:  ROI {r_in:.2f},  outside {r_out:.2f}", pad=4, color=INK_2)
        _panel_label(ax, "abc"[j], x=-0.06, y=1.10)
    # shared colourbar, aligned with the map row
    pos = fig.add_subplot(gs[0, 2]).get_position(); fig.axes[-1].remove()
    cax = fig.add_axes([0.915, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(tpc, cax=cax)
    ticks = np.array([0.7, 0.8, 1.0, 1.25, 1.4])
    ticks = ticks[np.abs(np.log2(ticks)) <= lim + 1e-9]
    cb.set_ticks(np.log2(ticks)); cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.5, length=2, labelsize=6.5)
    cb.set_label(RATIO, labelpad=2)

    # ---------------- (d) ----------------------------------------------------
    (_surface_panel if a.profile_3d else _profile_panel)(fig, gs[1, 0], mv, state_std, shade, squares, a)

    # ---------------- (e) inside vs outside across layers -------------------
    ax = fig.add_subplot(gs[1, 1])
    layers = np.arange(1, T + 1)
    for inside, col, mk in ((True, COOL, "o"), (False, WARM, "s")):
        for q, a_band in (("total", 0.04), ("epi", 0.025)):      # bootstrap 95 % CI of the pooled ratio
            ci = _bootstrap_pooled_ci(z, q, "in" if inside else "out")
            if ci is not None:
                _gradient_band(ax, layers, _region_ratio(mv, roi, q, inside), ci[0], ci[1], col, alpha=a_band)
        ax.plot(layers, _region_ratio(mv, roi, "total", inside), color=col, marker=mk, markersize=3.6,
                markeredgecolor="white", markeredgewidth=0.5, linewidth=1.5, zorder=3)
        ax.plot(layers, _region_ratio(mv, roi, "epi", inside), color=col, marker=mk, markersize=3.2,
                markerfacecolor="white", markeredgewidth=0.8, linewidth=1.0, linestyle=DASH, zorder=3)
    handles = [Line2D([], [], color=COOL, marker="o", markersize=3.6, linewidth=1.5, label="Inside ROI"),
               Line2D([], [], color=WARM, marker="s", markersize=3.6, linewidth=1.5, label="Outside ROI"),
               Line2D([], [], color=INK_2, linewidth=1.5, label="Total"),
               Line2D([], [], color=INK_2, linewidth=1.0, linestyle=DASH, label="Epistemic")]
    ax.legend(handles=handles, loc="center", ncols=2, handlelength=2.0, columnspacing=1.0,
              bbox_to_anchor=(0.5, 0.60), fontsize=6.5)
    _style_line_axes(ax, T)
    ax.set_ylabel(RATIO)
    ax.set_title("ROI vs. rest of part (95% CI)", pad=3, color=INK_2)
    _panel_label(ax, "e", x=-0.18)
    e_ax = ax

    # ---------------- (f) ID vs OOD per trajectory ---------------------------
    ax = fig.add_subplot(gs[1, 2], sharey=e_ax)
    if id_range is not None:
        id_mask = (per_traj_lp >= id_range[0]) & (per_traj_lp <= id_range[1])
        groups = [(f"In range ({id_range[0]:.0f}–{id_range[1]:.0f} W)", id_mask, ID_C, "-", "o"),
                  ("Out of range", ~id_mask, OOD_C, "-", "D")]
    else:
        groups = [("All test trajectories", np.ones_like(per_traj_lp, dtype=bool), ID_C, "-", "o")]
    for lab, mask, col, ls_, mk in groups:
        p25, p50, p75 = (np.array([np.percentile(per_traj_total[t][mask[t]], p) if mask[t].any()
                                   else np.nan for t in range(T)]) for p in (25, 50, 75))
        # one IQR band, drawn as a soft gradient: dense at the median, fading to the 25/75 % edges
        n_steps = 12
        for k in range(1, n_steps + 1):
            f = k / n_steps
            ax.fill_between(layers, p50 - f * (p50 - p25), p50 + f * (p75 - p50), color=col,
                            alpha=0.035, linewidth=0, zorder=2)
        ax.plot(layers, p50, color="white", linewidth=3.4, alpha=0.8, solid_capstyle="round", zorder=3)  # halo
        ax.plot(layers, p50, color=col, linestyle=ls_, linewidth=1.5, marker=mk, markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.5, label=lab, zorder=4)
    _style_line_axes(ax, T)
    ax.set_ylabel(RATIO)
    leg = ax.legend(loc="upper right", ncols=1, handlelength=1.8, fontsize=6.5, labelspacing=0.25,
                    title="Laser power (median, IQR band)", title_fontsize=6.5, borderaxespad=0.2)
    leg.get_title().set_color(INK_2)
    ax.set_title("Inside-ROI ratio by laser power", pad=3, color=INK_2)
    _panel_label(ax, "f", x=-0.18)

    out = a.out + ("_3d" if a.profile_3d else "")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    print(f"[paper_figure] Saved → {out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
