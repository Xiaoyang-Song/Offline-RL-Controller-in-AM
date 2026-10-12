"""
surrogate_model_v3/paper_figure_ood.py
--------------------------------------------------------
Paper-ready OOD stress-test figure for a surrogate_model_v3 checkpoint trained
on a narrow laser-power range. Reads the npz written by

    python -m surrogate_model_v3.evaluate_ood --split test --propagate_uncertainty \\
        --dump_npz <...>.npz ...

(per-sample arrays under heat_samples/ and comb_samples/) — no model / GPU.

Layout (double-column width, 1 x 3):
  (a) total, epistemic and aleatoric σ [K] vs. laser power on ONE shared linear axis
      starting at 0 (mean and IQR band per tested power level), training range shaded — the shared
      scale shows the aleatoric term as the near-constant it is instead of
      auto-zooming onto a ~2 % variation.
  (b) prediction RMSE [K] vs. laser power (binned), training range shaded.
  (c) per-sample epistemic σ vs. per-sample RMSE on log-log axes, shown as each
      group's (in / out of training range) KDE density: filled contours enclosing
      the densest 50 % and 90 % of its samples, plus the group median.

--stage comb (default): combined heating+cooling transition (what RL reads;
propagated if the npz was produced with --propagate_uncertainty).
--stage heat: heating stage only.
--stage both: one figure with both (dashed = heating stage, solid = heating + cooling;
(c) shows only the 50 % regions plus an arrow from each group's heating-stage median
to its full-transition median).

Usage
-----
    python -m surrogate_model_v3.paper_figure_ood \\
        --npz surrogate_model_v3/results_ood_narrow/ood_narrow_test.npz \\
        --out surrogate_model_v3/results_ood_narrow/paper_fig_ood [--stage heat]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import gaussian_kde

# Same typography / ink as paper_figure_propagation.py
INK, INK_2, INK_3 = "#1f1f1f", "#55545a", "#9a99a0"
GRID = "#ebeaee"
# (a)/(b) palette: deep teal-charcoal / coral / saffron / plum — muted, print-safe, distinct in grayscale
EPI_C, ALE_C, TOT_C, ERR_C = "#e76f51", "#e9b44c", "#264653", "#6d597a"   # epistemic, aleatoric, total σ, RMSE
ID_C, OOD_C = "#2f3e4e", "#a23b5e"                            # in / out of training range
TRAIN_SHADE = "#f1f0ed"                                       # training-range band (neutral)

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.framealpha": 0.88, "legend.edgecolor": "none",
    "legend.fancybox": False, "legend.facecolor": "white",
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
}


def _panel_label(ax, letter, x=-0.18, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def _gradient_band(ax, x, centre, lo, hi, color, n_steps=12, alpha=0.035, zorder=2):
    """Band from lo to hi drawn as a soft gradient, densest at `centre`."""
    for k in range(1, n_steps + 1):
        f = k / n_steps
        ax.fill_between(x, centre - f * (centre - lo), centre + f * (hi - centre), color=color,
                        alpha=alpha, linewidth=0, zorder=zorder)


def _group_index(action, bin_width):
    """Group samples by laser power: one group per distinct power level when the dataset uses a
    small discrete set of levels (it does: 100-400 W in 10 W steps), otherwise equal-width bins.
    Returns (group index per sample, group x-positions)."""
    levels = np.unique(np.round(action, 6))
    if levels.size <= 60:
        return np.searchsorted(levels, np.round(action, 6)), levels
    edges = np.arange(np.floor(action.min() / bin_width) * bin_width, action.max() + bin_width, bin_width)
    idx = np.clip(np.digitize(action, edges[1:-1]), 0, len(edges) - 2)
    return idx, 0.5 * (edges[:-1] + edges[1:])


def _binned(idx, n_groups, values, stat="median", n_boot=1000, seed=0):
    """Per laser-power group:
      stat="median": median of the samples, band = their 25-75 % quantiles (spread across samples).
      stat="rms":    sqrt(mean(values)) — i.e. the RMSE when `values` are per-sample squared errors —
                     band = percentile-bootstrap 95 % CI of that RMSE (resampling the group's samples)."""
    rng = np.random.default_rng(seed)
    c, lo, hi = (np.full(n_groups, np.nan) for _ in range(3))
    for b in range(n_groups):
        v = values[idx == b]
        if v.size == 0:
            continue
        if stat == "rms":
            c[b] = np.sqrt(np.mean(v))
            boot = np.sqrt(v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1))
            lo[b], hi[b] = np.percentile(boot, [2.5, 97.5])
        else:
            lo[b], c[b], hi[b] = np.percentile(v, [25, 50, 75])
    return c, lo, hi


def _shade_training(ax, id_ranges, label=True):
    """Light neutral band + thin dashed edges + a labelled bracket just under the top of the axes
    (same bracket style as the ROI marker in paper_figure_propagation.py)."""
    tr = ax.get_xaxis_transform()
    for i, (lo, hi) in enumerate(id_ranges):
        ax.axvspan(lo, hi, color=TRAIN_SHADE, zorder=0, linewidth=0)
        for xe in (lo, hi):
            ax.axvline(xe, color=INK_3, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        if label:
            yb, tick = 0.94, 0.025
            ax.plot([lo, lo, hi, hi], [yb - tick, yb, yb, yb - tick], color=INK_2, linewidth=0.7,
                    transform=tr, zorder=5, solid_joinstyle="miter", clip_on=False)
            x0, x1 = ax.get_xlim()
            narrow = (hi - lo) / (x1 - x0) < 0.25          # short label when the band is narrow
            # keep labels of bands that touch the axis edge inside the axes
            if lo - x0 < 0.05 * (x1 - x0):
                xt, ha = lo, "left"
            elif x1 - hi < 0.05 * (x1 - x0):
                xt, ha = hi, "right"
            else:
                xt, ha = 0.5 * (lo + hi), "center"
            ax.text(xt, yb + 0.012, "Training" if narrow else "Training range", transform=tr,
                    ha=ha, va="bottom", fontsize=6.5, color=INK_2, zorder=5)


def _hdr_contour(ax, lx, ly, GX, GY, mass):
    """Gaussian-KDE density of (log σ, log RMSE) on the grid and the density level enclosing `mass`."""
    kde = gaussian_kde(np.vstack([lx, ly]))
    kde.set_bandwidth(kde.factor * 1.35)
    dens = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    level = np.percentile(kde(np.vstack([lx, ly])), 100 * (1 - mass))
    return dens, level


def draw_both(z, a):
    """One figure, both stages: heating stage (dashed) vs. full heating->cooling transition (solid)."""
    stages = {"heat": "heat_samples/", "comb": "comb_samples/"}
    for pre in stages.values():
        if pre + "epi_K" not in z.files:
            raise SystemExit(f"{a.npz} has no per-sample Kelvin arrays — regenerate with evaluate_ood.py.")
    action = z["comb_samples/action"]
    id_mask = z["id_mask"].astype(bool)
    id_ranges = [tuple(r) for r in z["id_ranges"]]
    gidx, centres = _group_index(action, a.bin_width)
    ng = len(centres)
    style = {"heat": dict(linestyle=(0, (3.2, 1.6)), linewidth=1.2, mfc="white"),
             "comb": dict(linestyle="-", linewidth=1.5, mfc=None)}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.42))

    # ---------------- (a) σ components, both stages ---------------------------
    ax = axes[0]
    ax.set_xlim(action.min() - 5, action.max() + 5)
    _shade_training(ax, id_ranges)
    ymax = 0.0
    for st, pre in stages.items():
        for key, col, mk in (("tot_K", TOT_C, "D"), ("epi_K", EPI_C, "o"), ("ale_K", ALE_C, "s")):
            c, lo, hi = _binned(gidx, ng, z[pre + key])
            ymax = max(ymax, np.nanmax(c))
            if st == "comb":
                _gradient_band(ax, centres, c, lo, hi, col, alpha=0.025)
            ax.plot(centres, c, color="white", linewidth=3.0, alpha=0.75, zorder=3)
            ax.plot(centres, c, color=col, marker=mk, markersize=2.0, markeredgewidth=0.6,
                    markeredgecolor=col, markerfacecolor=style[st]["mfc"] or col,
                    linestyle=style[st]["linestyle"], linewidth=style[st]["linewidth"], zorder=4)
    ax.set_ylim(0, ymax * 1.62)                              # headroom for the legend
    ax.set_ylabel(r"$\sigma$, node-averaged [K]")
    ax.set_title("Uncertainty (median, IQR)", pad=3, color=INK_2)
    comp = [Line2D([], [], color=c_, linewidth=1.6, label=l_) for c_, l_ in
            ((TOT_C, "Total"), (EPI_C, "Epistemic"), (ALE_C, "Aleatoric"))]
    stg = [Line2D([], [], color=INK_2, linewidth=1.2, linestyle=(0, (3.2, 1.6)), marker="o", markersize=2.6,
                  markerfacecolor="white", label="Heating stage"),
           Line2D([], [], color=INK_2, linewidth=1.5, label="Full transition")]
    # one legend, two columns: components (colour) | stages (line style)
    ax.legend(handles=comp + stg, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncols=2,
              handlelength=1.9, columnspacing=0.9, labelspacing=0.15, borderaxespad=0.2, fontsize=6.2)
    _panel_label(ax, "a")

    # ---------------- (b) RMSE, both stages -----------------------------------
    ax = axes[1]
    ax.set_xlim(action.min() - 5, action.max() + 5)
    _shade_training(ax, id_ranges)
    ymax = 0.0
    for st, pre in stages.items():
        c, lo, hi = _binned(gidx, ng, z[pre + "sq_err"], stat="rms")
        ymax = max(ymax, np.nanmax(hi))
        ax.fill_between(centres, lo, hi, color=ERR_C, alpha=0.28 if st == "comb" else 0.18, linewidth=0, zorder=2)
        ax.plot(centres, c, color="white", linewidth=3.0, alpha=0.75, zorder=3)
        ax.plot(centres, c, color=ERR_C, marker="D", markersize=2.0, markeredgewidth=0.6, markeredgecolor=ERR_C,
                markerfacecolor=style[st]["mfc"] or ERR_C, linestyle=style[st]["linestyle"],
                linewidth=style[st]["linewidth"], zorder=4,
                label="Heating stage" if st == "heat" else "Full transition")
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("RMSE [K]")
    ax.set_title("Prediction error (95% CI)", pad=3, color=INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), handlelength=2.0, labelspacing=0.2)
    _panel_label(ax, "b")

    for ax in axes[:2]:
        ax.set_xlabel("Laser power [W]")
        ax.set_xlim(action.min() - 5, action.max() + 5)
        ax.set_xticks(np.arange(np.ceil(action.min() / 100) * 100, action.max() + 1, 100))
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    # ---------------- (c) 50 % density regions + median shift, both stages ----
    ax = axes[2]
    pts = {}
    for st, pre in stages.items():
        e, r = z[pre + "epi_K"], np.sqrt(z[pre + "sq_err"])
        ok = (e > 0) & (r > 0)
        pts[st] = (np.log10(e), np.log10(r), ok)
    allx = np.concatenate([pts[s_][0][pts[s_][2]] for s_ in pts])
    ally = np.concatenate([pts[s_][1][pts[s_][2]] for s_ in pts])
    gx = np.linspace(allx.min() - 0.25, allx.max() + 0.25, 220)
    gy = np.linspace(ally.min() - 0.25, ally.max() + 0.25, 220)
    GX, GY = np.meshgrid(gx, gy)
    X, Y = 10 ** GX, 10 ** GY
    handles = []
    for lab, gm, col in (("InD", id_mask, ID_C), ("OoD", ~id_mask, OOD_C)):
        med = {}
        for st in stages:
            lx, ly, ok = pts[st]
            m = gm & ok
            dens, lv = _hdr_contour(ax, lx[m], ly[m], GX, GY, mass=0.5)
            if st == "comb":
                ax.contourf(X, Y, dens, levels=[lv, dens.max() * 1.01], colors=[col], alpha=0.30, zorder=2)
                ax.contour(X, Y, dens, levels=[lv], colors=[col], linewidths=1.0, zorder=4)
            else:
                ax.contour(X, Y, dens, levels=[lv], colors=[col], linewidths=0.9,
                           linestyles=[(0, (3.2, 1.6))], zorder=4)
            med[st] = (10 ** np.median(lx[m]), 10 ** np.median(ly[m]))
        ax.annotate("", xy=med["comb"], xytext=med["heat"], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=7,
                                    shrinkA=3, shrinkB=3))
        ax.plot(*med["heat"], marker="o", markersize=4.2, color="white", markeredgecolor=col,
                markeredgewidth=1.0, zorder=7)
        ax.plot(*med["comb"], marker="o", markersize=4.5, color=col, markeredgecolor="white",
                markeredgewidth=0.8, zorder=7)
        handles.append(Patch(facecolor=to_rgba(col, 0.30), edgecolor=col, linewidth=0.8, label=lab))
    handles += [Patch(facecolor="none", edgecolor=INK_2, linewidth=0.9, linestyle=(0, (3.2, 1.6)),
                      label="Heating stage"),
                Patch(facecolor=to_rgba(INK_2, 0.25), edgecolor=INK_2, linewidth=0.9,
                      label="Full transition")]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(10 ** gx[0], 10 ** gx[-1]); ax.set_ylim(10 ** gy[0], 10 ** (gy[-1] + 0.75))
    ax.set_xlabel(r"Epistemic $\sigma$ [K]")
    ax.set_ylabel("Per-sample RMSE [K]")
    ax.set_title("Epistemic $\\sigma$ vs. error (50% regions)", pad=3, color=INK_2)
    ax.legend(handles=handles, loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.2,
              fontsize=6.2)
    ax.grid(which="major", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    _panel_label(ax, "c")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}")
    plt.close(fig)
    print(f"[paper_figure_ood] checkpoint: {z['checkpoint']}  split: {z['split']}  "
          f"propagated: {bool(z['propagate_uncertainty'])}")
    print(f"[paper_figure_ood] Saved → {a.out}.{{pdf,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--stage", choices=["comb", "heat", "both"], default="comb",
                   help="comb: heating+cooling transition; heat: heating stage; both: one figure with both.")
    p.add_argument("--bin_width", type=float, default=25.0,
                   help="Bin width [W] for (a)/(b), only used if laser power is not a small discrete set of levels.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output path without extension (.pdf and .png written).")
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)

    z = np.load(a.npz, allow_pickle=False)
    if a.stage == "both":
        draw_both(z, a)
        return
    pre = f"{a.stage}_samples/"
    if pre + "epi_K" not in z.files:
        raise SystemExit(f"{a.npz} has no per-sample Kelvin arrays — regenerate it with the current "
                         f"evaluate_ood.py (--dump_npz).")
    action = z[pre + "action"]
    epi, ale = z[pre + "epi_K"], z[pre + "ale_K"]
    rmse_s = np.sqrt(z[pre + "sq_err"])                      # per-sample RMSE over nodes [K]
    id_mask = z["id_mask"].astype(bool)
    id_ranges = [tuple(r) for r in z["id_ranges"]]
    stage_word = "Combined" if a.stage == "comb" else "Heating-stage"

    gidx, centres = _group_index(action, a.bin_width)
    ng = len(centres)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.42))

    # ---------------- (a) epistemic & aleatoric σ on a shared axis -----------
    ax = axes[0]
    ax.set_xlim(action.min() - 5, action.max() + 5)
    _shade_training(ax, id_ranges)
    tot = z[pre + "tot_K"]
    for vals, col, mk, lab in ((tot, TOT_C, "D", "Total"), (epi, EPI_C, "o", "Epistemic"),
                               (ale, ALE_C, "s", "Aleatoric")):
        c, lo, hi = _binned(gidx, ng, vals)
        _gradient_band(ax, centres, c, lo, hi, col)
        ax.plot(centres, c, color="white", linewidth=3.2, alpha=0.8, zorder=3)
        ax.plot(centres, c, color=col, marker=mk, markersize=2.2, markeredgecolor=col,
                markeredgewidth=0, linewidth=1.5, label=lab, zorder=4)
    ax.set_ylim(0, np.nanmax(_binned(gidx, ng, tot)[2]) * 1.22)       # headroom for the bracket
    ax.set_ylabel(r"Mean $\sigma$ over nodes [K]")
    ax.set_title(f"{stage_word} uncertainty", pad=3, color=INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), handlelength=1.8, labelspacing=0.25)
    _panel_label(ax, "a")

    # ---------------- (b) RMSE vs laser power --------------------------------
    ax = axes[1]
    _shade_training(ax, id_ranges)
    c, _, _ = _binned(gidx, ng, z[pre + "sq_err"], stat="rms")
    ax.set_ylim(0, np.nanmax(c) * 1.18)
    ax.plot(centres, c, color="white", linewidth=3.2, alpha=0.8, zorder=3)
    ax.plot(centres, c, color=ERR_C, marker="D", markersize=2.2, markeredgecolor=ERR_C,
            markeredgewidth=0, linewidth=1.5, zorder=4)
    ax.set_ylabel("RMSE [K]")
    ax.set_title(f"{stage_word} prediction error", pad=3, color=INK_2)
    _panel_label(ax, "b")

    for ax in axes[:2]:
        ax.set_xlabel("Laser power [W]")
        ax.set_xlim(action.min() - 5, action.max() + 5)
        ax.set_xticks(np.arange(np.ceil(action.min() / 100) * 100, action.max() + 1, 100))
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    # ---------------- (c) epistemic σ vs error: density contours, log-log ------
    # Each group's joint density of (log σ, log RMSE) via a Gaussian KDE; filled contours enclose the
    # densest 50 % and 90 % of that group's samples (highest-density regions); dot = group median.
    ax = axes[2]
    pos = (epi > 0) & (rmse_s > 0)
    lx_all, ly_all = np.log10(epi[pos]), np.log10(rmse_s[pos])
    pad = 0.25
    gx = np.linspace(lx_all.min() - pad, lx_all.max() + pad, 220)
    gy = np.linspace(ly_all.min() - pad, ly_all.max() + pad, 220)
    GX, GY = np.meshgrid(gx, gy)
    groups = [("InD", id_mask & pos, ID_C), ("OoD", ~id_mask & pos, OOD_C)]
    handles = []
    for lab, m, col in groups:
        lx, ly = np.log10(epi[m]), np.log10(rmse_s[m])
        kde = gaussian_kde(np.vstack([lx, ly]))
        kde.set_bandwidth(kde.factor * 1.35)                # slightly smoother: no tail islands
        dens = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        at_pts = kde(np.vstack([lx, ly]))
        lv90, lv50 = np.percentile(at_pts, [10, 50])         # density thresholds enclosing 90 % / 50 %
        X, Y = 10 ** GX, 10 ** GY
        ax.contourf(X, Y, dens, levels=[lv90, lv50], colors=[col], alpha=0.16, zorder=2)
        ax.contourf(X, Y, dens, levels=[lv50, dens.max() * 1.01], colors=[col], alpha=0.38, zorder=3)
        ax.contour(X, Y, dens, levels=[lv90, lv50], colors=[col], linewidths=[0.6, 1.0], zorder=4)
        mx, my = 10 ** np.median(lx), 10 ** np.median(ly)
        ax.plot(mx, my, marker="o", markersize=4.5, color=col, markeredgecolor="white",
                markeredgewidth=0.8, zorder=6)
        handles.append(Patch(facecolor=to_rgba(col, 0.38), edgecolor=col, linewidth=0.8, label=lab))
    handles += [Patch(facecolor="none", edgecolor=INK_2, linewidth=1.0, label="50% / 90% of samples"),
                Line2D([], [], marker="o", linestyle="none", color=INK_2, markeredgecolor="white",
                       markersize=4.5, label="Median")]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(10 ** gx[0], 10 ** gx[-1]); ax.set_ylim(10 ** gy[0], 10 ** (gy[-1] + 0.6))
    ax.set_xlabel(r"Epistemic $\sigma$ [K]")
    ax.set_ylabel("Per-sample RMSE [K]")
    ax.set_title("Epistemic $\\sigma$ vs. error", pad=3, color=INK_2)
    ax.legend(handles=handles, loc="upper left", handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)
    ax.grid(which="major", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    _panel_label(ax, "c")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}")
    plt.close(fig)
    print(f"[paper_figure_ood] checkpoint: {z['checkpoint']}  split: {z['split']}  "
          f"propagated: {bool(z['propagate_uncertainty'])}")
    print(f"[paper_figure_ood] Saved → {a.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
