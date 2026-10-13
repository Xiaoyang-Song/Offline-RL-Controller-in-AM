"""
surrogate_model_v3/paper_figure_aleatoric_alt.py
--------------------------------------------------------
Alternative (non-bar) representations of the aleatoric-calibration result of
paper_figure_aleatoric.py (same data, same expected values, InD + OoD):

  --style dot     2 x 3 grid, log y: expected σ as a short horizontal tick,
                  learned σ as a dot, heating and full transition side by side
                  for each ε (a "Cleveland" dot plot; log scale keeps ε = 0.01
                  readable).
  --style ratio   2 x 3 grid, learned / expected per ε; the ideal is the line at
                  1, the shaded band is ±5 %. Points beyond the axis range are
                  pinned to the edge with their value written next to them.
  --style parity  1 x 2 (InD, OoD), log-log: learned vs expected σ for every
                  (setting, stage, ε); colour = stage, marker = setting; the
                  diagonal is perfect recovery, the band ±10 %.

  --combined      InD and OoD test samples pooled: every style becomes one row
                  of three panels (full range / narrow / gap).

    python -m surrogate_model_v3.paper_figure_aleatoric_alt --style dot \\
        --out surrogate_model_v3/results_aleatoric_calibration/alt/paper_fig_aleatoric_dot
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import surrogate_model_v3.paper_figure_aleatoric as base
from surrogate_model_v3.paper_figure_aleatoric import (SETTINGS, HEAT_C, FULL_C, _lighten, _load_setting, _save,
                                                       grid_2x3)
from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK_2, INK_3, GRID, _panel_label

STAGES = (("eps", "heat", HEAT_C, "Heating stage"), ("full_exp", "full", FULL_C, "Full transition"))
SET_MARK = {"full_range": "o", "narrow_200_300W": "s", "gap_100-150_350-400": "D"}


LOG_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def _plain_log(axis):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    axis.set_major_locator(FixedLocator(LOG_TICKS))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _cat_axis(ax, rows):
    ax.set_xticks(np.arange(len(rows))); ax.set_xticklabels([f"{r['eps']:g}" for r in rows])
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, which="major"); ax.set_axisbelow(True)


# ── dot ─────────────────────────────────────────────────────────────────────────
def draw_dot(ax, rows, off=0.17):
    rows = [r for r in rows if r["eps"] > 0]
    x = np.arange(len(rows))
    for k, (ek, lk, col, _) in enumerate(STAGES):
        xs = x + (2 * k - 1) * off
        ax.scatter(xs, [r[ek] for r in rows], marker="_", s=230, linewidths=2.4, color=_lighten(col, 0.55), zorder=2)
        ax.scatter(xs, [r[lk] for r in rows], marker="o", s=22, color=col, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(0.007, 0.5); _plain_log(ax.yaxis)
    _cat_axis(ax, rows)


def dot_handles():
    h = []
    for _, _, col, lab in STAGES:
        h.append(Line2D([], [], marker="_", markersize=13, markeredgewidth=2.4, linestyle="none",
                        color=_lighten(col, 0.55), label=f"{lab}, expected"))
        h.append(Line2D([], [], marker="o", markersize=5, linestyle="none", color=col, markeredgecolor="white",
                        markeredgewidth=0.6, label=f"{lab}, learned"))
    return h


# ── ratio ───────────────────────────────────────────────────────────────────────
def draw_ratio(ax, rows, off=0.13, lim=(0.88, 1.12)):
    rows = [r for r in rows if r["eps"] > 0]
    x = np.arange(len(rows))
    ax.axhspan(0.95, 1.05, color="#f1f0f3", zorder=0, linewidth=0)
    ax.axhline(1.0, color=INK_3, linewidth=0.8, zorder=1)
    for k, (ek, lk, col, _) in enumerate(STAGES):
        for xi, r in zip(x + (2 * k - 1) * off, rows):
            v = r[lk] / r[ek]
            if lim[0] <= v <= lim[1]:
                ax.scatter(xi, v, marker="o", s=24, color=col, edgecolor="white", linewidth=0.6, zorder=3)
            else:                                                # pin inside the edge, write the value
                top = v > lim[1]
                pad = 0.012 * (lim[1] - lim[0]) / 0.24
                yv = lim[1] - pad if top else lim[0] + pad
                ax.scatter(xi, yv, marker="^" if top else "v", s=30, color=col, zorder=3)
                ax.annotate(f"{v:.2f}", (xi, yv), xytext=(-5 if k == 0 else 5, 0), textcoords="offset points",
                            ha="right" if k == 0 else "left", va="center", color=col,
                            fontsize=plt.rcParams["font.size"] - 1.5)
    ax.set_ylim(*lim)
    _cat_axis(ax, rows)


def ratio_handles():
    from matplotlib.patches import Patch
    return [Line2D([], [], marker="o", markersize=5, linestyle="none", color=HEAT_C, markeredgecolor="white",
                   label="Heating stage"),
            Line2D([], [], marker="o", markersize=5, linestyle="none", color=FULL_C, markeredgecolor="white",
                   label="Full transition"),
            Line2D([], [], color=INK_3, linewidth=0.8, label="Perfect recovery"),
            Patch(facecolor="#f1f0f3", label="±5 %")]


# ── parity ──────────────────────────────────────────────────────────────────────
def plot_parity(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9), sharey=True, gridspec_kw=dict(wspace=0.1))
    lo, hi = 0.007, 0.5
    g = np.geomspace(lo, hi, 50)
    for j, (ax, (lab, subset)) in enumerate(zip(axes, (("InD", "id"), ("OoD", "ood")))):
        ax.fill_between(g, g * 0.9, g * 1.1, color="#f1f0f3", linewidth=0, zorder=0)
        ax.plot(g, g, color=INK_3, linewidth=0.8, zorder=1)
        for pre, _title in SETTINGS:
            rows = [r for r in _load_setting(root, pre, subset) if r["eps"] > 0]
            for ek, lk, col, _ in STAGES:
                ax.scatter([r[ek] for r in rows], [r[lk] for r in rows], marker=SET_MARK[pre], s=24, color=col,
                           edgecolor="white", linewidth=0.5, alpha=0.7, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        _plain_log(ax.xaxis); _plain_log(ax.yaxis)
        ax.set_aspect("equal")
        ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.set_xlabel(r"Expected aleatoric $\sigma$ (× state std)")
        ax.set_title(f"{lab} test samples", pad=3, color=INK_2)
        _panel_label(ax, "ab"[j], x=-0.2 if j == 0 else -0.06)
    axes[0].set_ylabel(r"Learned aleatoric $\sigma$ (× state std)")
    h = [Line2D([], [], marker="o", linestyle="none", color=HEAT_C, markersize=5, label="Heating stage"),
         Line2D([], [], marker="o", linestyle="none", color=FULL_C, markersize=5, label="Full transition")]
    h += [Line2D([], [], marker=SET_MARK[p], linestyle="none", color=INK_2, markersize=4.5, label=t.split(" (")[0])
          for p, t in SETTINGS]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=False, handletextpad=0.3)
    _save(fig, out)


# ── combined InD + OoD (all test samples): one row, one panel per setting ──────────
def row_1x3(draw, ylabel, handles, root, subset="all", figsize=(7.4, 2.5), legend_cols=4):
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw=dict(wspace=0.12))
    for j, (ax, (pre, title)) in enumerate(zip(axes, SETTINGS)):
        draw(ax, _load_setting(root, pre, subset))
        ax.set_title(title, pad=3, color=INK_2)
        ax.set_xlabel(r"Injected noise std $\varepsilon$")
        _panel_label(ax, "abc"[j], x=-0.2 if j == 0 else -0.08)
    axes[0].set_ylabel(ylabel)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncols=legend_cols,
               handlelength=1.6, columnspacing=1.4)
    return fig


def draw_parity_panel(ax, rows, lo=0.007, hi=0.5):
    g = np.geomspace(lo, hi, 50)
    ax.fill_between(g, g * 0.9, g * 1.1, color="#f1f0f3", linewidth=0, zorder=0)
    ax.plot(g, g, color=INK_3, linewidth=0.8, zorder=1)
    rows = [r for r in rows if r["eps"] > 0]
    for ek, lk, col, _ in STAGES:
        ax.scatter([r[ek] for r in rows], [r[lk] for r in rows], marker="o", s=26, color=col,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    _plain_log(ax.xaxis); _plain_log(ax.yaxis)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def plot_parity_combined(root, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True, gridspec_kw=dict(wspace=0.1))
    for j, (ax, (pre, title)) in enumerate(zip(axes, SETTINGS)):
        draw_parity_panel(ax, _load_setting(root, pre, "all"))
        ax.set_title(title, pad=3, color=INK_2)
        ax.set_xlabel(r"Expected aleatoric $\sigma$ (× state std)")
        _panel_label(ax, "abc"[j], x=-0.22 if j == 0 else -0.06)
    axes[0].set_ylabel(r"Learned aleatoric $\sigma$ (× state std)")
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker="o", linestyle="none", color=HEAT_C, markersize=5, markeredgecolor="white",
                label="Heating stage"),
         Line2D([], [], marker="o", linestyle="none", color=FULL_C, markersize=5, markeredgecolor="white",
                label="Full transition"),
         Line2D([], [], color=INK_3, linewidth=0.8, label="Perfect recovery"),
         Patch(facecolor="#f1f0f3", label="±10 %")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncols=4, handlelength=1.6,
               columnspacing=1.4)
    _save(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="surrogate_model_v3/results_aleatoric_calibration")
    p.add_argument("--style", choices=["dot", "ratio", "parity"], required=True)
    p.add_argument("--combined", action="store_true",
                   help="Pool InD and OoD test samples (subset 'all'): one row of three panels, one per setting.")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)
    base.ROOT[0] = a.root
    if a.combined:
        if a.style == "dot":
            _save(row_1x3(draw_dot, r"Aleatoric $\sigma$ (× state std, log scale)", dot_handles(), a.root), a.out)
        elif a.style == "ratio":
            _save(row_1x3(draw_ratio, r"Learned / expected aleatoric $\sigma$", ratio_handles(), a.root), a.out)
        else:
            plot_parity_combined(a.root, a.out)
        print(f"[paper_figure_aleatoric_alt] Saved → {a.out}.{{pdf,png}}")
        return
    if a.style == "dot":
        fig, _ = grid_2x3(draw_dot, r"Aleatoric $\sigma$ (× state std, log scale)", dot_handles())
        _save(fig, a.out)
    elif a.style == "ratio":
        fig, _ = grid_2x3(draw_ratio, r"Learned / expected aleatoric $\sigma$", ratio_handles())
        _save(fig, a.out)
    else:
        plot_parity(a.root, a.out)
    print(f"[paper_figure_aleatoric_alt] Saved → {a.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
