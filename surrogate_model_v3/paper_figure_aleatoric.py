"""
surrogate_model_v3/paper_figure_aleatoric.py
--------------------------------------------------------
Paper figure: does the learned ALEATORIC uncertainty recover a known amount of
data noise? One panel per data-coverage setting; each panel compares, for
models trained with injected target noise of std ε (x state_std, per node;
train.py --perturb_frac), the learned aleatoric σ with its expected value:

  heating stage   : expected = ε                  (one noise draw on its target)
  full transition : expected = ε · mean_i √(2 + ‖a_i‖²), a_i = row i of the
                    model's own cooling Jacobian A = I + J, averaged over test
                    transitions (expected_aleatoric_jacobian.py). Reduces to
                    √3 ε when A ≈ I.

All σ in normalised units (fraction of each node's state_std), node-averaged,
mean over IN-DISTRIBUTION test samples (laser power inside the model's training
range; --subset all to use every sample); learned full-transition σ is the
propagated one. Bars: 5-95 % over those samples.

--merged draws InD and OoD test samples in one 2 x 3 figure (columns = settings,
top row InD a-c, bottom row OoD d-e for narrow / gap; full range has no OoD
samples, so that slot holds the legend).

Inputs: surrogate_model_v3/results_aleatoric_calibration/<setting>_perturb<ε>/
          ood_test.npz                  (evaluate_ood.py --split test --propagate_uncertainty --dump_npz)
          expected_full_aleatoric.json  (expected_aleatoric_jacobian.py)
Settings without any finished run are skipped.

Usage
-----
    python -m surrogate_model_v3.paper_figure_aleatoric \\
        --out surrogate_model_v3/results_aleatoric_calibration/paper_fig_aleatoric
    # InD + OoD in one 2 x 3 figure:
    python -m surrogate_model_v3.paper_figure_aleatoric --merged \\
        --out surrogate_model_v3/results_aleatoric_calibration/paper_fig_aleatoric_merged
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK_2, INK_3, GRID, _panel_label

SETTINGS = [  # folder prefix, panel title
    ("full_range",          "Full range (100–400 W)"),
    ("narrow_200_300W",     "Narrow (200–300 W)"),
    ("gap_100-150_350-400", "Gap (100–150, 350–400 W)"),
]
# same heating-red / cooling-blue pair as paper_figure_propagation.py (WARM / COOL, light variants for fills)
HEAT_C, FULL_C = "#c4352d", "#3b55c4"           # heating stage (red), full transition = after cooling (blue)
HEAT_L, FULL_L = "#efa99b", "#9fb0ea"
_lighten = lambda c, f=0.62: tuple(1 - f * (1 - np.array(to_rgb(c))))   # mix toward white


def _load_setting(root, prefix, subset="id"):
    rows = []
    for d in glob.glob(os.path.join(root, f"{prefix}_perturb*")):
        m = re.search(r"_perturb([0-9.]+)$", d)
        npz, js = os.path.join(d, "ood_test.npz"), os.path.join(d, "expected_full_aleatoric.json")
        if not (m and os.path.exists(npz) and os.path.exists(js)):
            continue
        z = np.load(npz)
        exp = json.load(open(js))[0]
        idm = z["id_mask"].astype(bool)
        keep = {"id": idm, "ood": ~idm, "all": np.ones(len(idm), bool)}[subset]
        if not keep.any():                                   # e.g. full range has no OOD samples
            continue
        h, c = z["heat_samples/ale"][keep], z["comb_samples/ale"][keep]
        rows.append(dict(eps=float(m.group(1)),
                         heat=h.mean(), heat_lo=np.percentile(h, 5), heat_hi=np.percentile(h, 95),
                         full=c.mean(), full_lo=np.percentile(c, 5), full_hi=np.percentile(c, 95),
                         full_exp=exp["expected_full"], checkpoint=str(z["checkpoint"])))
    return sorted(rows, key=lambda r: r["eps"])


def _ymax(datasets):
    return max(max(r["full_hi"], r["full_exp"]) for data in datasets for _, rows in data for r in rows) * 1.08


def _draw_panel(ax, rows, ymax, bw=0.19):
    rows = [r for r in rows if r["eps"] > 0]                     # no injected noise -> nothing to recover
    x = np.arange(len(rows))
    # per noise level: [heating expected | heating learned]  [full expected | full learned]
    for off, key, col in ((-1.5, "eps", HEAT_C), (-0.5, "heat", HEAT_C), (0.5, "full_exp", FULL_C), (1.5, "full", FULL_C)):
        learned = key in ("heat", "full")
        ax.bar(x + off * bw, [r[key] for r in rows], width=bw * 0.92,
               color=col if learned else _lighten(col), edgecolor=col, linewidth=0.6 if not learned else 0,
               zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['eps']:g}" for r in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def _legend(fig, y):
    handles = [Patch(facecolor=_lighten(HEAT_C), edgecolor=HEAT_C, linewidth=0.6, label="Heating stage, expected"),
               Patch(facecolor=HEAT_C, label="Heating stage, learned"),
               Patch(facecolor=_lighten(FULL_C), edgecolor=FULL_C, linewidth=0.6, label="Full transition, expected"),
               Patch(facecolor=FULL_C, label="Full transition, learned")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncols=4, handlelength=1.2,
               columnspacing=1.4)


def _save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)


def _report(label, data):
    print(f"[paper_figure_aleatoric] learned vs expected (mean over {label} test samples, normalised units):")
    for title, rows in data:
        print(f"  {title}")
        for r in rows:
            dh = 100 * (r["heat"] / r["eps"] - 1) if r["eps"] > 0 else float("nan")
            df = 100 * (r["full"] / r["full_exp"] - 1) if r["eps"] > 0 else float("nan")
            print(f"    ε={r['eps']:<5g} heating {r['heat']:.4f} vs {r['eps']:.4f} ({dh:+.1f}%)   "
                  f"full {r['full']:.4f} vs {r['full_exp']:.4f} ({df:+.1f}%)")


def plot_row(root, subset, out):
    data = [(t, _load_setting(root, pre, subset)) for pre, t in SETTINGS]
    data = [(t, rows) for t, rows in data if rows]
    if not data:
        raise SystemExit(f"No finished runs under {root}")
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n + 0.2, 2.45), sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.12))
    axes = axes[0]
    ymax = _ymax([data])
    for j, (ax, (title, rows)) in enumerate(zip(axes, data)):
        _draw_panel(ax, rows, ymax)
        ax.set_xlabel(r"Injected noise std $\varepsilon$")
        ax.set_title(title, pad=3, color=INK_2)
        _panel_label(ax, "abc"[j], x=-0.08 if j else -0.2)
    axes[0].set_ylabel(r"Aleatoric $\sigma$ (× state std)")
    _legend(fig, 1.10)
    _save(fig, out)
    _report(subset.upper(), data)
    print(f"[paper_figure_aleatoric] Saved → {out}.{{pdf,png}}")


def _legend_handles():
    return [Patch(facecolor=_lighten(HEAT_C), edgecolor=HEAT_C, linewidth=0.6, label="Heating stage, expected"),
            Patch(facecolor=HEAT_C, label="Heating stage, learned"),
            Patch(facecolor=_lighten(FULL_C), edgecolor=FULL_C, linewidth=0.6, label="Full transition, expected"),
            Patch(facecolor=FULL_C, label="Full transition, learned")]


def legend_slot(ax, handles, note="Full range has no OoD test samples\n(training covers 100–400 W)."):
    """Turn an empty grid slot into the figure legend (+ small note)."""
    ax.axis("off")
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.58), frameon=False, handlelength=1.6,
              labelspacing=0.7, borderaxespad=0)
    if note:
        ax.text(0.5, 0.08, note, transform=ax.transAxes, ha="center", va="bottom", color=INK_3,
                fontsize=plt.rcParams["font.size"] - 1, linespacing=1.3)


def grid_2x3(draw, ylabel, handles, figsize=(7.6, 4.6), **sub_kw):
    """Shared layout for all InD/OoD variants: top row InD (a-c), bottom-left = legend, bottom (d, e) = OoD
    narrow / gap. draw(ax, rows) fills one panel."""
    fig, axes = plt.subplots(2, 3, figsize=figsize, gridspec_kw=dict(wspace=0.14, hspace=0.45), **sub_kw)
    letters = iter("abcde")
    for i, (lab, subset) in enumerate((("InD", "id"), ("OoD", "ood"))):
        for j, (pre, title) in enumerate(SETTINGS):
            ax = axes[i, j]
            rows = _load_setting(ROOT[0], pre, subset)
            if not rows:
                legend_slot(ax, handles)
                continue
            draw(ax, rows)
            ax.set_title(f"{title} · {lab}", pad=3, color=INK_2)
            if i == 1:
                ax.set_xlabel(r"Injected noise std $\varepsilon$")
            first = j == 0 or not _load_setting(ROOT[0], SETTINGS[j - 1][0], subset)
            if not first:
                ax.tick_params(labelleft=False)
            _panel_label(ax, next(letters), x=-0.15 if first else -0.08)
    fig.supylabel(ylabel, x=0.035, fontsize=plt.rcParams["axes.labelsize"])
    return fig, axes


ROOT = ["surrogate_model_v3/results_aleatoric_calibration"]


def plot_merged(root, out):
    """2 x 3: InD top row (a-c); OoD bottom row (d, e = narrow, gap); legend in the empty full-range OoD slot."""
    ROOT[0] = root
    ymax = _ymax([[(t, _load_setting(root, pre, s)) for pre, t in SETTINGS] for s in ("id", "ood")])
    fig, _ = grid_2x3(lambda ax, rows: _draw_panel(ax, rows, ymax), r"Aleatoric $\sigma$ (× state std)",
                      _legend_handles())
    _save(fig, out)
    print(f"[paper_figure_aleatoric] Saved → {out}.{{pdf,png}}")

def _draw_panel_target(ax, rows, ymax, bw=0.29, shade=False, annotate=False):
    """Learned sigma as bars (one per stage), expected as a dark target line across each bar,
    5-95 % whiskers over test samples; optionally the deviation (%) above each bar (annotate) and light
    bands per noise level (shade) — both off in the paper version."""
    rows = [r for r in rows if r["eps"] > 0]
    x = np.arange(len(rows))
    if shade:
        for i in x[::2]:
            ax.axvspan(i - 0.5, i + 0.5, color="#f6f5f8", zorder=0, linewidth=0)
    fs = plt.rcParams["font.size"] - 2.6
    for k, (ek, lk, lo_k, hi_k, col) in enumerate((("eps", "heat", "heat_lo", "heat_hi", HEAT_C),
                                                   ("full_exp", "full", "full_lo", "full_hi", FULL_C))):
        xs = x + (k - 0.5) * (bw + 0.08)
        learned = np.array([r[lk] for r in rows]); expected = np.array([r[ek] for r in rows])
        ax.bar(xs, learned, width=bw, color=HEAT_L if k == 0 else FULL_L, edgecolor=col, linewidth=0.8, zorder=2)
        ax.vlines(xs, [r[lo_k] for r in rows], [r[hi_k] for r in rows], color=INK_2,
                  linewidth=0.6, zorder=3)
        ax.hlines(expected, xs - bw * 0.62, xs + bw * 0.62, color="#1f1f1f", linewidth=1.3, zorder=4)
        for xi, l, e, hi in (zip(xs, learned, expected, [r[hi_k] for r in rows]) if annotate else ()):
            dev = 100 * (l / e - 1)
            txt = "0%" if abs(dev) < 0.5 else f"{dev:+.0f}%".replace("-", "\u2212")
            ax.text(xi, max(l, e, hi) + ymax * 0.012, txt, ha="center", va="bottom", fontsize=fs, color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['eps']:g}" for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5); ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def plot_row_target(root, subset, out):
    data = [(t, _load_setting(root, pre, subset)) for pre, t in SETTINGS]
    data = [(t, rows) for t, rows in data if rows]
    ymax = max(max(r["full_hi"], r["full_exp"], r["full"]) for _, rows in data for r in rows) * 1.06
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 0.2, 2.5), sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.1))
    for j, (ax, (title, rows)) in enumerate(zip(axes[0], data)):
        _draw_panel_target(ax, rows, ymax)
        ax.set_xlabel(r"Injected noise std $\varepsilon$")
        ax.set_title(title, pad=3, color=INK_2)
        _panel_label(ax, "abc"[j], x=-0.08 if j else -0.2)
    axes[0, 0].set_ylabel(r"Aleatoric $\sigma$ (× state std)")
    handles = [Patch(facecolor=HEAT_L, edgecolor=HEAT_C, linewidth=0.8, label="Heating stage, learned"),
               Patch(facecolor=FULL_L, edgecolor=FULL_C, linewidth=0.8, label="Full transition, learned"),
               Line2D([], [], color="#1f1f1f", linewidth=1.3, label="Expected"),
               Line2D([], [], color=INK_2, linewidth=0.6, label="5–95 % over test samples")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncols=4, handlelength=1.5,
               columnspacing=1.4)
    _save(fig, out)
    _report(subset.upper(), data)
    print(f"[paper_figure_aleatoric] Saved → {out}.{{pdf,png}}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="surrogate_model_v3/results_aleatoric_calibration")
    p.add_argument("--subset", choices=["id", "ood", "all"], default="id",
                   help="Test samples used: 'id' (default) = laser power inside the model's training "
                        "range, where the noise level is identifiable; 'ood' = only outside it; 'all' = every test sample (the "
                        "aleatoric head is extrapolated out of distribution and can be far off there, "
                        "notably at tiny noise levels).")
    p.add_argument("--merged", action="store_true",
                   help="2 x 3 figure: InD (top row, a-c) and OoD (bottom row, d-e) test samples together; "
                        "--subset is ignored.")
    p.add_argument("--target", action="store_true",
                   help="Row layout with learned-sigma bars, expected value as a target line, 5-95 %% whiskers "
                        "and the deviation in %% (instead of separate expected / learned bars).")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)
    if a.merged:
        plot_merged(a.root, a.out)
    elif a.target:
        plot_row_target(a.root, a.subset, a.out)
    else:
        plot_row(a.root, a.subset, a.out)


if __name__ == "__main__":
    main()
