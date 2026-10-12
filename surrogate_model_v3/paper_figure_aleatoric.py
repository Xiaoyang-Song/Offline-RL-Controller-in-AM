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
test-split mean; learned full-transition σ is the propagated one.

Inputs: surrogate_model_v3/results_aleatoric_calibration/<setting>_perturb<ε>/
          ood_test.npz                  (evaluate_ood.py --split test --propagate_uncertainty --dump_npz)
          expected_full_aleatoric.json  (expected_aleatoric_jacobian.py)
Settings without any finished run are skipped.

Usage
-----
    python -m surrogate_model_v3.paper_figure_aleatoric \\
        --out surrogate_model_v3/results_aleatoric_calibration/paper_fig_aleatoric
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK, INK_2, INK_3, GRID, _panel_label

SETTINGS = [  # folder prefix, panel title
    ("full_range",          "Full range (100–400 W)"),
    ("narrow_200_300W",     "Narrow (200–300 W)"),
    ("gap_100-150_350-400", "Gap (100–150, 350–400 W)"),
]
HEAT_C, FULL_C = "#e76f51", "#264653"          # heating stage, full transition (match OOD figure palette)


def _load_setting(root, prefix):
    rows = []
    for d in glob.glob(os.path.join(root, f"{prefix}_perturb*")):
        m = re.search(r"_perturb([0-9.]+)$", d)
        npz, js = os.path.join(d, "ood_test.npz"), os.path.join(d, "expected_full_aleatoric.json")
        if not (m and os.path.exists(npz) and os.path.exists(js)):
            continue
        z = np.load(npz)
        exp = json.load(open(js))[0]
        h, c = z["heat_samples/ale"], z["comb_samples/ale"]
        rows.append(dict(eps=float(m.group(1)),
                         heat=h.mean(), heat_lo=np.percentile(h, 5), heat_hi=np.percentile(h, 95),
                         full=c.mean(), full_lo=np.percentile(c, 5), full_hi=np.percentile(c, 95),
                         full_exp=exp["expected_full"], checkpoint=str(z["checkpoint"])))
    return sorted(rows, key=lambda r: r["eps"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="surrogate_model_v3/results_aleatoric_calibration")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)

    data = [(t, _load_setting(a.root, pre)) for pre, t in SETTINGS]
    data = [(t, rows) for t, rows in data if rows]
    if not data:
        raise SystemExit(f"No finished runs under {a.root}")

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n + 0.2, 2.45), sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.12))
    axes = axes[0]
    emax = max(r["eps"] for _, rows in data for r in rows) * 1.12
    ymax = max(max(r["full_hi"], r["full_exp"]) for _, rows in data for r in rows) * 1.08

    for j, (ax, (title, rows)) in enumerate(zip(axes, data)):
        eps = np.array([r["eps"] for r in rows])
        # expected: heating y = ε (line), full transition from the Jacobian (dashed through points)
        ax.plot([0, emax], [0, emax], color=HEAT_C, linewidth=0.8, alpha=0.6, zorder=1)
        fe = np.array([r["full_exp"] for r in rows])
        ax.plot(np.r_[0, eps], np.r_[0, fe], color=FULL_C, linewidth=0.8, alpha=0.6,
                linestyle=(0, (3, 2)), zorder=1)
        ax.scatter(eps, fe, marker="_", s=110, linewidths=1.4, color=FULL_C, zorder=2)
        # learned
        for key, col, mk, face in (("heat", HEAT_C, "o", "white"), ("full", FULL_C, "s", FULL_C)):
            y = np.array([r[key] for r in rows])
            yerr = np.array([[r[key] - r[f"{key}_lo"] for r in rows], [r[f"{key}_hi"] - r[key] for r in rows]])
            ax.errorbar(eps, y, yerr=yerr, fmt=mk, color=col, markerfacecolor=face, markeredgecolor=col,
                        markeredgewidth=1.0, markersize=4.2, capsize=1.5, elinewidth=0.7, zorder=3)
        ax.set_xlim(-0.005, emax); ax.set_ylim(-0.005, ymax)
        ax.set_xticks([t for t in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3) if t <= emax])
        ax.set_xlabel(r"Injected noise std $\varepsilon$")
        ax.set_title(title, pad=3, color=INK_2)
        ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        _panel_label(ax, "abc"[j], x=-0.08 if j else -0.2)
    axes[0].set_ylabel(r"Aleatoric $\sigma$ (× state std)")

    handles = [
        Line2D([], [], color=HEAT_C, marker="o", markerfacecolor="white", linestyle="none", markersize=4.2,
               label="Heating stage, learned"),
        Line2D([], [], color=HEAT_C, linewidth=0.8, alpha=0.6, label=r"Heating stage, expected ($\varepsilon$)"),
        Line2D([], [], color=FULL_C, marker="s", linestyle="none", markersize=4.2, label="Full transition, learned"),
        Line2D([], [], color=FULL_C, marker="_", markersize=9, markeredgewidth=1.4, linewidth=0.8, alpha=0.8,
               linestyle=(0, (3, 2)), label="Full transition, expected (Jacobian)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncols=2, handlelength=2.0,
               columnspacing=1.2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}")
    plt.close(fig)

    print("[paper_figure_aleatoric] learned vs expected (test-set mean, normalised units):")
    for title, rows in data:
        print(f"  {title}")
        for r in rows:
            dh = 100 * (r["heat"] / r["eps"] - 1) if r["eps"] > 0 else float("nan")
            df = 100 * (r["full"] / r["full_exp"] - 1) if r["eps"] > 0 else float("nan")
            print(f"    ε={r['eps']:<5g} heating {r['heat']:.4f} vs {r['eps']:.4f} ({dh:+.1f}%)   "
                  f"full {r['full']:.4f} vs {r['full_exp']:.4f} ({df:+.1f}%)")
    print(f"[paper_figure_aleatoric] Saved → {a.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
