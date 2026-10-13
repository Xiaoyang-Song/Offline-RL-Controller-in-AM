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
HEAT_C, FULL_C = "#c8553d", "#2a6f97"          # heating stage (terracotta), full transition (steel blue)


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="surrogate_model_v3/results_aleatoric_calibration")
    p.add_argument("--subset", choices=["id", "ood", "all"], default="id",
                   help="Test samples used: 'id' (default) = laser power inside the model's training "
                        "range, where the noise level is identifiable; 'ood' = only outside it; 'all' = every test sample (the "
                        "aleatoric head is extrapolated out of distribution and can be far off there, "
                        "notably at tiny noise levels).")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)

    data = [(t, _load_setting(a.root, pre, a.subset)) for pre, t in SETTINGS]
    data = [(t, rows) for t, rows in data if rows]
    if not data:
        raise SystemExit(f"No finished runs under {a.root}")

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n + 0.2, 2.45), sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.12))
    axes = axes[0]
    emax = max(r["eps"] for _, rows in data for r in rows) * 1.12
    ymax = max(max(r["full_hi"], r["full_exp"]) for _, rows in data for r in rows) * 1.08

    from matplotlib.colors import to_rgb
    lighten = lambda c, f=0.62: tuple(1 - f * (1 - np.array(to_rgb(c))))   # mix toward white
    bw = 0.19                                                              # bar width (category units)
    for j, (ax, (title, rows)) in enumerate(zip(axes, data)):
        rows = [r for r in rows if r["eps"] > 0]                 # no injected noise -> nothing to recover
        x = np.arange(len(rows))
        # per noise level: [heating expected | heating learned]  [full expected | full learned]
        for off, key, col, hatch in ((-1.5, "eps", HEAT_C, None), (-0.5, "heat", HEAT_C, None),
                                     (0.5, "full_exp", FULL_C, None), (1.5, "full", FULL_C, None)):
            learned = key in ("heat", "full")
            ax.bar(x + off * bw, [r[key] for r in rows], width=bw * 0.92,
                   color=col if learned else lighten(col), edgecolor=col, linewidth=0.6 if not learned else 0,
                   zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['eps']:g}" for r in rows])
        ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_ylim(0, ymax)
        ax.set_xlabel(r"Injected noise std $\varepsilon$")
        ax.set_title(title, pad=3, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        _panel_label(ax, "abc"[j], x=-0.08 if j else -0.2)
    axes[0].set_ylabel(r"Aleatoric $\sigma$ (× state std)")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=lighten(HEAT_C), edgecolor=HEAT_C, linewidth=0.6, label="Heating stage, expected"),
               Patch(facecolor=HEAT_C, label="Heating stage, learned"),
               Patch(facecolor=lighten(FULL_C), edgecolor=FULL_C, linewidth=0.6, label="Full transition, expected"),
               Patch(facecolor=FULL_C, label="Full transition, learned")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncols=4, handlelength=1.2,
               columnspacing=1.4)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}")
    plt.close(fig)

    print(f"[paper_figure_aleatoric] learned vs expected (mean over {a.subset.upper()} test samples, normalised units):")
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
