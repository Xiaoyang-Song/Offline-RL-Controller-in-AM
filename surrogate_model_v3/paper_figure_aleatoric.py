"""
surrogate_model_v3/paper_figure_aleatoric.py
--------------------------------------------------------
Paper figure: does the learned ALEATORIC uncertainty recover a known amount of
data noise? Uses surrogates trained with Gaussian target noise of std
perturb_frac x state_std (per node) added to the end-of-heating and
next-state fields (--perturb_frac in train.py), one npz per noise level from

    python -m surrogate_model_v3.evaluate_ood --split test --propagate_uncertainty --dump_npz <...>

All σ are in normalised units (fraction of each node's state_std), the unit the
noise is injected in. Expected aleatoric σ:
  heating stage   : perturb_frac              (one noise draw on its target)
  full transition : sqrt(3) x perturb_frac    (heating noise, plus the cooling
                    stage's target is next-state minus the NOISY heating field:
                    two more independent draws)

Layout (double-column, 1 x 3):
  (a) learned vs injected noise, both stages, with the expected y = x and
      y = sqrt(3) x lines.
  (b) full-transition aleatoric σ vs laser power, one line per noise level, with
      the expected level dashed: flat across InD and OoD.
  (c) full-transition epistemic σ vs laser power, same runs: rises in the gap.

Usage
-----
    python -m surrogate_model_v3.paper_figure_aleatoric \\
        --npz surrogate_model_v3/results_ood_gap_100-150_350-400_perturb0.05/ood_gap_test.npz \\
              surrogate_model_v3/results_ood_gap_100-150_350-400_perturb0.1/ood_gap_test.npz \\
              surrogate_model_v3/results_ood_gap_100-150_350-400/ood_gap_test.npz \\
        --perturb 0.05 0.1 0.2 \\
        --out surrogate_model_v3/results_aleatoric_calibration/paper_fig_aleatoric
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK, INK_2, INK_3, GRID, _group_index, _shade_training, \
    _panel_label

# one hue per noise level (light -> dark = more noise); stage by marker fill / line style
LEVEL_C = ["#7fb7b0", "#2a8a80", "#0f4740"]
HEAT_STYLE = dict(marker="o", markerfacecolor="white", linestyle=(0, (3.2, 1.6)))
FULL_STYLE = dict(marker="s", linestyle="-")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", nargs="+", required=True)
    p.add_argument("--perturb", nargs="+", type=float, required=True, help="perturb_frac of each npz, same order.")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    assert len(a.npz) == len(a.perturb)
    plt.rcParams.update(PAPER_RC)
    order = np.argsort(a.perturb)
    runs = [(a.perturb[i], np.load(a.npz[i])) for i in order]
    colors = LEVEL_C if len(runs) <= 3 else plt.cm.viridis(np.linspace(0.3, 0.9, len(runs)))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.45))

    # ---------------- (a) learned vs injected --------------------------------
    ax = axes[0]
    pmax = max(pf for pf, _ in runs) * 1.15
    xs = np.linspace(0, pmax, 50)
    ax.plot(xs, xs, color=INK_3, linewidth=0.8, zorder=1)
    ax.plot(xs, np.sqrt(3) * xs, color=INK_3, linewidth=0.8, linestyle=(0, (1.5, 1.5)), zorder=1)
    ax.text(pmax * 0.97, pmax * 0.97, "$y = x$", ha="right", va="bottom", fontsize=6.5, color=INK_2,
            rotation=0)
    ax.text(pmax * 0.60, np.sqrt(3) * pmax * 0.60, r"$y = \sqrt{3}\,x$", ha="right", va="bottom",
            fontsize=6.5, color=INK_2)
    for (pf, z), col in zip(runs, colors):
        for st, style in (("heat", HEAT_STYLE), ("comb", FULL_STYLE)):
            v = z[f"{st}_samples/ale"]
            lo, hi = np.percentile(v, [5, 95])
            ax.errorbar(pf, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], color=col, capsize=1.5,
                        elinewidth=0.7, markersize=4.5, markeredgecolor=col, zorder=3,
                        marker=style["marker"], markerfacecolor=style.get("markerfacecolor", col), linestyle="none")
    ax.set_xlim(0, pmax); ax.set_ylim(0, np.sqrt(3) * pmax * 0.9)
    ax.set_xlabel(r"Injected noise std ($\times$ state std)")
    ax.set_ylabel(r"Learned aleatoric $\sigma$")
    ax.set_title("Recovered noise level", pad=3, color=INK_2)
    ax.legend(handles=[Line2D([], [], color=INK_2, linestyle="none", label="Heating stage", **{k: v for k, v in HEAT_STYLE.items() if k != "linestyle"}),
                       Line2D([], [], color=INK_2, linestyle="none", label="Full transition", marker="s")],
              loc="upper left", handlelength=1.0, borderaxespad=0.2)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    _panel_label(ax, "a")

    # ---------------- (b)/(c) σ vs laser power, full transition --------------
    id_ranges = [tuple(r) for r in runs[0][1]["id_ranges"]]
    for ax, key, title, letter in ((axes[1], "ale", "Aleatoric", "b"), (axes[2], "epi", "Epistemic", "c")):
        action = runs[0][1]["comb_samples/action"]
        ax.set_xlim(action.min() - 5, action.max() + 5)
        _shade_training(ax, id_ranges)
        ymax = 0.0
        for (pf, z), col in zip(runs, colors):
            act = z["comb_samples/action"]
            gidx, centres = _group_index(act, 25.0)
            v = z[f"comb_samples/{key}"]
            med = np.array([np.median(v[gidx == b]) for b in range(len(centres))])
            ymax = max(ymax, med.max())
            if key == "ale":
                ax.axhline(np.sqrt(3) * pf, color=col, linewidth=0.8, linestyle=(0, (1.5, 1.5)), zorder=1)
            ax.plot(centres, med, color="white", linewidth=3.0, alpha=0.8, zorder=3)
            ax.plot(centres, med, color=col, linewidth=1.5, marker="s", markersize=2.0, zorder=4,
                    label=f"{pf:g}")
        ax.set_ylim(0, ymax * (1.45 if key == "ale" else 1.3))
        ax.set_xlabel("Laser power [W]")
        ax.set_xticks(np.arange(np.ceil(action.min() / 100) * 100, action.max() + 1, 100))
        ax.set_ylabel(f"{title} $\\sigma$, full transition")
        ax.set_title(f"{title} uncertainty (median)", pad=3, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        _panel_label(ax, letter)
    leg = axes[1].legend(title="Injected noise", title_fontsize=6.5, loc="upper center",
                         bbox_to_anchor=(0.5, 0.88), ncols=3, handlelength=1.4, columnspacing=0.8)
    leg.get_title().set_color(INK_2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}")
    plt.close(fig)

    print("[paper_figure_aleatoric] learned aleatoric σ (test-set mean) vs expected:")
    for pf, z in runs:
        h, c = z["heat_samples/ale"].mean(), z["comb_samples/ale"].mean()
        print(f"  perturb {pf:g}: heating {h:.4f} (expected {pf:.4f}, {100 * (h / pf - 1):+.1f}%)   "
              f"full {c:.4f} (expected {np.sqrt(3) * pf:.4f}, {100 * (c / (np.sqrt(3) * pf) - 1):+.1f}%)   "
              f"[{z['checkpoint']}]")
    print(f"[paper_figure_aleatoric] Saved → {a.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
