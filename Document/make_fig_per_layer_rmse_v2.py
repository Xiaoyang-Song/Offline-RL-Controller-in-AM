"""
Document/make_fig_per_layer_rmse_v2.py
Restyled copy of Document/make_fig_per_layer_rmse.py (same data, same drawing via its draw()):
a figure title on top, setting names as bold capitals in a sans-serif face (FULL RANGE / NARROW / GAP,
no power ranges), the evaluation mode (teacher-forced / autoregressive) in italics, larger fonts overall.

    python Document/make_fig_per_layer_rmse_v2.py   -> Document/figures/fig_per_layer_rmse_v2.{pdf,png}
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK_2, INK_3, _panel_label
from Document.make_fig_per_layer_rmse import METHODS, OUT, draw

TITLE = "Comparison of per-layer prediction error"
SANS = dict(fontweight="bold")                    # same serif face as the rest of the figure
SETTING_COLOR = "#1f2329"
MODE_COLOR = {"tf": "#4a5563", "ar": "#9b2a22"}          # autoregressive panel picked out in red


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="baseline_surrogate/results_v3_raw_v2")
    p.add_argument("--out", default=os.path.join(OUT, "fig_per_layer_rmse_v2"))
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5,
                         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]})

    full = np.load(os.path.join(a.raw_dir, "full.npz"))
    narrow = np.load(os.path.join(a.raw_dir, "narrow.npz"))
    gap_path = os.path.join(a.raw_dir, "gap.npz")
    panels = [(full, "tf", "FULL RANGE"), (narrow, "tf", "NARROW")]
    if os.path.exists(gap_path):
        panels.append((np.load(gap_path), "tf", "GAP"))
    panels.append((full, "ar", "FULL RANGE"))
    mode = {"tf": "teacher-forced", "ar": "autoregressive"}

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.25 * n + 0.3, 2.9), sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.08))
    for k, (ax, (z, tag, setting)) in enumerate(zip(axes[0], panels)):
        draw(ax, z, tag)
        ax.text(0.5, 1.105, setting, transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
                color=SETTING_COLOR, **SANS)
        ax.text(0.5, 1.02, mode[tag], transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                fontstyle="italic", color=MODE_COLOR[tag], fontweight="bold" if tag == "ar" else "normal")
        ax.set_xlabel("Layer")
        if k == 0:
            ax.set_ylabel("Next-state RMSE [K]")
        _panel_label(ax, "abcd"[k], x=-0.17 if k == 0 else -0.05)
        ax.texts[-1].set_y(1.105); ax.texts[-1].set_fontsize(10)   # panel letter level with the setting name

    handles = [Line2D([], [], color=c, marker=m, markersize=4.2 if i == 0 else 3.6, linewidth=2.2 if i == 0 else 1.4,
                      markeredgecolor="white", markeredgewidth=0.4, label=nm)
               for i, (_, nm, c, m) in enumerate(METHODS)]
    handles.append(Line2D([], [], color=INK_3, linewidth=0.9, label="10–90th pct. over trajectories"))
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.05), ncols=6,
                     handlelength=1.8, columnspacing=1.2, frameon=False)
    leg.get_texts()[0].set_fontweight("bold")
    fig.suptitle(TITLE, y=1.21, fontsize=13, fontweight="bold", color="#1f2329")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved → {a.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
