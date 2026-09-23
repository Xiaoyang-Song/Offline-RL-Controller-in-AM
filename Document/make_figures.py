"""
Document/make_figures.py
Publication-style versions of the surrogate-paper figures, drawn from raw
arrays dumped by `baseline_surrogate.summarize_results --dump_npz` and
`surrogate_model_v3.evaluate_ood --dump_npz` (see Document/README_figures.md
for the exact commands).

    python Document/make_figures.py --raw_dir <dir with full.npz narrow.npz ood.npz>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "xtick.color": "#333333", "ytick.color": "#333333", "legend.frameon": False,
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

# canonical method order / look (matches the RL figures: ours = the single saturated accent)
METHODS = [  # key substring, display name, colour, marker
    ("surrogate_v3",     "Ours",             "#C0392B", "o"),
    ("mlp",              "MLP",              "#2E6F9E", "s"),
    ("lstm",             "LSTM",             "#2A9D8F", "^"),
    ("vanilla_ensemble", "Vanilla ensemble", "#8E7CC3", "D"),
    ("kalman_filter",    "Kalman filter",    "#8C8C8C", "v"),
]
BAND, BAND_EDGE = "#DCE6F0", "#7C93AB"


def _find(npz, tag, key, field):
    for k in npz.files:
        parts = k.split("/")
        if parts[0] == tag and parts[-1] == field and key in "/".join(parts[1:-1]):
            return npz[k]
    raise KeyError((tag, key, field))


def _style(ours):
    return dict(lw=2.8 if ours else 1.6, ms=6.5 if ours else 4.8, zorder=4 if ours else 3)


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print("Saved →", p)


# ---------------------------------------------------------------------------
def per_layer_mae(full):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, tag, title in ((axes[0], "tf", "(a) Teacher-forced"), (axes[1], "ar", "(b) Auto-regressive")):
        for key, name, c, m in METHODS:
            mae = _find(full, tag, key, "mae")
            layers = np.arange(1, len(mae) + 1)
            ax.plot(layers, mae, color=c, marker=m, label=name, **_style(key == "surrogate_v3"))
        ax.set_yscale("log"); ax.set_xticks(np.arange(1, 13))
        ax.set_xlabel("Layer"); ax.set_title(title, fontweight="bold", loc="left")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Next-state MAE [K]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "fig_per_layer_mae_full.png")


def rmse_vs_action(narrow, id_range=(200, 300), bin_w=20.0):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.axvspan(*id_range, color=BAND, zorder=0, label=f"Training range ({id_range[0]}–{id_range[1]} W)")
    for x in id_range:
        ax.axvline(x, color=BAND_EDGE, ls="--", lw=0.9, zorder=1)
    for key, name, c, m in METHODS:
        a, sq = _find(narrow, "tf", key, "action"), _find(narrow, "tf", key, "sq_err")
        edges = np.arange(np.floor(a.min() / bin_w) * bin_w, np.ceil(a.max() / bin_w) * bin_w + bin_w, bin_w)
        idx = np.digitize(a, edges) - 1
        ctr = (edges[:-1] + edges[1:]) / 2
        r = np.array([np.sqrt(sq[idx == b].mean()) if (idx == b).sum() >= 5 else np.nan for b in range(len(ctr))])
        ax.plot(ctr, r, color=c, marker=m, label=name, **_style(key == "surrogate_v3"))
    ax.set_xlabel("Laser power [W]"); ax.set_ylabel("Next-state RMSE [K]")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    _save(fig, "fig_rmse_vs_action_narrow.png")


def ood(ood_npz, prefix, stage_word, fname, ylab_note):
    ctr = ood_npz[f"{prefix}/centers"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    specs = (("epi", "Epistemic uncertainty", "#2E6F9E", "Mean $\\sigma^{\\mathrm{epi}}$" + ylab_note),
             ("ale", "Aleatoric uncertainty", "#8E7CC3", "Mean $\\sigma^{\\mathrm{ale}}$" + ylab_note),
             ("rmse", "Single-step error", "#C0392B", "RMSE [K]"))
    ids = ood_npz["id_ranges"]
    for ax, (k, title, c, ylab), tag in zip(axes, specs, ("(a)", "(b)", "(c)")):
        for lo, hi in ids:
            ax.axvspan(lo, hi, color=BAND, zorder=0, label="Training range" if ax is axes[0] else None)
            ax.axvline(lo, color=BAND_EDGE, ls="--", lw=0.9); ax.axvline(hi, color=BAND_EDGE, ls="--", lw=0.9)
        ax.plot(ctr, ood_npz[f"{prefix}/{k}"], color=c, marker="o", ms=5, lw=2.0, zorder=3)
        ax.set_xlabel("Laser power [W]"); ax.set_ylabel(ylab)
        ax.set_title(f"{tag} {title}", fontweight="bold", loc="left")
        if k != "rmse":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    axes[0].legend(loc="upper center")
    fig.suptitle(f"{stage_word}", y=1.02, fontsize=11, style="italic")
    fig.tight_layout()
    _save(fig, fname)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--raw_dir", required=True)
    a = ap.parse_args()
    full = np.load(os.path.join(a.raw_dir, "full.npz"))
    narrow = np.load(os.path.join(a.raw_dir, "narrow.npz"))
    oodz = np.load(os.path.join(a.raw_dir, "ood.npz"))
    os.makedirs(OUT, exist_ok=True)
    per_layer_mae(full)
    rmse_vs_action(narrow)
    ood(oodz, "heat", "Heating stage", "fig_ood_heating.png", " (normalized)")
    ood(oodz, "comb", "Combined heating + cooling (post-cooling state)", "fig_ood_combined.png", " (normalized)")
