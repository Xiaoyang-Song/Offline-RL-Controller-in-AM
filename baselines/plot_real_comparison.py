"""
baselines/plot_real_comparison.py
Real-physics comparison figures (leaderboard + overlaid per-layer reward
curves) from per-layer rewards copied from the evaluate_real_all_methods logs.
Offline Q and Proportional are deterministic, so their 3 logged episodes are
identical and one is used; Kalman / Particle / UCPG are 1 episode each.

    python -m baselines.plot_real_comparison
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "baselines/results_narrow_200_300W_real_rest"
T_L, T_H = 2000.0, 2800.0
SPAN = T_H - T_L

# per-layer rewards (12 layers), real MATLAB physics
REWARDS = {
    "UCPG (ours)":      [-0.9360, -0.3052, -0.1884, -0.3193, -0.2468, -0.1133, -0.0945, -0.1022, -0.1221, -0.1769, -0.2177, -0.2478],
    "Kalman filter":    [-1.4514, -0.4775, -0.3066, -0.3566, -0.2817, -0.3962, -0.3739, -0.2615, -0.3444, -0.4060, -0.3002, -0.4546],
    "Particle filter":  [-1.4547, -1.3974, -1.0465, -0.2272, -0.1944, -0.4102, -0.4006, -0.2803, -0.2883, -0.3831, -0.2796, -0.2911],
    "Offline Q-learning": [-0.4844, -0.2592, -0.2102, -0.2258, -0.2495, -0.3637, -0.4980, -0.6088, -0.6132, -0.8520, -0.9943, -1.0983],
    "Proportional":     [-0.6910, -1.1670, -1.6245, -1.9866, -2.2651, -2.9924, -3.3724, -3.6384, -3.5745, -4.0835, -4.3844, -4.5978],
}
# muted, colour-blind-safe palette; ours is the single saturated accent
COLORS = {
    "UCPG (ours)":        "#C0392B",
    "Kalman filter":      "#2E6F9E",
    "Particle filter":    "#6FA8C7",
    "Offline Q-learning": "#5B5B5B",
    "Proportional":       "#A6A6A6",
}
MARKERS = {"UCPG (ours)": "o", "Kalman filter": "s", "Particle filter": "^",
           "Offline Q-learning": "D", "Proportional": "v"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "legend.frameon": False, "figure.dpi": 150,
})


def _bars(ax, stats, order, key, xlabel, title, fmt, show_names=True):
    y = np.arange(len(order))
    vals = [stats[k][key] for k in order]
    ax.barh(y, vals, color=[COLORS[k] for k in order], height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v, yi, (" " + fmt.format(v)) if v >= 0 else (fmt.format(v) + " "),
                va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color="#222222")
    ax.set_xlabel(xlabel); ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.set_yticks(y); ax.set_yticklabels(order if show_names else [])
    ax.invert_yaxis()
    for lbl in ax.get_yticklabels():
        if "ours" in lbl.get_text(): lbl.set_fontweight("bold")
    if key == "ret": ax.set_xlim(min(vals) * 1.18, 0)
    else:            ax.set_xlim(0, max(vals) * 1.18)


def _curves(ax, stats, order, n, title):
    layers = np.arange(1, n + 1)
    for k in order[::-1]:            # draw best last so it sits on top
        ours = "ours" in k
        ax.plot(layers, REWARDS[k], color=COLORS[k], marker=MARKERS[k], ms=6 if ours else 4.5,
                lw=2.6 if ours else 1.5, label=k, zorder=3 if ours else 2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Per-layer reward"); ax.set_title(title, fontweight="bold")
    ax.set_xticks(layers); ax.grid(True, alpha=0.25, lw=0.6)
    sec = ax.secondary_yaxis("right", functions=(lambda r: -r * SPAN, lambda d: -d / SPAN))
    sec.set_ylabel("Temperature deviation [K]"); sec.spines["right"].set_visible(True)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="lower left", fontsize=9.5, ncol=2)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(next(iter(REWARDS.values())))
    stats = {k: dict(ret=float(np.sum(v)), dev=float(-np.mean(v) * SPAN)) for k, v in REWARDS.items()}
    order = sorted(stats, key=lambda k: stats[k]["ret"], reverse=True)   # best first

    with open(os.path.join(OUT_DIR, "comparison_real.csv"), "w") as f:
        f.write("name,return,deviation_K_mean\n")
        for k in order:
            f.write(f"{k},{stats[k]['ret']:.4f},{stats[k]['dev']:.2f}\n")

    RET = ("ret", "Episode return (higher is better)", "Return", "{:.2f}")
    DEV = ("dev", "Mean temperature deviation [K] (lower is better)", "Physical deviation", "{:.0f}")
    CURVE_TITLE = "Per-layer reward under real physics"

    def save(fig, name):
        p = os.path.join(OUT_DIR, name)
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print("Saved →", p)

    # ---- leaderboard -------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    _bars(axL, stats, order, *RET); _bars(axR, stats, order, *DEV, show_names=False)
    fig.tight_layout(); save(fig, "comparison_real_leaderboard.png")

    # ---- per-layer reward curves -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _curves(ax, stats, order, n, CURVE_TITLE)
    fig.tight_layout(); save(fig, "comparison_real_curves.png")

    # ---- combined: leaderboard (left, stacked) | curves (right) -------------
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], wspace=0.42, hspace=0.75)
    axR_ = fig.add_subplot(gs[0, 0]); axD_ = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[:, 1])
    _bars(axR_, stats, order, *RET); _bars(axD_, stats, order, *DEV)
    _curves(axC, stats, order, n, CURVE_TITLE)
    for ax, tag in ((axR_, "(a)"), (axD_, "(b)"), (axC, "(c)")):
        ax.text(-0.02, 1.08, tag, transform=ax.transAxes, fontweight="bold", fontsize=12, ha="right")
    save(fig, "comparison_real_combined.png")


if __name__ == "__main__":
    main()
