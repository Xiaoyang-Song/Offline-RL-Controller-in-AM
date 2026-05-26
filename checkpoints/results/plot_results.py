"""
checkpoints/results/plot_results.py
------------------------------------
Academic-style per-layer reward comparison:
  - Offline RL        (evaluated on real MATLAB simulator)
  - Online RL (Real)  (evaluated on real MATLAB simulator)
  - Online RL (Sim)   (evaluated on surrogate environment)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── reproducible font / style ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.5,
    "grid.alpha":         0.35,
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
})

# ── data ───────────────────────────────────────────────────────────────────
layers = np.arange(1, 13)   # 1 … 12

offline_rl = np.array([
    -0.015978, -0.013530, -0.023281,  0.000000,  0.000000, -0.012902,
    -0.040061, -0.065268, -0.090749, -0.266384, -0.383086, -0.458948,
])

online_real = np.array([
    -0.002435,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    -0.013406, -0.038874, -0.072376, -0.247180, -0.365974, -0.447297,
])

online_sim = np.array([
    -0.002430,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
    -0.013100, -0.039710, -0.061850, -0.117840, -0.136300, -0.222020,
])

# ── colour palette (colour-blind friendly) ─────────────────────────────────
C_OFFLINE = "#E05C4B"   # red-orange
C_REAL    = "#3A7DC9"   # steel blue
C_SIM     = "#4BAE8A"   # teal green

# ── figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.5))

# shaded region: perfect zone (reward == 0)
ax.axhspan(-0.005, 0.005, color="gold", alpha=0.18, zorder=0,
           label="_nolegend_")
ax.axhline(0, color="goldenrod", linewidth=0.8, linestyle="--",
           alpha=0.7, zorder=1)

# ── line plots ─────────────────────────────────────────────────────────────
ax.plot(layers, offline_rl,
        color=C_OFFLINE, marker="o", linestyle="-",
        label="Offline RL (Real Env.)", zorder=3)

ax.plot(layers, online_real,
        color=C_REAL, marker="s", linestyle="-",
        label="Online RL (Real Env.)", zorder=3)

ax.plot(layers, online_sim,
        color=C_SIM, marker="^", linestyle="-",
        label="Online RL (Sim. Env.)", zorder=3)

# ── annotations: final-layer values ────────────────────────────────────────
for val, color, va in [
    (offline_rl[-1],  C_OFFLINE, "top"),
    (online_real[-1], C_REAL,    "center"),
    (online_sim[-1],  C_SIM,     "bottom"),
]:
    ax.annotate(
        f"{val:.3f}",
        xy=(12, val),
        xytext=(12.2, val),
        fontsize=9,
        color=color,
        va=va,
        ha="left",
    )

# ── axes ───────────────────────────────────────────────────────────────────
ax.set_xlabel("Layer Index", labelpad=6)
ax.set_ylabel("Per-Layer Reward", labelpad=6)
ax.set_title("Layer-wise Reward: Offline RL vs. Online RL", pad=10)

ax.set_xlim(0.5, 13.2)
ax.set_xticks(layers)
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

ax.grid(axis="y", which="major", linestyle=":")

# custom legend — add golden-zone entry
legend_entries = [
    Line2D([0], [0], color=C_OFFLINE, marker="o", linewidth=1.8,
           markersize=6, label="Offline RL (Real Env.)"),
    Line2D([0], [0], color=C_REAL,    marker="s", linewidth=1.8,
           markersize=6, label="Online RL (Real Env.)"),
    Line2D([0], [0], color=C_SIM,     marker="^", linewidth=1.8,
           markersize=6, label="Online RL (Sim. Env.)"),
    matplotlib.patches.Patch(facecolor="gold", alpha=0.4,
                             label="Target zone ($r = 0$)"),
]
ax.legend(handles=legend_entries, loc="lower left",
          frameon=True, edgecolor="0.8")

fig.tight_layout()

# ── save ───────────────────────────────────────────────────────────────────
out = "checkpoints/results/reward_comparison_methods.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"Saved → {out}  (+ .png)")
plt.show()
