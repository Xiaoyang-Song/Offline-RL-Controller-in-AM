"""
Document/make_fig_per_layer_rmse.py
Per-layer next-state RMSE of the proposed surrogate vs. the baselines, in the
same paper style as the surrogate_model_v3 figures (paper_figure_ood.PAPER_RC):

  one row, teacher-forced first: (a) full range | (b) narrow 200-300 W |
  (c) gap 100-150 & 350-400 W (only if <raw_dir>/gap.npz exists) — then the
  last panel: full range, autoregressive.

Inputs are the `baseline_surrogate.summarize_results --dump_npz` files
(full.npz, narrow.npz[, gap.npz]). Lines = pooled RMSE over all test
trajectories and nodes (the RMSE the tables report); whiskers = 10-90th
percentile over test trajectories of the per-trajectory RMSE over nodes (the
pooled value always lies inside; 25-75 does not, the error is heavy-tailed).
Methods are offset slightly along x so the whiskers do not overlap.

    python Document/make_fig_per_layer_rmse.py --raw_dir baseline_surrogate/results_v3_raw_v2
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK_2, INK_3, GRID, _panel_label

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

METHODS = [  # npz key, display name, colour, marker  (ours = the one saturated steel blue)
    ("surrogate_v3 (proposed)", "Ours",             "#2a6f97", "o"),
    ("lstm",                    "LSTM",             "#6d597a", "^"),
    ("vanilla_ensemble",        "Vanilla ensemble", "#e9b44c", "D"),
    ("mlp",                     "MLP",              "#e76f51", "s"),
    ("kalman_filter",           "Kalman filter",    "#9a99a0", "v"),
]
EVAL_BATCH = 32          # summarize_results' DataLoader batch size (sq_err is stored batch by batch)


def per_traj_rmse(sq_flat, n_traj, T=12, batch=EVAL_BATCH):
    """sq_err is appended per batch, layer-major inside the batch: [b0 t0 (B)], [b0 t1 (B)], ...
    -> (n_traj, T) per-trajectory RMSE over nodes."""
    out = np.empty((n_traj, T)); i = 0
    for b0 in range(0, n_traj, batch):
        B = min(batch, n_traj - b0)
        for t in range(T):
            out[b0:b0 + B, t] = sq_flat[i:i + B]; i += B
    assert i == len(sq_flat)
    return np.sqrt(out)


def draw(ax, z, tag, band=(10, 90), dodge=0.11):
    for m, (key, name, col, mk) in enumerate(METHODS):
        rmse = z[f"{tag}/{key}/rmse"]
        L = np.arange(1, len(rmse) + 1)
        sq = z[f"{tag}/{key}/sq_err"]
        tr = per_traj_rmse(sq, len(sq) // len(rmse), len(rmse))
        assert np.allclose(np.sqrt((tr ** 2).mean(0)), rmse, rtol=1e-4), (tag, key)   # layout check
        ours = key.startswith("surrogate_v3")
        lo, hi = np.percentile(tr, band, axis=0)
        x = L + (m - (len(METHODS) - 1) / 2) * dodge                  # small sideways offset per method
        ax.vlines(x, lo, hi, color=col, alpha=0.75 if ours else 0.55, linewidth=1.0 if ours else 0.8, zorder=2)
        ax.plot(x, rmse, color=col, marker=mk, markersize=3.6 if ours else 3.0, linewidth=1.9 if ours else 1.2,
                markeredgecolor="white", markeredgewidth=0.4, zorder=4 if ours else 3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1, 3, 6, 9, 12]); ax.set_xlim(0.6, 12.4)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="baseline_surrogate/results_v3_raw_v2")
    p.add_argument("--out", default=os.path.join(OUT, "fig_per_layer_rmse"))
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)

    full = np.load(os.path.join(a.raw_dir, "full.npz"))
    narrow = np.load(os.path.join(a.raw_dir, "narrow.npz"))
    gap_path = os.path.join(a.raw_dir, "gap.npz")
    panels = [(full, "tf", "Full range (100–400 W) · teacher-forced"),
              (narrow, "tf", "Narrow (200–300 W) · teacher-forced")]
    if os.path.exists(gap_path):
        panels.append((np.load(gap_path), "tf", "Gap (100–150 & 350–400 W) · teacher-forced"))
    panels.append((full, "ar", "Full range (100–400 W) · autoregressive"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.25 * n + 0.3, 2.6), sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.08))
    for k, (ax, (z, tag, title)) in enumerate(zip(axes[0], panels)):
        draw(ax, z, tag)
        ax.set_title(title.replace(" · ", "\n"), pad=3, color=INK_2, linespacing=1.25)
        ax.set_xlabel("Layer")
        if k == 0:
            ax.set_ylabel("Next-state RMSE [K]")
        _panel_label(ax, "abcd"[k], x=-0.15 if k == 0 else -0.04)

    handles = [Line2D([], [], color=c, marker=m, markersize=3.6 if i == 0 else 3.0, linewidth=1.9 if i == 0 else 1.2,
                      markeredgecolor="white", markeredgewidth=0.4, label=n)
               for i, (_, n, c, m) in enumerate(METHODS)]
    handles.append(Line2D([], [], color=INK_3, linewidth=0.9, marker="|", markersize=0,
                          label="10–90th pct. over trajectories"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.14),
               ncols=6, handlelength=1.8, columnspacing=1.1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}")
    plt.close(fig)
    print(f"Saved → {a.out}.{{pdf,png}}")
    for z, tag, title in panels:
        print(f"  {title}")
        for key, name, *_ in METHODS:
            r = z[f"{tag}/{key}/rmse"]
            print(f"    {name:17s} layer 1/6/12: {r[0]:6.2f} {r[5]:6.2f} {r[-1]:6.2f}   mean {r.mean():6.2f} K")


if __name__ == "__main__":
    main()
