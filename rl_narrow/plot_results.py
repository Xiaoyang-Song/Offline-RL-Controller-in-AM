"""
rl_narrow/plot_results.py
-------------------------
Paper figure of the real-physics comparison (narrow setting), from rl_narrow/results_real/*.json:
  (a) average per-layer reward of each method (bars: mean over layers, episodes and seeds;
      error bars: std across seeds, or across episodes for single-run methods);
  (b) per-layer reward (mean over episodes and seeds; band: std across seeds for multi-seed methods).
Reward = method.tex reward with w = 1, i.e. the ROI mean exp(-d / W_T) of the real end-of-heating field,
so every method is scored on the same physical quality measure. Constant power: the best value in
hindsight on these episodes.

    python -m rl_narrow.plot_results --results_dir rl_narrow/results_real \
        --out Document/figures/fig_rl_comparison
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK_2, GRID, _panel_label

# display order and style (family name as written by rl_narrow/jobs/eval_real.sh)
STYLE = [
    ("PG shaped (ours)",  "Online PG (ours)",  "#c4352d", "o", 2.0),
    ("Online Q shaped",   "Online Q-learning", "#e09f3e", "s", 1.3),
    ("Offline FQI",       "Offline Q-learning", "#3b55c4", "D", 1.3),
    ("Offline CQL",       "Offline CQL",       "#6d8fd6", "^", 1.3),
    ("Offline IQL",       "Offline IQL",       "#9fb4e6", "v", 1.3),
    ("Kalman filter",     "Kalman filter",     "#2a9d8f", "P", 1.3),
    ("Proportional",      "Proportional",      "#76ad8f", "X", 1.3),
    ("Constant",          None,                "#8a8f98", "", 1.3),     # label set from best power
]


def load(results_dir):
    fams, const = {}, []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        r = json.load(open(f))
        if not r.get("rows"):
            continue
        L = max(x["layer"] for x in r["rows"])
        cts = sorted({x["cool_time"] for x in r["rows"]})
        M = np.array([[np.mean([x["score_heat"] for x in r["rows"] if x["cool_time"] == ct and x["layer"] == l])
                       for l in range(1, L + 1)] for ct in cts])                     # (episodes, layers)
        if r["type"] == "const":
            const.append((float(r["path"]), M))
        else:
            fams.setdefault(re.sub(r"\s*seed\s*\d+\s*$", "", r["method"]).strip(), []).append(M)
    if const:
        p_best, M_best = max(const, key=lambda c: c[1].mean())
        fams[f"Constant"] = [M_best]
        fams["_const_power"] = p_best
    return fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="rl_narrow/results_real")
    ap.add_argument("--out", default="Document/figures/fig_rl_comparison")
    args = ap.parse_args()
    fams = load(args.results_dir)
    p_best = fams.pop("_const_power", None)

    plt.rcParams.update(PAPER_RC)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.42))
    rows = []
    for key, label, col, mk, lw in STYLE:
        if key not in fams:
            continue
        Ms = fams[key]
        label = label or f"Constant ({p_best:.0f} W, best)"
        per_seed = np.array([M.mean() for M in Ms])
        mean = per_seed.mean()
        err = per_seed.std() if len(Ms) > 1 else Ms[0].mean(1).std()
        curve = np.stack([M.mean(0) for M in Ms])                                    # (seeds, layers)
        rows.append((label, col, mk, lw, mean, err, curve))

    # (a) average reward
    y = np.arange(len(rows))[::-1]
    for yi, (label, col, _mk, _lw, mean, err, _c) in zip(y, rows):
        axa.barh(yi, mean, xerr=err, color=col, height=0.66, error_kw=dict(ecolor="#1f2329", lw=0.8, capsize=2.2))
        axa.text(mean + err + 0.012, yi, f"{mean:.3f}", va="center", ha="left", fontsize=6.8, color="#1f2329")
    axa.set_yticks(y)
    axa.set_yticklabels([r[0] for r in rows])
    for t, r in zip(axa.get_yticklabels(), rows):
        if "ours" in r[0]:
            t.set_fontweight("bold")
    axa.set_xlim(0, 1.0)
    axa.set_xlabel("Average per-layer reward")
    axa.grid(axis="x", color=GRID, lw=0.6); axa.set_axisbelow(True)
    axa.tick_params(axis="y", length=0)
    _panel_label(axa, "a", x=-0.62)

    # (b) per-layer reward
    for label, col, mk, lw, _m, _e, curve in rows[::-1]:
        L = np.arange(1, curve.shape[1] + 1)
        mu = curve.mean(0)
        if curve.shape[0] > 1:
            axb.fill_between(L, mu - curve.std(0), mu + curve.std(0), color=col, alpha=0.15, lw=0)
        axb.plot(L, mu, color=col, marker=mk or None, ms=3.2, lw=lw, label=label,
                 ls="--" if label.startswith("Constant") else "-", zorder=5 if "ours" in label else 3)
    axb.set_xlabel("Layer")
    axb.set_ylabel("Reward")
    axb.set_xticks(range(1, 13))
    axb.set_ylim(0, 1.0)
    axb.grid(color=GRID, lw=0.6); axb.set_axisbelow(True)
    h, l = axb.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], fontsize=6.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               handlelength=2.2, columnspacing=1.2, frameon=False)
    _panel_label(axb, "b", x=-0.13)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    with open(f"{args.out}_data.json", "w") as f:                   # everything drawn, for restyling later
        json.dump(dict(reward="EoH ROI mean exp(-d/W_T) (w = 1), real physics",
                       methods=[dict(label=r[0], color=r[1], marker=r[2], mean=float(r[4]), err=float(r[5]),
                                     per_layer_seed_curves=r[6].tolist()) for r in rows]), f, indent=1)
    print("saved", args.out, "(+ _data.json) |", ", ".join(f"{r[0]} {r[4]:.3f}±{r[5]:.3f}" for r in rows))


if __name__ == "__main__":
    main()
