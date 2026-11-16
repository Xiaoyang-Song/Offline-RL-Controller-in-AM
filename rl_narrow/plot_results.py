"""
rl_narrow/plot_results.py
-------------------------
Paper figure of the real-physics comparison (narrow setting), from rl_narrow/results_real/*.json:
  (a) average per-layer reward of each method (bars: mean over layers, episodes and seeds;
      error bars: std across seeds, or across episodes for single-run methods);
  (b) per-layer reward (mean over episodes and seeds; band: std across seeds for multi-seed methods).
Reward = method.tex reward with w = 1, i.e. the ROI mean exp(-d / W_T) of the real end-of-heating field,
so every method is scored on the same physical quality measure. Constant power: the best value in
hindsight on these episodes. Methods are grouped into digital twin-enabled online RL and offline methods
(no digital twin); the Kalman filter is not shown.

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

# display order and style (family name as written by rl_narrow/jobs/eval_real.sh); group: "online" uses the
# digital twin, "offline" never does (Kalman filter deliberately not shown)
ONLINE_TITLE = "Digital twin-enabled online RL"
OFFLINE_TITLE = "Offline methods (baselines)"
STYLE = [
    ("PG shaped (ours)",  "Online PG",          "#c4352d", "o", 1.9, "online"),
    ("Online Q shaped",   "Online Q-learning",  "#e8873a", "s", 1.9, "online"),
    ("Offline FQI",       "Offline Q-learning", "#2f4b9a", "D", 1.1, "offline"),
    ("Offline CQL",       "Offline CQL",        "#5f7fcf", "^", 1.1, "offline"),
    ("Offline IQL",       "Offline IQL",        "#9db1e3", "v", 1.1, "offline"),
    ("Proportional",      "Proportional",       "#4f5b69", "X", 1.1, "offline"),
    ("Constant",          None,                 "#9aa1ab", "",  1.1, "offline"),   # label set from best power
]
BAND = {"online": "#fbe9e7", "offline": "#eef2f8"}


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
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1.2], wspace=0.42))
    rows = []
    for key, label, col, mk, lw, grp in STYLE:
        if key not in fams:
            continue
        Ms = fams[key]
        label = label or f"Constant ({p_best:.0f} W, best)"
        per_seed = np.array([M.mean() for M in Ms])
        mean = per_seed.mean()
        err = per_seed.std() if len(Ms) > 1 else Ms[0].mean(1).std()
        curve = np.stack([M.mean(0) for M in Ms])                                    # (seeds, layers)
        rows.append((label, col, mk, lw, mean, err, curve, grp))

    # sort within each group by average reward (best first)
    rows = sorted(rows, key=lambda r: (r[7] != "online", -r[4]))

    # (a) average reward (bars from zero)
    ypos, y, headers = {}, 0.0, []
    for grp, title in (("online", ONLINE_TITLE), ("offline", OFFLINE_TITLE)):
        members = [r for r in rows if r[7] == grp]
        if not members:
            continue
        top = y
        headers.append((y, title, grp)); y -= 0.9
        for r in members:
            ypos[r[0]] = y; y -= 1.0
        axa.axhspan(y + 0.45, top + 0.5, color=BAND[grp], lw=0, zorder=0)
        y -= 0.35
    x_lo, x_hi = 0.0, 1.05
    for label, col, mk, _lw, mean, err, _c, grp in rows:
        yi = ypos[label]
        axa.barh(yi, mean, xerr=err, color=col, height=0.66, zorder=2,
                 error_kw=dict(ecolor="#1f2329", lw=0.8, capsize=2.2))
        axa.text(mean + err + 0.012, yi, f"{mean:.3f}", va="center", ha="left", fontsize=6.9,
                 color="#1f2329", fontweight="bold" if grp == "online" else "normal", zorder=5)
    for yh, title, grp in headers:
        axa.text(0.008, yh, title, ha="left", va="center", fontsize=7.4, fontstyle="italic",
                 fontweight="bold", color="#9b2a22" if grp == "online" else "#2f4b9a", zorder=3)
    best_on = max((r for r in rows if r[7] == "online"), key=lambda r: r[4])
    best_off = max((r for r in rows if r[7] == "offline"), key=lambda r: r[4])
    gain = 100.0 * (best_on[4] / best_off[4] - 1.0)
    xb = x_hi - 0.025
    axa.annotate("", xy=(xb, ypos[best_on[0]]), xytext=(xb, ypos[best_off[0]]),
                 arrowprops=dict(arrowstyle="<->", color="#1f2329", lw=0.8, shrinkA=2, shrinkB=2), zorder=5)
    axa.text(xb - 0.008, (ypos[best_on[0]] + ypos[best_off[0]]) / 2 - 0.15, f"+{gain:.0f}%", ha="right",
             va="center", fontsize=7.6, fontweight="bold", color="#9b2a22", zorder=5,
             bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    axa.set_yticks([ypos[r[0]] for r in rows])
    axa.set_yticklabels([r[0] for r in rows])
    for t, r in zip(axa.get_yticklabels(), rows):
        if r[7] == "online":
            t.set_fontweight("bold")
    axa.set_ylim(y + 0.3, 0.6)
    axa.set_xlim(x_lo, x_hi)
    axa.set_xlabel("Average per-layer reward")
    axa.grid(axis="x", color=GRID, lw=0.6); axa.set_axisbelow(True)
    axa.tick_params(axis="y", length=0)
    _panel_label(axa, "a", x=-0.10)
    axa.set_title("Average reward over all layers", pad=11, fontsize=9.5, fontweight="bold")

    # (b) per-layer reward: online solid / offline dashed; bands = std across seeds
    handles = {"online": [], "offline": []}
    Lx = np.arange(1, rows[0][6].shape[1] + 1)
    for label, col, mk, lw, _m, _e, curve, grp in rows[::-1]:
        mu = curve.mean(0)
        if curve.shape[0] > 1:
            axb.fill_between(Lx, mu - curve.std(0), mu + curve.std(0), color=col, alpha=0.15, lw=0, zorder=2)
        ls = "-" if grp == "online" else ((0, (1.2, 1.2)) if label.startswith("Constant") else (0, (3.5, 1.6)))
        h, = axb.plot(Lx, mu, color=col, marker=mk or None, ms=5.0 if grp == "online" else 4.2, mec="white",
                      mew=0.6, lw=lw, label=label, ls=ls, zorder=5 if grp == "online" else 3)
        handles[grp].insert(0, h)
    axb.set_xlabel("Layer")
    axb.set_ylabel("Reward")
    axb.set_xticks(range(1, 13))
    axb.set_xlim(0.6, 12.4)
    axb.set_ylim(0.3, 1.0)
    axb.grid(color=GRID, lw=0.6); axb.set_axisbelow(True)
    leg_kw = dict(fontsize=6.8, frameon=False, handlelength=2.4, columnspacing=1.2, title_fontsize=7.2)
    l1 = fig.legend(handles["online"], [h.get_label() for h in handles["online"]], ncol=1, loc="upper left",
                    bbox_to_anchor=(0.08, -0.01), title=ONLINE_TITLE, **leg_kw)
    l2 = fig.legend(handles["offline"], [h.get_label() for h in handles["offline"]], ncol=3, loc="upper left",
                    bbox_to_anchor=(0.40, -0.01), title=OFFLINE_TITLE, **leg_kw)
    for lg, c in ((l1, "#9b2a22"), (l2, "#2f4b9a")):
        lg.get_title().set_fontweight("bold"); lg.get_title().set_fontstyle("italic"); lg.get_title().set_color(c)
        lg._legend_box.align = "left"
    _panel_label(axb, "b", x=-0.13)
    axb.set_title("Reward per layer", pad=11, fontsize=9.5, fontweight="bold")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    with open(f"{args.out}_data.json", "w") as f:                   # everything drawn, for restyling later
        json.dump(dict(reward="EoH ROI mean exp(-d/W_T) (w = 1), real physics",
                       methods=[dict(label=r[0], group=r[7], color=r[1], marker=r[2], mean=float(r[4]),
                                     err=float(r[5]), per_layer_seed_curves=r[6].tolist()) for r in rows]), f, indent=1)
    print("saved", args.out, "(+ _data.json) |", ", ".join(f"{r[0]} {r[4]:.3f}±{r[5]:.3f}" for r in rows))


if __name__ == "__main__":
    main()
