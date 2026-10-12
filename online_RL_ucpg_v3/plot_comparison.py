"""
online_RL_ucpg_v3/plot_comparison.py
----------------------------------------
Comparison figure for the v3 experiment, from the files in --results_dir:
  per_episode_surrogate.json  (evaluate_surrogate.py)  — optional
  per_layer_real.json         (evaluate_real.py)       — optional
(at least one must exist).

Panels, all scored with the PLAIN (unpenalised) deviation on the field the
methods were trained on (--field cool: post-cooling; heat: end-of-heating):
  (a) surrogate evaluation — mean ROI deviation per layer (bar), episode std (whisker)
  (b) real-physics evaluation — same, whisker = spread over the real episodes
  (c) laser power per layer in the real evaluation (or the surrogate one if no real results)

One shared method order (by real deviation if available, else surrogate).
Shaped (v3) methods in solid colour, their matched plain baselines in a lighter
shade of the same colour, earlier reference methods in grey.

Usage
-----
    python -m online_RL_ucpg_v3.plot_comparison --results_dir online_RL_ucpg_v3/results_real_narrow_cool --field cool
"""

import argparse
import json
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK_2, GRID, _panel_label

PG_C, PG_L = "#264653", "#9fb8bf"       # policy gradient: shaped / plain
Q_C, Q_L = "#e76f51", "#f4b9a7"         # Q-learning:      shaped / plain
REF_C = "#b8b6b0"                       # earlier reference methods


def _colour(name):
    n = name.lower()
    if "earlier" in n:
        return REF_C
    is_q = n.startswith("q") or " q" in n
    shaped = "shaped" in n or "v3" in n
    return (Q_C if shaped else Q_L) if is_q else (PG_C if shaped else PG_L)


def _load(results_dir, field):
    """-> (surrogate stats or None, real stats or None); stats: {method: dict(dev=array per episode,
    layers=[...], lp=array per layer)}"""
    sur = real = None
    f = os.path.join(results_dir, "per_episode_surrogate.json")
    if os.path.exists(f):
        sur = OrderedDict()
        for e in json.load(open(f))["episodes"]:
            d = sur.setdefault(e["method"], dict(dev=[], acts=[]))
            d["dev"].append(e["dev_K"]); d["acts"].append(e["actions"])
        for d in sur.values():
            d["dev"] = np.array(d["dev"]); A = np.array(d["acts"])
            d["layers"] = list(range(1, A.shape[1] + 1)); d["lp"] = A.mean(0)
    f = os.path.join(results_dir, "per_layer_real.json")
    if os.path.exists(f):
        real = OrderedDict()
        rows = json.load(open(f))
        key = f"dev_{field}_K"
        for m in OrderedDict.fromkeys(r["method"] for r in rows):
            rs = [r for r in rows if r["method"] == m]
            eps = sorted({r["cool_time"] for r in rs})
            layers = sorted({r["layer"] for r in rs})
            real[m] = dict(dev=np.array([np.mean([r[key] for r in rs if r["cool_time"] == c]) for c in eps]),
                           layers=layers,
                           lp=np.array([np.mean([r["action_W"] for r in rs if r["layer"] == L]) for L in layers]))
    return sur, real


def _bars(ax, stats, names, y, whisker, title):
    for yi, n in zip(y, names):
        if n not in stats:
            continue
        v = stats[n]["dev"]
        m = v.mean()
        lo, hi = (m - v.std(), m + v.std()) if whisker == "std" else (v.min(), v.max())
        ax.barh(yi, m, color=_colour(n), height=0.68, zorder=2)
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], color=INK_2, elinewidth=0.6, capsize=1.5, zorder=3)
        ax.text(hi * 1.02 + 2, yi, f"{m:.0f}", va="center", fontsize=6.2, color=INK_2)
    ax.set_xlim(0, max(stats[n]["dev"].mean() + stats[n]["dev"].std() if whisker == "std" else stats[n]["dev"].max()
                       for n in stats) * 1.3)
    ax.set_xlabel("ROI deviation per layer [K]")
    ax.set_title(title, pad=3, color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--field", choices=["cool", "heat"], required=True,
                   help="Field the methods were trained on — the deviation that is scored.")
    p.add_argument("--out", default="", help="Default: <results_dir>/comparison")
    a = p.parse_args()
    plt.rcParams.update(PAPER_RC)

    sur, real = _load(a.results_dir, a.field)
    if sur is None and real is None:
        raise SystemExit(f"No per_episode_surrogate.json / per_layer_real.json in {a.results_dir}")
    ref = real if real is not None else sur
    names = sorted(ref, key=lambda n: ref[n]["dev"].mean())
    names += [n for n in (sur or {}) if n not in names]
    y = np.arange(len(names))[::-1]
    fname = "post-cooling" if a.field == "cool" else "end-of-heating"

    panels = [(s, w, t) for s, w, t in ((sur, "std", "Surrogate"), (real, "range", "Real physics")) if s is not None]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(7.0, 2.7),
                             gridspec_kw=dict(wspace=0.14, width_ratios=[1] * len(panels) + [1.15]))
    for j, (ax, (stats, whisker, title)) in enumerate(zip(axes, panels)):
        _bars(ax, stats, names, y, whisker, f"{title}: {fname} deviation")
        ax.set_yticks(y)
        ax.set_yticklabels(names if j == 0 else [], fontsize=6.2)
        _panel_label(ax, "abc"[j], x=-0.02, y=1.04)

    ax = axes[-1]
    lp_src, lp_title = (real, "real") if real is not None else (sur, "surrogate")
    for n in names:
        if n in lp_src:
            s = lp_src[n]
            ls = (0, (3, 1.6)) if "earlier" in n.lower() else "-"
            ax.plot(s["layers"], s["lp"], color=_colour(n), linestyle=ls, linewidth=1.3, marker="o",
                    markersize=2.2, label=n)
    ax.set_xlabel("Layer"); ax.set_ylabel("Laser power [W]")
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(False)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_title(f"Chosen laser power ({lp_title})", pad=3, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncols=2, handlelength=1.6)
    _panel_label(ax, "abc"[len(panels)], x=0.02)

    out = a.out or os.path.join(a.results_dir, "comparison")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    print(f"[plot_comparison] Saved → {out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
