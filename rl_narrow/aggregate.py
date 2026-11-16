"""
rl_narrow/aggregate.py
----------------------
Combines the per-method JSONs of rl_narrow/evaluate_real.py into the real-physics leaderboard.

Per method family (seeds pooled; names "... seed N"): EoH quality score (method.tex reward with w = 1,
ROI mean exp(-d/W_T), per layer, higher is better) and EoH ROI mean deviation [K] (lower is better),
first averaged over the 12 layers and the evaluation episodes of each seed, then mean ± std across seeds
(single-seed methods: std across episodes). Constant powers: every value is reported in
constant_sweep.csv and the best one (highest score, chosen in hindsight on these episodes) enters the
leaderboard as "Constant (best, X W)".

    python -m rl_narrow.aggregate --results_dir rl_narrow/results_real
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="rl_narrow/results_real")
    p.add_argument("--id_min", type=float, default=200.0)
    p.add_argument("--id_max", type=float, default=300.0)
    args = p.parse_args()

    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.results_dir, "*.json")))]
    runs = [r for r in runs if r.get("rows")]
    incomplete = [r["method"] for r in runs if not r.get("complete")]
    if incomplete:
        print(f"WARNING incomplete (partial episodes): {incomplete}")

    def per_episode(r, key):
        return np.array([np.mean([x[key] for x in r["rows"] if x["cool_time"] == ct])
                         for ct in r["cool_times"] if any(x["cool_time"] == ct for x in r["rows"])])

    fams, const = {}, []
    for r in runs:
        if r["type"] == "const":
            const.append(dict(power_W=float(r["path"]), score=per_episode(r, "score_heat").mean(),
                              dev_K=per_episode(r, "dev_heat_K").mean(), dev_cool_K=per_episode(r, "dev_cool_K").mean()))
            continue
        fam = re.sub(r"\s*seed\s*\d+\s*$", "", r["method"]).strip()
        fams.setdefault(fam, []).append(r)

    rows = []
    for fam, rs in fams.items():
        sc = [per_episode(r, "score_heat") for r in rs]
        dv = [per_episode(r, "dev_heat_K") for r in rs]
        dc = [per_episode(r, "dev_cool_K") for r in rs]
        acts = np.concatenate([[x["action_W"] for x in r["rows"]] for r in rs])
        agg = (lambda v: (np.mean([x.mean() for x in v]), np.std([x.mean() for x in v]))) if len(rs) > 1 else \
              (lambda v: (v[0].mean(), v[0].std()))
        (s_m, s_s), (d_m, d_s), (c_m, _c) = agg(sc), agg(dv), agg(dc)
        rows.append(dict(method=fam, n_seeds=len(rs), score_mean=s_m, score_std=s_s, dev_heat_K_mean=d_m,
                         dev_heat_K_std=d_s, dev_cool_K_mean=c_m, action_W_mean=float(acts.mean()),
                         frac_actions_outside_data=float(((acts < args.id_min) | (acts > args.id_max)).mean())))
    if const:
        const.sort(key=lambda c: c["power_W"])
        best = max(const, key=lambda c: c["score"])
        b = [r for r in runs if r["type"] == "const" and float(r["path"]) == best["power_W"]][0]
        sc, dv = per_episode(b, "score_heat"), per_episode(b, "dev_heat_K")
        rows.append(dict(method=f"Constant (best, {best['power_W']:.0f} W)", n_seeds=1, score_mean=sc.mean(),
                         score_std=sc.std(), dev_heat_K_mean=dv.mean(), dev_heat_K_std=dv.std(),
                         dev_cool_K_mean=best["dev_cool_K"], action_W_mean=best["power_W"],
                         frac_actions_outside_data=float(not args.id_min <= best["power_W"] <= args.id_max)))
        with open(os.path.join(args.results_dir, "constant_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(const[0].keys())); w.writeheader(); w.writerows(const)

    rows.sort(key=lambda r: -r["score_mean"])
    with open(os.path.join(args.results_dir, "leaderboard_real.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.results_dir, "per_layer_real.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "cool_time", "layer", "action_W", "score_heat", "dev_heat_K", "dev_cool_K"])
        for r in runs:
            for x in r["rows"]:
                w.writerow([r["method"], x["cool_time"], x["layer"], x["action_W"], x["score_heat"],
                            x["dev_heat_K"], x["dev_cool_K"]])

    print(f"\nREAL-physics leaderboard (narrow, EoH field; {len(runs)} runs)")
    print(f"{'method':<34}{'seeds':>6}{'score (w=1)':>18}{'deviation [K]':>18}{'mean LP':>9}{'outside':>9}")
    for r in rows:
        print(f"{r['method']:<34}{r['n_seeds']:>6}{r['score_mean']:>11.4f} ± {r['score_std']:<5.4f}"
              f"{r['dev_heat_K_mean']:>10.1f} ± {r['dev_heat_K_std']:<6.1f}{r['action_W_mean']:>8.1f}"
              f"{r['frac_actions_outside_data'] * 100:>8.0f}%")
    print(f"Saved → {args.results_dir}/leaderboard_real.csv, per_layer_real.csv, constant_sweep.csv")


if __name__ == "__main__":
    main()
