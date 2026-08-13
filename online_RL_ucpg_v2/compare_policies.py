"""
online_RL_ucpg_v2/compare_policies.py
--------------------------------------
Side-by-side comparison of two or more trained policies (typically
baselines/naive_pg vs. online_RL_ucpg_v2) purely from the JSON files
online_RL_ucpg_v2/evaluate_real.py already writes via --results_out — no
surrogate/MATLAB/torch dependency here, just post-hoc analysis of
already-computed REAL-simulator episodes.

This is the deliverable for the "does the uncertainty constraint actually
help" question: a naive policy has no reason to avoid a region the surrogate
can't reliably model (e.g. an interior laser-power gap trained via
surrogate_model_latent_uncertainty_v2's --lp_filter_ranges — see that
package's README), so it may end up choosing actions there whenever doing so
looks good ACCORDING TO THE SURROGATE, even if the real physics disagrees.
The uncertainty-constrained policy, in contrast, pays a Lagrangian penalty
for high predicted uncertainty and should be pushed away from that region on
its own, without the reward function ever knowing the region exists. This
script quantifies both sides of that story from real-simulator episodes:
  - REAL return (the only number that isn't self-deceiving — see
    evaluate_real.py's docstring for why the surrogate's own reported reward
    can't be trusted for actions outside its training coverage)
  - fraction of chosen actions landing outside the surrogate's training
    coverage (--id_ranges), i.e. how often each policy actually visited the
    danger zone

Usage
-----
    # 1. Generate one --results_out JSON per policy first (real-simulator
    #    episodes; module load matlab; see jobs/evaluate_real_ucpg_v2.sh):
    #        python -m online_RL_ucpg_v2.evaluate_real --checkpoint ... --results_out a.json
    #        python -m online_RL_ucpg_v2.evaluate_real --checkpoint ... --results_out b.json

    # 2. Compare:
    python -m online_RL_ucpg_v2.compare_policies \\
        --results naive_pg=jobs/eval_real_naive_pg.json ucpg_v2=jobs/eval_real_ucpg_v2.json \\
        --id_ranges "150-200,300-350" \\
        --out_dir online_RL_ucpg_v2/runs/compare_gap

Outputs (--out_dir)
--------------------
  compare_returns_actions.png  — return bar chart + action-histogram overlay
                                  (training coverage shaded, same convention
                                  as surrogate_model_latent_uncertainty_v2's
                                  evaluate_ood.py plots)
  compare_summary.json         — the printed table, machine-readable
  Console table: per-policy real return (mean +/- std across episodes),
  mean/std action, and fraction of chosen actions outside --id_ranges.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_ranges(spec: str):
    """'150-200,300-350' -> [(150.0, 200.0), (300.0, 350.0)] — same syntax as
    train.py's --lp_filter_ranges / evaluate_ood.py's --id_ranges."""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_str, hi_str = part.split("-")
        ranges.append((float(lo_str), float(hi_str)))
    if not ranges:
        raise ValueError(f"'{spec}' parsed to an empty range list.")
    return ranges


def _parse_results(specs):
    """['naive_pg=a.json', 'ucpg_v2=b.json'] -> {'naive_pg': 'a.json', ...}"""
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--results entries must be 'label=path.json', got '{spec}'")
        label, path = spec.split("=", 1)
        out[label] = path
    return out


def load_policy_results(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    actions = np.asarray(payload["greedy_actions"])   # (n_episodes, n_layers)
    rewards = np.asarray(payload["greedy_rewards"])   # (n_episodes, n_layers)
    ep_returns = rewards.sum(axis=1)
    return {
        "checkpoint":     payload.get("checkpoint", path),
        "actions":        actions,
        "rewards":        rewards,
        "ep_returns":     ep_returns,
        "return_mean":    float(ep_returns.mean()),
        "return_std":     float(ep_returns.std()),
        "n_episodes":     actions.shape[0],
    }


def summarize(label: str, r: dict, id_ranges) -> dict:
    flat_actions = r["actions"].reshape(-1)
    if id_ranges is not None:
        in_id = np.zeros_like(flat_actions, dtype=bool)
        for lo, hi in id_ranges:
            in_id |= (flat_actions >= lo) & (flat_actions <= hi)
        frac_outside = float((~in_id).mean())
    else:
        frac_outside = float("nan")

    n = r["n_episodes"]
    return_str = (f"{r['return_mean']:+.4f} ± {r['return_std']:.4f}" if n > 1
                  else f"{r['return_mean']:+.4f} (n=1 episode)")
    print(f"  {label:<14} return={return_str:<28} "
          f"action mean={flat_actions.mean():7.1f}W std={flat_actions.std():6.1f}W  "
          f"{'frac. outside training coverage=' + f'{frac_outside:.1%}' if id_ranges is not None else ''}")
    return {
        "checkpoint":       r["checkpoint"],
        "n_episodes":       n,
        "return_mean":      r["return_mean"],
        "return_std":       r["return_std"],
        "action_mean":      float(flat_actions.mean()),
        "action_std":       float(flat_actions.std()),
        "frac_outside_id":  frac_outside,
    }


def plot_comparison(results: dict, id_ranges, out_path: str) -> None:
    labels = list(results.keys())
    fig, (ax_ret, ax_act) = plt.subplots(1, 2, figsize=(13, 4.5))

    means = [results[l]["return_mean"] for l in labels]
    stds  = [results[l]["return_std"]  for l in labels]
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    ax_ret.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    ax_ret.set_ylabel("Real-simulator episode return (greedy)")
    ax_ret.set_title("Return vs. real physics", fontsize=10)
    ax_ret.axhline(0, color="grey", linewidth=0.8)
    ax_ret.grid(True, axis="y", alpha=0.3)

    all_actions = np.concatenate([results[l]["actions"].reshape(-1) for l in labels])
    bins = np.linspace(all_actions.min() - 5, all_actions.max() + 5, 40)
    for l, color in zip(labels, colors):
        ax_act.hist(results[l]["actions"].reshape(-1), bins=bins, alpha=0.5,
                    label=l, color=color, density=True)
    if id_ranges is not None:
        for i, (lo, hi) in enumerate(id_ranges):
            ax_act.axvspan(lo, hi, color="tab:green", alpha=0.08,
                          label="Surrogate training coverage" if i == 0 else None)
    ax_act.set_xlabel("Chosen laser power [W]")
    ax_act.set_ylabel("Density")
    ax_act.set_title("Action distribution", fontsize=10)
    ax_act.legend(fontsize=8)
    ax_act.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare_policies] Saved → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two or more trained policies' REAL-simulator evaluate_real.py "
                    "results (e.g. baselines/naive_pg vs. online_RL_ucpg_v2)."
    )
    p.add_argument("--results", type=str, nargs="+", required=True,
                   help="One or more 'label=path/to/results.json' — the --results_out file "
                        "from online_RL_ucpg_v2/evaluate_real.py for that checkpoint.")
    p.add_argument("--id_ranges", type=str, default=None,
                   help="Comma-separated 'lo-hi' laser-power ranges the SURROGATE these "
                        "policies were trained against actually covers, e.g. "
                        "'150-200,300-350' (match the surrogate's --lp_filter_ranges, or a "
                        "single 'lo-hi' for a plain narrow surrogate). Used to report/shade "
                        "what fraction of each policy's chosen actions fall outside the "
                        "surrogate's training coverage. Omit to skip this comparison.")
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    id_ranges = _parse_ranges(args.id_ranges) if args.id_ranges is not None else None

    specs = _parse_results(args.results)
    raw = {label: load_policy_results(path) for label, path in specs.items()}

    print("=" * 78)
    print("[compare_policies] Real-simulator policy comparison")
    print("=" * 78)
    summary = {label: summarize(label, r, id_ranges) for label, r in raw.items()}
    print("=" * 78)

    plot_comparison(raw, id_ranges, os.path.join(args.out_dir, "compare_returns_actions.png"))

    summary_path = os.path.join(args.out_dir, "compare_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[compare_policies] Summary → {summary_path}")


if __name__ == "__main__":
    main()
