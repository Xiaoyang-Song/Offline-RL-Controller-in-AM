# rl_narrow — RL experiments of the paper (NARROW setting)

Every method controls the same 12-layer build. The digital twin and the offline methods only see data
with laser power in 200–300 W (`surrogate_model_v3/runs/narrow_200_300W`, and the same transitions of
the training builds). All methods are evaluated in the **real physics** simulator (MATLAB PDE,
heatTime 0.025 s) on identical greedy episodes, one per cooling time 0.05, 0.07, …, 0.15 s.

| Method | Learns from | Reward | Code |
|---|---|---|---|
| PG shaped (ours), 3 seeds | digital twin, online | method.tex: mean_ROI w exp(−d/W_T), w = τ²/(τ²+σ²), σ = propagated total EoH std, τ = τ_k × median in-distribution σ (τ_k = 8, `TAU_K`) | `online_RL_ucpg_v3.train_pg` |
| Online Q shaped, 3 seeds | digital twin, online | same as ours | `online_RL_ucpg_v3.train_q` |
| Offline FQI / CQL / IQL, 3 seeds each | logged narrow data | same with w = 1 (no model ⇒ no model uncertainty) | `rl_narrow/offline_rl.py` |
| Proportional | logged narrow data, least squares | — | `baselines.proportional` |
| Kalman filter | logged narrow data, least squares | — | `baselines.kalman_particle` |
| Constant 100–400 W | — | — (best in hindsight reported) | `baselines.constant` |

Scores (`rl_narrow/aggregate.py`): EoH quality score = ROI mean exp(−d/W_T) (method.tex reward with
w = 1), and EoH ROI mean deviation [K]; mean ± std across seeds.

```bash
bash rl_narrow/jobs/run_all.sh          # submits training (16 GPU tasks) -> real eval (48 CPU tasks) -> aggregate
python -m rl_narrow.aggregate           # re-aggregate at any time (partial results are flagged)
```
Outputs: `rl_narrow/runs/<method>_s<seed>/`, `rl_narrow/results_real/{*.json, leaderboard_real.csv,
per_layer_real.csv, constant_sweep.csv}`, logs in `rl_narrow/jobs/logs/`.

Figures (after the evaluation):
```bash
python -m rl_narrow.plot_results --out Document/figures/fig_rl_comparison   # (a) average reward bars, (b) reward per layer
python -m rl_narrow.plot_maps    --out Document/figures/fig_rl_maps         # real EoH deviation maps, rows = methods (ours last), layers 1/4/8/12
```
