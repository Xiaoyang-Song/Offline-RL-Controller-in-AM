"""
Document/make_traj_figure_wide.py
Wide (3 x 8) version of Document/make_traj_figure.py: auto-regressive rollout of one held-out build,
rows = true / predicted / relative error, columns = layers 1, 4, 8, 12, each with its heating (EoH) and
cooling (EoC) field. Same conventions as the original (jet fields, hot relative-error map); fonts and sizes from PAPER_RC,
as in the other paper figures.
The build is the first test build whose maximum temperature over the SHOWN layers (both stages) is
<= --max_temp, else the coolest one (unless --traj is given).

    python Document/make_traj_figure_wide.py   -> Document/figures/fig_example_trajectory_wide.{png,pdf}
"""
import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from surrogate_model_v3.dataset import TwoStageTrajectoryDataset, load_trajectories, split_trajectories
from surrogate_model_v3.evaluate import _DropBootstrapMask, evaluate_rollout, load_mesh
from surrogate_model_v3.model import load_surrogate
from surrogate_model_v3.paper_figure_ood import PAPER_RC

plt.rcParams.update(PAPER_RC)                                   # same fonts/sizes as the other paper figures
plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})   # full frame around each map

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="surrogate_model_v3/runs/full_range/surrogate_best.pt")
p.add_argument("--data_path", default="Data/DatasetV2_layer_12_samples_5000.pkl")
p.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
p.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12])
p.add_argument("--traj", type=int, default=-1, help="Test-build index; -1 = first one with max T <= --max_temp.")
p.add_argument("--max_temp", type=float, default=3000.0)
p.add_argument("--out", default=os.path.join(ROOT, "Document/figures/fig_example_trajectory_wide"))
a = p.parse_args()

model, sm, ss, lm, ls, cm, cs = load_surrogate(a.checkpoint, "cpu")
_, _, te = split_trajectories(load_trajectories(a.data_path), 0.10, 0.10, 42)
tmax = lambda tr, ls=None: max(max(np.max(s.u_heat_final), np.max(s.u_final))
                               for k, s in enumerate(tr) if ls is None or k + 1 in ls)
all_max = np.array([tmax(tr) for tr in te]); shown_max = np.array([tmax(tr, a.layers) for tr in te])
print("test-build max T, all layers   : min %.0f, 5%% %.0f, median %.0f K" % (all_max.min(), *np.percentile(all_max, [5, 50])))
print("test-build max T, shown layers : min %.0f, 5%% %.0f, median %.0f K, %d builds <= %.0f K"
      % (shown_max.min(), *np.percentile(shown_max, [5, 50]), (shown_max <= a.max_temp).sum(), a.max_temp))
if a.traj < 0:
    ok = np.flatnonzero(shown_max <= a.max_temp)              # criterion on the layers shown in the figure
    a.traj = int(ok[0]) if len(ok) else int(np.argmin(shown_max))
traj = te[a.traj:a.traj + 1]
print(f"test build {a.traj}: max T = {tmax(traj[0]):.0f} K, cool time {traj[0][0].cool_time:.2f} s, "
      f"powers {[s.lp_action for s in traj[0]]}")

ds = TwoStageTrajectoryDataset(traj, state_mean=sm, state_std=ss, lp_mean=lm, lp_std=ls, cool_mean=cm,
                               cool_std=cs, initial_temp=300.0, n_ensemble=model.n_ensemble, bootstrap_seed=42)
r = evaluate_rollout(model, _DropBootstrapMask(DataLoader(ds, batch_size=1, num_workers=0)),
                     sm, ss, lm, ls, cm, cs, "cpu", traj_len=len(traj[0]))
triang = load_mesh(a.mesh_path)
hp, hg, np_, ng = (r[k][0] for k in ("all_heat_pred", "all_heat_gt", "all_next_pred", "all_next_gt"))
acts = r["all_actions"][0]
L = [l - 1 for l in a.layers]

pct = lambda pr, gt: np.abs(pr - gt) / np.maximum(np.abs(gt), 1.0) * 100.0
cols = [(l, st, gt, pr) for l in L for st, gt, pr in (("Heating", hg, hp), ("Cooling", ng, np_))]
vmax_T = float(np.ceil(max(max(gt[l].max(), pr[l].max()) for l, _s, gt, pr in cols) / 250) * 250)
emax = max(pct(pr[l], gt[l]).max() for l, _s, gt, pr in cols)

fig, axes = plt.subplots(3, len(cols), figsize=(7.2, 2.75), sharex=True, sharey=True,
                         gridspec_kw=dict(hspace=0.10, wspace=0.10))
for j, (l, st, gt, pr) in enumerate(cols):
    for i, (fld, kind) in enumerate(((gt[l], "True"), (pr[l], "Predicted"), (pct(pr[l], gt[l]), "Rel. error"))):
        ax = axes[i, j]
        if kind == "Rel. error":
            ax.tripcolor(triang, fld, cmap="hot", vmin=0, vmax=emax, shading="gouraud")
        else:
            ax.tripcolor(triang, fld, cmap="jet", vmin=300, vmax=vmax_T, shading="gouraud")
        ax.set_aspect("auto"); ax.set_xticks([0, 6, 12]); ax.set_yticks([0, 1.5, 3])
        if i == 0:
            ax.set_title(st, fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{kind}\nY [mm]")
        if i == 2:
            ax.set_xlabel("X [mm]")
cb = fig.colorbar(axes[0, 0].collections[0], ax=axes[:2, :], pad=0.012, shrink=0.92, aspect=18)
cb.set_label("Temperature [K]")
cb2 = fig.colorbar(axes[2, 0].collections[0], ax=axes[2, :], pad=0.012, shrink=0.92, aspect=9)
cb2.set_label("Rel. error [%]")
for j, (l, st, _gt, _pr) in enumerate(cols):            # layer header over each heating/cooling pair
    if st == "Heating":                                  # (after the colorbars, which shrink the axes)
        x0, x1 = axes[0, j].get_position().x0, axes[0, j + 1].get_position().x1
        fig.text((x0 + x1) / 2, axes[0, j].get_position().y1 + 0.085, f"Layer {l + 1},  $a$ = {acts[l]:.0f} W",
                 ha="center", fontsize=8, fontweight="bold")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
np.savez_compressed(f"{a.out}_data.npz", heat_true=hg, heat_pred=hp, cool_true=ng, cool_pred=np_, actions=acts,
                    layers=np.array(a.layers), test_build=a.traj)
print("Saved →", a.out, "| T range 300 -", vmax_T, "K | max rel. error", f"{emax:.2f}%")
for l in L:
    print(f"layer {l+1}: a={acts[l]:.0f}W  heat MAE={np.abs(hp[l]-hg[l]).mean():.2f}K  cool MAE={np.abs(np_[l]-ng[l]).mean():.2f}K")
