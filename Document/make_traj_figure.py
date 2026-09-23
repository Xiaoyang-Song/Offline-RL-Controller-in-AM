"""
Document/make_traj_figure.py
Compact example-trajectory figure for the surrogate paper: auto-regressive rollout of one
held-out build (test trajectory 0), four layers, heating and cooling stages, in the same
colour conventions as surrogate_model_v3/plot_transitions.py (jet fields, hot relative-error map).

    python Document/make_traj_figure.py --checkpoint surrogate_model_v3/runs/full_range/surrogate_best.pt
"""
import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from surrogate_model_v3.dataset import TwoStageTrajectoryDataset, load_trajectories, split_trajectories
from surrogate_model_v3.evaluate import _DropBootstrapMask, evaluate_rollout, load_mesh
from surrogate_model_v3.model import load_surrogate

plt.rcParams.update({"font.family": "serif", "font.size": 9})

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="surrogate_model_v3/runs/full_range/surrogate_best.pt")
p.add_argument("--data_path", default="Data/DatasetV2_layer_12_samples_5000.pkl")
p.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
p.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12])
p.add_argument("--traj", type=int, default=0)
p.add_argument("--out", default=os.path.join(ROOT, "Document/figures/fig_example_trajectory.png"))
a = p.parse_args()

model, sm, ss, lm, ls, cm, cs = load_surrogate(a.checkpoint, "cpu")
_, _, te = split_trajectories(load_trajectories(a.data_path), 0.10, 0.10, 42)
traj = te[a.traj:a.traj + 1]
ds = TwoStageTrajectoryDataset(traj, state_mean=sm, state_std=ss, lp_mean=lm, lp_std=ls, cool_mean=cm,
                               cool_std=cs, initial_temp=300.0, n_ensemble=model.n_ensemble, bootstrap_seed=42)
r = evaluate_rollout(model, _DropBootstrapMask(DataLoader(ds, batch_size=1, num_workers=0)),
                     sm, ss, lm, ls, cm, cs, "cpu", traj_len=len(traj[0]))
triang = load_mesh(a.mesh_path)
hp, hg, np_, ng = (r[k][0] for k in ("all_heat_pred", "all_heat_gt", "all_next_pred", "all_next_gt"))
acts = r["all_actions"][0]
L = [l - 1 for l in a.layers]

pct = lambda pr, gt: np.abs(pr - gt) / np.maximum(np.abs(gt), 1.0) * 100.0
stages = (("Heating", hg, hp), ("Cooling", ng, np_))
emax = {s: max(pct(pr[l], gt[l]).max() for l in L) for s, gt, pr in stages}

fig, axes = plt.subplots(6, len(L), figsize=(7.4, 7.6), sharex=True, sharey=True,
                         gridspec_kw=dict(hspace=0.16, wspace=0.08))
for si, (name, gt, pr) in enumerate(stages):
    for j, l in enumerate(L):
        for ri, (fld, kind) in enumerate(((gt[l], "GT"), (pr[l], "Pred"), (pct(pr[l], gt[l]), "Err"))):
            ax = axes[3 * si + ri, j]
            if kind == "Err":
                h = ax.tripcolor(triang, fld, cmap="hot", vmin=0, vmax=emax[name], shading="gouraud")
            else:
                h = ax.tripcolor(triang, fld, cmap="jet", vmin=300, vmax=5000, shading="gouraud")
            ax.set_aspect("auto")
            if 3 * si + ri == 0: ax.set_title(f"Layer {l+1}\n$a$ = {acts[l]:.0f} W", fontsize=9)
            if j == 0: ax.set_ylabel({"GT": "True", "Pred": "Predicted", "Err": "Rel. error"}[kind] + "\nY [mm]", fontsize=8)
            if 3 * si + ri == 5: ax.set_xlabel("X [mm]")
            if j == len(L) - 1 and kind != "Pred":
                pass
    cb = fig.colorbar(h if False else axes[3 * si, 0].collections[0], ax=axes[3 * si:3 * si + 2, :], pad=0.015, shrink=0.9)
    cb.set_label("Temperature [K]", fontsize=8)
    cb2 = fig.colorbar(axes[3 * si + 2, 0].collections[0], ax=axes[3 * si + 2, :], pad=0.015, shrink=0.9)
    cb2.set_label("Rel. error [%]", fontsize=8)
    fig.text(0.005, 0.72 - 0.4 * si, name, rotation=90, fontsize=11, fontweight="bold", va="center")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=250, bbox_inches="tight"); print("Saved →", a.out)

print("cool time [s]:", float(traj[0][0].cool_time))
for l in L:
    print(f"layer {l+1}: a={acts[l]:.0f}W  heat MAE={np.abs(hp[l]-hg[l]).mean():.2f}K max rel={pct(hp[l],hg[l]).max():.2f}%  "
          f"cool MAE={np.abs(np_[l]-ng[l]).mean():.2f}K max rel={pct(np_[l],ng[l]).max():.2f}%")
