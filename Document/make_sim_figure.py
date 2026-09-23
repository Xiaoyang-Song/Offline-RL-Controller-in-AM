"""Figure for surrogate.tex: LPBF simulation geometry + temperature evolution for one trajectory."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_model_v3.dataset import load_trajectories
from baselines.common.data_utils import load_mesh_nodes

W, H, L = 12.0, 3.0, 12
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

trajs = load_trajectories("Data/DatasetV2_layer_12_samples_5000.pkl")
traj = trajs[0]
nodes = load_mesh_nodes("surrogate_model/mesh.mat")
tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
fracs = np.linspace(0.4, 0.5, L)

layers = [0, 3, 7, 11]
fig, axes = plt.subplots(2, len(layers), figsize=(13, 2.5), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.02, wspace=0.08))
vmax = max(np.max(traj[l].u_heat_final) for l in layers)
for j, l in enumerate(layers):
    for i, (fld, name) in enumerate(((traj[l].u_heat_final, "$u_l$ (heating)"), (traj[l].u_final, "$s_{l+1}$ (cooling)"))):
        ax = axes[i, j]
        cs = ax.tricontourf(tri, np.clip(np.asarray(fld).ravel(), 300, vmax), levels=np.linspace(300, vmax, 41), cmap="inferno")
        side = min(W, H) * fracs[l]
        ax.add_patch(Rectangle((W/2 - side/2, H/2 - side/2), side, side, fill=False, ec="cyan", lw=1.4))
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title(f"Layer {l+1}  ($a$={traj[l].lp_action:.0f} W)")
        if j == 0:
            ax.set_ylabel(name + "\n$y$ [mm]", fontsize=9)
        if i == 1:
            ax.set_xlabel("$x$ [mm]")
fig.colorbar(cs, ax=axes, label="Temperature [K]", shrink=0.9, pad=0.015)
out = "Document/figures/sim_evolution.png"
fig.savefig(out, dpi=250, bbox_inches="tight")
print("saved", out, "cool_time", traj[0].cool_time, "actions", [round(s.lp_action) for s in traj])
