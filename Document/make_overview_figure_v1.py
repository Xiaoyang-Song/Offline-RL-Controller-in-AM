"""
Document/make_overview_figure.py
Method overview figure for the paper (Section Methodology): the multilayer FE process, the
two-stage probabilistic ensemble digital twin with stage/layer uncertainty propagation, and
surrogate-based online RL with the uncertainty-shaped reward. All small panels are drawn from
real data / results:
  * temperature fields   : test build 0 (as in fig_example_trajectory) from Data/DatasetV2_layer_12_samples_5000.pkl
  * propagation ratio map: surrogate_model_v3/results_propagation_maps_narrow/propagation_maps.npz
  * multistep curves     : surrogate_model_v3/results_multistep/paper_fig_multistep_gap_matched.npz (narrow)

    python Document/make_overview_figure.py      -> Document/figures/fig_overview.{pdf,png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from surrogate_model_v3.dataset import load_trajectories, split_trajectories
from surrogate_model_v3.paper_figure_ood import PAPER_RC, INK, INK_2, INK_3
from baselines.common.data_utils import load_mesh_nodes

HEAT, HEAT_L = "#c4352d", "#f6d9d3"          # heating red (paper palette)
COOL, COOL_L = "#3b55c4", "#dbe1f6"          # cooling blue
PROP, PROP_L = "#1f7a70", "#d7eeea"          # propagation teal
RL, RL_L = "#6d597a", "#ece6f0"              # control purple
PANEL_BG = "#fafafb"
W_DOM, H_DOM = 12.0, 3.0

plt.rcParams.update(PAPER_RC)
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 5.5})

fig = plt.figure(figsize=(7.2, 3.75))
F = fig.transFigure


def box(x, y, w, h, fc, ec, lw=0.8, r=0.012, z=1, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=F,
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    fig.add_artist(p)
    return p


def arrow(x0, y0, x1, y1, color=INK_2, lw=1.0, style="-|>", ms=7, ls="-", rad=0.0, z=5):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=F, arrowstyle=style, mutation_scale=ms,
                                   color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))


def text(x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    fig.text(x, y, s, transform=F, **kw)


def badge(x, y, n, color):
    fig.add_artist(plt.Circle((x, y), 0.0115, transform=F, fc=color, ec="none", zorder=6))
    text(x, y - 0.001, str(n), color="white", fontsize=7, fontweight="bold", zorder=7)


# ---------------------------------------------------------------- column frames
cols = [(0.010, 0.300, HEAT), (0.322, 0.355, PROP), (0.690, 0.300, RL)]
y0, h0 = 0.205, 0.765
titles = ["Multilayer thermal process", "Two-stage probabilistic digital twin", "Uncertainty-aware online RL"]
subs = ["finite-element simulation (ground truth)", "stage-wise ensembles + propagated uncertainty",
        "surrogate as the training environment"]
for i, ((x, w, c), t, s) in enumerate(zip(cols, titles, subs)):
    box(x, y0, w, h0, PANEL_BG, "#d9d8de", lw=0.6, r=0.015, z=0)
    badge(x + 0.022, y0 + h0 - 0.035, i + 1, c)
    text(x + 0.040, y0 + h0 - 0.031, t, ha="left", fontsize=8, fontweight="bold")
    text(x + 0.040, y0 + h0 - 0.072, s, ha="left", fontsize=6.2, color=INK_2, style="italic")
arrow(0.312, 0.60, 0.330, 0.60, color=INK_2, lw=1.4, ms=9)
arrow(0.679, 0.60, 0.698, 0.60, color=INK_2, lw=1.4, ms=9)

# ---------------------------------------------------------------- (1) process: real fields
_, _, test_trajs = split_trajectories(load_trajectories(os.path.join(ROOT, "Data/DatasetV2_layer_12_samples_5000.pkl")),
                                     0.10, 0.10, 42)
traj = test_trajs[0]                 # same held-out build as fig_example_trajectory
nodes = load_mesh_nodes(os.path.join(ROOT, "surrogate_model/mesh.mat"))
tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
fracs = np.linspace(0.4, 0.5, 12)
layers = [0, 11]
vmax = 5000.0
fx, fw, fh = 0.030, 0.255, 0.105
rows = [(0.735, "heating", HEAT), (0.600, "cooling", COOL), (0.430, "heating", HEAT), (0.295, "cooling", COOL)]
for r, (yy, phase, col) in enumerate(rows):
    l = layers[r // 2]
    st = traj[l]
    fld = np.asarray(st.u_heat_final if phase == "heating" else st.u_final).ravel()
    ax = fig.add_axes([fx + 0.045, yy, fw - 0.045, fh])
    ax.tricontourf(tri, np.clip(fld, 300, vmax), levels=np.linspace(300, vmax, 31), cmap="inferno")
    side = min(W_DOM, H_DOM) * fracs[l]
    ax.add_patch(Rectangle((W_DOM / 2 - side / 2, H_DOM / 2 - side / 2), side, side, fill=False, ec="#5ee0d6", lw=0.9))
    ax.set_xlim(0, W_DOM); ax.set_ylim(0, H_DOM); ax.set_aspect("equal"); ax.axis("off")
    for sp in ax.spines.values():
        sp.set_visible(False)
    lab = (f"$\\mathbf{{u}}_{{{l + 1}}}$" if phase == "heating" else f"$\\mathbf{{s}}_{{{l + 2}}}$")
    text(fx + 0.020, yy + fh / 2 + 0.012, lab, fontsize=7.5, color=col)
    text(fx + 0.020, yy + fh / 2 - 0.022, phase, fontsize=5.5, color=col)
text(fx + 0.15, 0.852, f"layer 1  ($a_1$ = {traj[0].lp_action:.0f} W)", fontsize=6.3, color=INK_2)
text(fx + 0.15, 0.547, f"layer 12  ($a_{{12}}$ = {traj[11].lp_action:.0f} W)", fontsize=6.3, color=INK_2)
arrow(fx + 0.012, 0.585, fx + 0.012, 0.545, color=INK_3, lw=0.8, ms=6, ls=(0, (2, 1.5)))
text(fx + 0.15, 0.255, "heat accumulates over layers $\\Rightarrow$ layer-wise control", fontsize=6, color=INK_2)
cax = fig.add_axes([0.105, 0.232, 0.15, 0.010])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(300, vmax), cmap="inferno"), cax=cax, orientation="horizontal")
cb.set_ticks([300, 2000, 2800, 5000]); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
cb.outline.set_linewidth(0.4)
text(0.070, 0.237, "T [K]", fontsize=5.5, color=INK_2)

# ---------------------------------------------------------------- (2) surrogate schematic
x2 = 0.322
yb, hb = 0.735, 0.085


def ensemble(cx, label, cond, col, colL):
    for k in range(3, 0, -1):                                    # stacked member cards
        box(cx - 0.045 + 0.006 * k, yb + 0.006 * k, 0.090, hb, colL, col, lw=0.5, r=0.008, z=2)
    box(cx - 0.045, yb, 0.090, hb, "white", col, lw=0.9, r=0.008, z=3)
    text(cx, yb + hb * 0.66, label, fontsize=6.6, fontweight="bold", color=col, zorder=4)
    text(cx, yb + hb * 0.28, f"$K$ Gaussian nets | {cond}", fontsize=5.4, color=INK_2, zorder=4)


ensemble(x2 + 0.105, "Heating", "$a_l$", HEAT, HEAT_L)
ensemble(x2 + 0.255, "Cooling", "$c$", COOL, COOL_L)
text(x2 + 0.025, yb + hb / 2, "$\\mathbf{s}_l$", fontsize=8)
text(x2 + 0.180, yb + hb / 2 + 0.002, "$\\mathbf{u}_l$", fontsize=8)
text(x2 + 0.333, yb + hb / 2, "$\\mathbf{s}_{l+1}$", fontsize=8)
arrow(x2 + 0.036, yb + hb / 2, x2 + 0.058, yb + hb / 2, color=INK_2)
arrow(x2 + 0.152, yb + hb / 2, x2 + 0.170, yb + hb / 2, color=INK_2)
arrow(x2 + 0.190, yb + hb / 2, x2 + 0.208, yb + hb / 2, color=INK_2)
arrow(x2 + 0.302, yb + hb / 2, x2 + 0.318, yb + hb / 2, color=INK_2)
arrow(x2 + 0.333, yb - 0.006, x2 + 0.025, yb - 0.006, color=INK_3, lw=0.7, ms=5, rad=-0.18, ls=(0, (2, 1.5)))
text(x2 + 0.180, yb - 0.048, "next layer", fontsize=5.3, color=INK_3)

# variance decomposition
text(x2 + 0.178, 0.648, "$\\mathrm{Var}(\\Delta)\\;=\\;\\mathbb{E}_k[\\sigma_k^2]\\;+\\;\\mathrm{Var}_k(\\mu_k)$",
     fontsize=7.4)
text(x2 + 0.192, 0.612, "aleatoric", fontsize=5.4, color=COOL)
text(x2 + 0.268, 0.612, "epistemic", fontsize=5.4, color=HEAT)
text(x2 + 0.178, 0.560,
     "$\\mathbf{P}\\;\\leftarrow\\;(\\mathbf{I}+\\mathbf{J})\\,\\mathbf{P}\\,(\\mathbf{I}+\\mathbf{J})^{\\top}+\\boldsymbol{\\Sigma}$"
     "   (stages & layers)", fontsize=6.6, color=PROP)

# inset A: propagation ratio map (real, narrow model, layer 6)
z = np.load(os.path.join(ROOT, "surrogate_model_v3/results_propagation_maps_narrow/propagation_maps.npz"))
t = 5
ratio = np.sqrt(z["all_prop_total"][t] / np.maximum(z["all_naive_total"][t], 1e-30))
axm = fig.add_axes([x2 + 0.012, 0.290, 0.165, 0.150])
lim = 1.6
axm.tricontourf(tri, np.clip(ratio, 1 / lim, lim), levels=np.geomspace(1 / lim, lim, 25), cmap="RdBu_r",
                norm=TwoSlopeNorm(vcenter=1.0, vmin=1 / lim, vmax=lim))
side = min(W_DOM, H_DOM) * fracs[t]
axm.add_patch(Rectangle((W_DOM / 2 - side / 2, H_DOM / 2 - side / 2), side, side, fill=False, ec=INK, lw=0.6, ls="--"))
axm.set_xlim(2.5, 9.5); axm.set_ylim(0, H_DOM); axm.set_aspect("equal"); axm.axis("off")
text(x2 + 0.095, 0.462, "cooling redistributes uncertainty", fontsize=5.8, color=INK_2)
text(x2 + 0.095, 0.270, "$\\sigma_{\\mathrm{prop}}/\\sigma_{\\mathrm{naive}}$:  < 1 in scan region, > 1 around",
     fontsize=5.3, color=INK_2)

# inset B: multistep curves (real, narrow model)
res = {n: r for n, r in np.load(os.path.join(ROOT, "surrogate_model_v3/results_multistep/paper_fig_multistep_gap_matched.npz"),
                                 allow_pickle=True)["results"]}
r = res["Narrow (200–300 W)"]
axc = fig.add_axes([x2 + 0.228, 0.300, 0.112, 0.150])
L = r["layers"]
axc.plot(L, r["rmse"], color=INK, lw=1.1, label="error")
axc.plot(L, r["prop"], color=PROP, lw=1.3, label="propagated $\\sigma$")
axc.plot(L, r["one"], color=INK_3, lw=1.0, ls=(0, (3, 1.5)), label="one-step $\\sigma$")
axc.set_xticks([1, 6, 12]); axc.set_yticks([0, 10, 20]); axc.set_xlim(0.5, 12.5); axc.set_ylim(0, 21)
axc.set_xlabel("layer", labelpad=1); axc.set_ylabel("[K]", labelpad=1)
axc.tick_params(length=1.5, pad=1)
axc.legend(loc="upper left", fontsize=4.6, frameon=False, handlelength=1.4, borderaxespad=0.1, labelspacing=0.2)
text(x2 + 0.270, 0.462, "uncertainty accumulates", fontsize=5.8, color=INK_2)

# ---------------------------------------------------------------- (3) RL loop
x3 = 0.690
pol = box(x3 + 0.030, 0.735, 0.110, 0.085, RL_L, RL, lw=0.9, r=0.010, z=2)
text(x3 + 0.085, 0.790, "Policy $\\pi_\\theta$", fontsize=7, fontweight="bold", color=RL)
text(x3 + 0.085, 0.757, "REINFORCE", fontsize=5.4, color=INK_2)
box(x3 + 0.165, 0.735, 0.115, 0.085, PROP_L, PROP, lw=0.9, r=0.010, z=2)
text(x3 + 0.2225, 0.790, "Digital twin", fontsize=7, fontweight="bold", color=PROP)
text(x3 + 0.2225, 0.757, "$\\hat{\\mathbf{u}}_l,\\;\\sigma_{l,i}$", fontsize=6.4, color=INK_2)
arrow(x3 + 0.140, 0.800, x3 + 0.165, 0.800, color=RL, lw=1.0)
text(x3 + 0.1525, 0.833, "$a_l$", fontsize=6.5, color=RL)
arrow(x3 + 0.165, 0.752, x3 + 0.140, 0.752, color=PROP, lw=1.0)
text(x3 + 0.1525, 0.712, "$\\mathbf{o}_{l+1},\\,\\tilde r_l$", fontsize=6.2, color=PROP)

text(x3 + 0.150, 0.632, "$\\tilde r_l=\\frac{1}{|\\mathcal{R}_l|}\\sum_{i\\in\\mathcal{R}_l} w_{l,i}\\,e^{-d_{l,i}/W_T}$",
     fontsize=7.4)
text(x3 + 0.150, 0.575, "$w_{l,i}=\\tau^2/(\\tau^2+\\sigma_{l,i}^2)$  (trust)", fontsize=6.4, color=INK_2)

# inset C: trust weight curve
axw = fig.add_axes([x3 + 0.040, 0.300, 0.105, 0.150])
sr = np.linspace(0, 3, 200)
axw.plot(sr, 1 / (1 + sr ** 2), color=RL, lw=1.3)
axw.axvspan(0, 0.5, color=PROP_L, lw=0)
axw.axvspan(1.5, 3, color="#f3e3e7", lw=0)
axw.set_xlim(0, 3); axw.set_ylim(0, 1.05); axw.set_xticks([0, 1, 2, 3]); axw.set_yticks([0, 0.5, 1])
axw.set_xlabel("$\\sigma/\\tau$", labelpad=1); axw.set_ylabel("$w$", labelpad=1)
axw.tick_params(length=1.5, pad=1)
axw.text(0.25, 0.12, "InD", fontsize=4.8, color=PROP, ha="center")
axw.text(2.25, 0.62, "OoD", fontsize=4.8, color="#a23b5e", ha="center")
text(x3 + 0.093, 0.462, "unreliable predictions earn less", fontsize=5.8, color=INK_2)

# deploy box
box(x3 + 0.170, 0.300, 0.112, 0.150, "white", HEAT, lw=0.8, r=0.010, z=2)
text(x3 + 0.226, 0.413, "Deploy", fontsize=6.6, fontweight="bold", color=HEAT)
text(x3 + 0.226, 0.375, "greedy policy", fontsize=5.4, color=INK_2)
text(x3 + 0.226, 0.348, "in FE simulator", fontsize=5.4, color=INK_2)
text(x3 + 0.226, 0.318, "(plain deviation)", fontsize=5.0, color=INK_3)
arrow(x3 + 0.150, 0.375, x3 + 0.168, 0.375, color=INK_3, lw=0.9, ms=6)
text(x3 + 0.150, 0.232, "online RL is affordable only with the surrogate", fontsize=6, color=INK_2)

# ---------------------------------------------------------------- bottom: why it is better
why = [("Physics-structured", "heating $\\leftrightarrow$ laser power,\ncooling $\\leftrightarrow$ dwell time", HEAT),
       ("Both uncertainty types", "epistemic (data coverage) +\naleatoric (noise), calibrated", COOL),
       ("Propagated, not summed", "through stages and layers\nvia the delta method", PROP),
       ("Enables online RL", "ms per transition vs.\n$\\approx$20 s per FE layer", RL)]
bw, gap = 0.235, 0.012
for j, (h1, h2, c) in enumerate(why):
    bx = 0.010 + j * (bw + gap)
    box(bx, 0.020, bw, 0.155, "white", c, lw=0.9, r=0.012)
    fig.add_artist(Rectangle((bx + 0.008, 0.045), 0.004, 0.105, transform=F, fc=c, ec="none"))
    text(bx + 0.022, 0.135, h1, ha="left", fontsize=7, fontweight="bold", color=c)
    text(bx + 0.022, 0.072, h2, ha="left", fontsize=5.9, color=INK_2, linespacing=1.25)

out = os.path.join(ROOT, "Document/figures/fig_overview")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=300)
print("saved", out)
