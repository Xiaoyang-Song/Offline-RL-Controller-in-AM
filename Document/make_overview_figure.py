"""
Document/make_overview_figure.py   (v4; v1 kept as make_overview_figure_v1.py)
Method overview figure: (1) the multilayer FE process, (2) the two-stage probabilistic digital twin
(increment prediction, uncertainty decomposition and propagation), (3) online learning in the twin,
and a footer strip with the four key properties.
Notation follows the paper: s_0 = ambient field, layer l maps s_{l-1} -> u_l (heating) -> s_l (cooling).
All insets are drawn from real data / results:
  * temperature fields   : test build 0 (as in fig_example_trajectory)
  * propagation ratio map: surrogate_model_v3/results_propagation_maps_narrow/propagation_maps.npz (layer 6)
  * multistep curves     : surrogate_model_v3/results_multistep/paper_fig_multistep_gap_matched.npz (narrow)

    python Document/make_overview_figure.py      -> Document/figures/fig_overview.{pdf,png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from surrogate_model_v3.dataset import load_trajectories, split_trajectories
from surrogate_model_v3.paper_figure_ood import PAPER_RC
from baselines.common.data_utils import load_mesh_nodes

# ------------------------------------------------------------------ palette (color only where it encodes meaning)
INK, INK2, INK3 = "#1f2329", "#4b505a", "#9a9ca5"
LINE = "#d4d5db"                                               # neutral frames
HEAT, HEAT_L, HEAT_XL = "#c4352d", "#f3c6bd", "#fbeae6"        # heating (paper red)
COOL, COOL_L, COOL_XL = "#3b55c4", "#c3cdf1", "#ebeefb"        # cooling (paper blue)
UNC, UNC_XL = "#1f7a70", "#eaf5f2"                             # propagated uncertainty (teal)
RL, RL_XL = "#6d4c8d", "#f3eef8"                               # control (plum)
GREY_XL = "#f4f4f6"
THERMAL = LinearSegmentedColormap.from_list(
    "thermal", ["#11163a", "#2c2f6e", "#5b3a8c", "#9a4a8c", "#d4626a", "#f08f4f", "#f8c86a", "#fdf3d8"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#2f5aa8", "#7fa6d6", "#f7f7f5", "#e79a86", "#b3362f"])
W_DOM, H_DOM = 12.0, 3.0

FS = 7.8                                                       # body font size (min ~6 everywhere)
plt.rcParams.update(PAPER_RC)
plt.rcParams.update({"font.size": FS, "xtick.labelsize": 6.9, "ytick.labelsize": 6.9, "axes.labelsize": 7.4})

FW, FH_TOP, FH_BOT = 7.2, 3.75, 0.70
root = plt.figure(figsize=(FW, FH_TOP + FH_BOT))
fig, foot = root.subfigures(2, 1, height_ratios=[FH_TOP, FH_BOT])
F = fig.transSubfigure
RND = root.canvas.get_renderer()


# ------------------------------------------------------------------ helpers (all in the coordinates of `sf`)
def box(x, y, w, h, fc, ec="none", lw=0.8, r=0.010, z=1, sf=None):
    sf = sf or fig
    sf.add_artist(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=sf.transSubfigure,
                                 fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(p0, p1, color=INK2, lw=0.9, ms=7, ls="-", z=6):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=F, arrowstyle="-|>", mutation_scale=ms, color=color,
                                   lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0))


def poly(xs_, ys_, color=INK3, lw=0.75, ls=(0, (2.2, 1.4))):
    fig.add_artist(mlines.Line2D(xs_, ys_, transform=F, color=color, lw=lw, ls=ls, zorder=5))


def text(x, y, s, sf=None, **kw):
    sf = sf or fig
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    kw.setdefault("zorder", 8); kw.setdefault("fontsize", FS)
    return sf.text(x, y, s, transform=sf.transSubfigure, **kw)


def segments(x, y, parts, fontsize=9.4):
    """(string, color) pieces side by side on one baseline, centered at x; returns their x-extents."""
    arts = [text(0, y, s, color=c, fontsize=fontsize, ha="left", va="baseline") for s, c in parts]
    widths = [a.get_window_extent(RND).width / fig.bbox.width for a in arts]
    x0 = x - sum(widths) / 2
    spans = []
    for a, w_ in zip(arts, widths):
        a.set_x(x0); spans.append((x0, x0 + w_)); x0 += w_
    return spans


def underline(span, y, label, color):
    x0, x1 = span
    fig.add_artist(mlines.Line2D([x0 + 0.002, x1 - 0.002], [y, y], transform=F, color=color, lw=0.9,
                                 solid_capstyle="round", zorder=8))
    text((x0 + x1) / 2, y - 0.020, label, color=color, fontsize=7.1)


def disc(x, y, radius_in, color):
    """A true circle of radius `radius_in` inches in the top subfigure (figure fractions are not isotropic)."""
    fig.add_artist(Ellipse((x, y), 2 * radius_in / FW, 2 * radius_in / FH_TOP, transform=F, fc=color, ec="none", zorder=6))


def panel(x, w, n, title, sub, color, xl):
    box(x, 0.020, w, 0.965, "white", LINE, lw=0.6, r=0.014, z=0)
    box(x, 0.900, w, 0.085, xl, "none", r=0.014, z=0.5)
    fig.add_artist(Rectangle((x, 0.900), w, 0.014, transform=F, fc=xl, ec="none", zorder=0.6))
    disc(x + 0.021, 0.943, 0.085, color)
    text(x + 0.021, 0.942, str(n), color="white", fontsize=9.0, fontweight="bold")
    text(x + 0.042, 0.955, title, ha="left", fontsize=10.1, fontweight="bold")
    text(x + 0.042, 0.922, sub, ha="left", fontsize=7.6, color=INK2, style="italic")


# ================================================================== panels
P1, P2, P3 = (0.008, 0.280), (0.300, 0.400), (0.712, 0.280)
panel(*P1, 1, "Multilayer LPBF process", "finite-element simulation (ground truth)", HEAT, HEAT_XL)
panel(*P2, 2, "Two-stage probabilistic digital twin", "learned increments with propagated uncertainty", UNC, UNC_XL)
panel(*P3, 3, "Online learning in the twin", "uncertainty-aware policy optimization", RL, RL_XL)
for xa in (P1[0] + P1[1], P2[0] + P2[1]):
    arrow((xa + 0.001, 0.50), (xa + 0.011, 0.50), color=INK3, lw=1.4, ms=9)

# ------------------------------------------------------------------ (1) process
_, _, test = split_trajectories(load_trajectories(os.path.join(ROOT, "Data/DatasetV2_layer_12_samples_5000.pkl")),
                                0.10, 0.10, 42)
traj = test[0]
nodes = load_mesh_nodes(os.path.join(ROOT, "surrogate_model/mesh.mat"))
tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
fracs = np.linspace(0.4, 0.5, 12)
VMIN, VMAX = 300.0, 5000.0
x1, wfield = P1[0] + 0.066, 0.196
hf = wfield * (H_DOM / W_DOM) * (FW / FH_TOP)

# governing equation (heat conduction), compact chip under the header
box(P1[0] + 0.014, 0.828, P1[1] - 0.028, 0.064, GREY_XL, "none", r=0.008, z=1)
text(P1[0] + P1[1] / 2, 0.874,
     "$\\rho c_p\\,\\partial T/\\partial t=\\nabla\\cdot(k\\nabla T)+q(a_l)$", fontsize=8.5)
text(P1[0] + P1[1] / 2, 0.842, "transient heat conduction", fontsize=7.0, color=INK2)


def field(y, values, layer, label, sub, col):
    ax = fig.add_axes([x1, y, wfield, hf])
    ax.tricontourf(tri, np.clip(values, VMIN, VMAX), levels=np.linspace(VMIN, VMAX, 40), cmap=THERMAL)
    side = min(W_DOM, H_DOM) * fracs[layer]
    ax.add_patch(Rectangle((W_DOM / 2 - side / 2, H_DOM / 2 - side / 2), side, side, fill=False,
                           ec="white", lw=0.7, ls=(0, (2, 1.2)), alpha=0.9))
    ax.set_xlim(0, W_DOM); ax.set_ylim(0, H_DOM); ax.axis("off")
    text(P1[0] + 0.032, y + hf * 0.62, label, fontsize=10.3, color=col)
    text(P1[0] + 0.032, y + hf * 0.20, sub, fontsize=6.9, color=col)


def step(y_top, y_bot, label, col):
    arrow((x1 + wfield / 2, y_top), (x1 + wfield / 2, y_bot), color=col, lw=0.9, ms=6)
    text(x1 + wfield / 2 + 0.008, (y_top + y_bot) / 2, label, ha="left", fontsize=7.1, color=col)


ys = [0.726, 0.592, 0.458, 0.268, 0.134]
field(ys[0], np.full(len(nodes), 300.0), 0, "$\\mathbf{s}_0$", "ambient", INK2)
field(ys[1], np.asarray(traj[0].u_heat_final).ravel(), 0, "$\\mathbf{u}_1$", "heated", HEAT)
field(ys[2], np.asarray(traj[0].u_final).ravel(), 0, "$\\mathbf{s}_1$", "cooled", COOL)
field(ys[3], np.asarray(traj[11].u_heat_final).ravel(), 11, "$\\mathbf{u}_{12}$", "heated", HEAT)
field(ys[4], np.asarray(traj[11].u_final).ravel(), 11, "$\\mathbf{s}_{12}$", "cooled", COOL)
step(ys[0] - 0.004, ys[1] + hf + 0.004, f"heating, {traj[0].lp_action:.0f} W", HEAT)
step(ys[1] - 0.004, ys[2] + hf + 0.004, "cooling time $c$", COOL)
ymid = (ys[2] + ys[3] + hf) / 2
text(x1 + wfield / 2, ymid + 0.010, "$\\vdots$", fontsize=11.5, color=INK3)
text(x1 + wfield / 2 + 0.012, ymid + 0.010, "heat accumulates", ha="left", fontsize=7.1, color=INK2)
text(x1 + wfield / 2 + 0.008, ys[3] + hf + 0.014, f"layer 12, {traj[11].lp_action:.0f} W",
     ha="left", fontsize=7.1, color=HEAT)
step(ys[3] - 0.004, ys[4] + hf + 0.004, "cooling time $c$", COOL)
cax = fig.add_axes([x1 + 0.012, 0.060, wfield - 0.024, 0.013])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(VMIN, VMAX), cmap=THERMAL), cax=cax, orientation="horizontal")
cb.set_ticks([300, 2000, 2800, 5000]); cb.ax.tick_params(labelsize=6.9, length=1.6, pad=1); cb.outline.set_linewidth(0.3)
text(P1[0] + 0.032, 0.067, "T [K]", fontsize=7.1, color=INK2)

# ------------------------------------------------------------------ (2) digital twin
x2, w2 = P2
yf = 0.800                                                   # flow line height
CW, CH = 0.096, 0.088


def ensemble(cx, title, out, args, col, light, xl):
    for k in (3, 2, 1):
        box(cx - CW / 2 + 0.0045 * k, yf - CH / 2 + 0.0045 * k, CW, CH, xl, light, lw=0.5, r=0.008, z=2)
    box(cx - CW / 2, yf - CH / 2, CW, CH, "white", col, lw=1.0, r=0.008, z=3)
    text(cx, yf + 0.020, title, fontsize=8.3, fontweight="bold", color=col)
    text(cx, yf - 0.015, f"$\\Delta^{{\\mathrm{{{out}}}}}{args}$", fontsize=8.7)
    return cx - CW / 2, cx + CW / 2


def plus(cx, col):
    fig.add_artist(Circle((cx, yf), 0.0115, transform=F, fc="white", ec=col, lw=1.0, zorder=7))
    text(cx, yf - 0.0008, "+", fontsize=9.8, color=col, fontweight="bold")


xs = x2 + 0.026
hx0, hx1 = ensemble(x2 + 0.098, "Heating", "h", "(\\mathbf{s}_{l-1}, a_l)", HEAT, HEAT_L, HEAT_XL)
px1 = x2 + 0.176
cx0, cx1 = ensemble(x2 + 0.274, "Cooling", "c", "(\\mathbf{u}_l, c)", COOL, COOL_L, COOL_XL)
px2 = x2 + 0.352
text(xs, yf, "$\\mathbf{s}_{l-1}$", fontsize=9.9)
arrow((xs + 0.017, yf), (hx0 - 0.002, yf))
arrow((hx1 + 0.001, yf), (px1 - 0.012, yf), color=HEAT)
plus(px1, HEAT)
text(px1 + 0.022, yf + 0.001, "$\\mathbf{u}_l$", fontsize=9.9)
arrow((px1 + 0.033, yf), (cx0 - 0.002, yf))
arrow((cx1 + 0.001, yf), (px2 - 0.012, yf), color=COOL)
plus(px2, COOL)
text(px2 + 0.022, yf + 0.001, "$\\mathbf{s}_{l}$", fontsize=9.9)
ytop = yf + 0.075                                            # identity (residual) paths above the cards
for x_from, x_to in ((xs, px1), (px1 + 0.022, px2)):
    poly([x_from, x_from, x_to], [yf + 0.020, ytop, ytop])
    arrow((x_to, ytop), (x_to, yf + 0.012), color=INK3, lw=0.75, ms=5)
ybot = yf - 0.066                                            # next layer, below the cards
poly([px2 + 0.022, px2 + 0.022, xs, xs], [yf - 0.018, ybot, ybot, yf - 0.026])
arrow((xs, ybot + 0.010), (xs, yf - 0.020), color=INK3, lw=0.75, ms=5)
text(x2 + w2 / 2, ybot - 0.020, "each stage predicts an increment $\\Delta$ added to its input",
     fontsize=7.1, color=INK2)

# predictive-uncertainty card (shading covers every label, with margin)
cy0, ch = 0.418, 0.272
box(x2 + 0.012, cy0, w2 - 0.024, ch, UNC_XL, "none", r=0.012, z=1)
text(x2 + 0.026, cy0 + ch - 0.024, "Predictive uncertainty", ha="left", fontsize=8.5, fontweight="bold")
text(x2 + 0.026, cy0 + ch - 0.054,
     "$\\mathbf{P}_l$: covariance after layer $l$;   "
     "$\\mathbf{A}=\\mathbf{I}+\\partial\\bar{\\boldsymbol{\\mu}}/\\partial\\mathbf{x}$: stage Jacobian",
     ha="left", fontsize=7.0, color=INK2)
sp = segments(x2 + w2 / 2, cy0 + 0.160, [
    ("$\\mathrm{Var}(\\Delta)\\;=\\;$", INK), ("$\\mathbb{E}_k[\\sigma_k^2]$", COOL),
    ("$\\;+\\;$", INK), ("$\\mathrm{Var}_k(\\mu_k)$", HEAT)])
underline(sp[1], cy0 + 0.139, "aleatoric", COOL)
underline(sp[3], cy0 + 0.139, "epistemic", HEAT)
sp = segments(x2 + w2 / 2, cy0 + 0.068, [
    ("$\\mathbf{P}_{l}\\;=\\;\\mathbf{A}^{\\mathrm{c}}($", INK),
    ("$\\mathbf{A}^{\\mathrm{h}}\\mathbf{P}_{l-1}\\mathbf{A}^{\\mathrm{h}\\top}$", UNC),
    ("$+$", INK), ("$\\boldsymbol{\\Sigma}^{\\mathrm{h}}$", HEAT),
    ("$)\\,\\mathbf{A}^{\\mathrm{c}\\top}+$", INK), ("$\\boldsymbol{\\Sigma}^{\\mathrm{c}}$", COOL)])
underline(sp[1], cy0 + 0.048, "propagated", UNC)
underline(sp[3], cy0 + 0.048, "heating", HEAT)
underline(sp[5], cy0 + 0.048, "cooling", COOL)

# insets (neutral frames, black titles)
iy0, ih = 0.035, 0.352
for ix, ttl in ((x2 + 0.012, "Within a layer"), (x2 + 0.206, "Across layers")):
    box(ix, iy0, 0.182, ih, "white", LINE, lw=0.7, r=0.010, z=1)
    text(ix + 0.091, iy0 + ih - 0.024, ttl, fontsize=8.3, fontweight="bold")
text(x2 + 0.103, iy0 + ih - 0.052, "redistributed by cooling", fontsize=7.2, color=INK2)
text(x2 + 0.297, iy0 + ih - 0.052, "uncertainty accumulates", fontsize=7.2, color=INK2)
for xc in (x2 + 0.103, x2 + 0.297):
    arrow((xc, cy0 - 0.002), (xc, iy0 + ih + 0.002), color=INK3, lw=0.75, ms=5, ls=(0, (2.2, 1.4)))

z = np.load(os.path.join(ROOT, "surrogate_model_v3/results_propagation_maps_narrow/propagation_maps.npz"))
t = 5
ratio = np.sqrt(z["all_prop_total"][t] / np.maximum(z["all_naive_total"][t], 1e-30))
lim = 1.6
axm = fig.add_axes([x2 + 0.024, iy0 + 0.118, 0.158, 0.165])
axm.tricontourf(tri, np.clip(ratio, 1 / lim, lim), levels=np.geomspace(1 / lim, lim, 31), cmap=DIVERGE,
                norm=TwoSlopeNorm(vcenter=1.0, vmin=1 / lim, vmax=lim))
side = min(W_DOM, H_DOM) * fracs[t]
axm.add_patch(Rectangle((W_DOM / 2 - side / 2, H_DOM / 2 - side / 2), side, side, fill=False, ec=INK, lw=0.7, ls="--"))
axm.set_xlim(3.0, 9.0); axm.set_ylim(0, H_DOM); axm.set_aspect("equal"); axm.axis("off")
cax = fig.add_axes([x2 + 0.040, iy0 + 0.090, 0.126, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vcenter=1.0, vmin=1 / lim, vmax=lim), cmap=DIVERGE),
                  cax=cax, orientation="horizontal")
cb.set_ticks([1 / lim, 1.0, lim]); cb.set_ticklabels(["0.6", "1", "1.6"])
cb.ax.tick_params(labelsize=6.9, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
text(x2 + 0.103, iy0 + 0.040, "propagated / summed $\\sigma$", fontsize=6.9, color=INK2)

res = dict(np.load(os.path.join(ROOT, "surrogate_model_v3/results_multistep/paper_fig_multistep_gap_matched.npz"),
                   allow_pickle=True)["results"])
r = res["Narrow (200–300 W)"]
L = r["layers"]
axc = fig.add_axes([x2 + 0.238, iy0 + 0.068, 0.140, 0.205])
lo, hi = np.percentile(r["prop_traj"], [10, 90], axis=0)
axc.fill_between(L, lo, hi, color=UNC, alpha=0.14, lw=0)
axc.plot(L, r["rmse"], color=INK, lw=1.2, label="prediction error")
axc.plot(L, r["prop"], color=UNC, lw=1.5, label="propagated $\\sigma$")
axc.plot(L, r["one"], color=INK3, lw=1.1, ls=(0, (3, 1.5)), label="single-step $\\sigma$")
axc.set_xticks([1, 6, 12]); axc.set_yticks([0, 10, 20]); axc.set_xlim(0.5, 12.5); axc.set_ylim(0, 30)
axc.set_xlabel("layer", labelpad=1); axc.set_ylabel("[K]", labelpad=1)
axc.tick_params(length=1.6, pad=1.5)
for s_ in ("top", "right"):
    axc.spines[s_].set_visible(False)
axc.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=1.5, borderaxespad=0.1, labelspacing=0.3)

# ------------------------------------------------------------------ (3) online learning
x3, w3 = P3
yl = 0.800
box(x3 + 0.010, yl - 0.042, 0.106, 0.084, RL_XL, RL, lw=1.0, r=0.010, z=2)
text(x3 + 0.063, yl + 0.014, "Policy", fontsize=8.7, fontweight="bold", color=RL)
text(x3 + 0.063, yl - 0.016, "any RL algorithm", fontsize=6.9, color=INK2)
box(x3 + 0.164, yl - 0.042, 0.106, 0.084, UNC_XL, UNC, lw=1.0, r=0.010, z=2)
text(x3 + 0.217, yl + 0.014, "Digital twin", fontsize=8.7, fontweight="bold", color=UNC)
text(x3 + 0.217, yl - 0.016, "field + uncertainty", fontsize=6.9, color=INK2)
arrow((x3 + 0.118, yl + 0.012), (x3 + 0.162, yl + 0.012), color=RL, lw=1.0)
text(x3 + 0.140, yl + 0.058, "action", fontsize=7.1, color=RL)
arrow((x3 + 0.162, yl - 0.012), (x3 + 0.118, yl - 0.012), color=UNC, lw=1.0)
text(x3 + 0.140, yl - 0.060, "state, reward", fontsize=6.9, color=UNC)


def contrast(y, h, title, lines, ok, col, fc, tcol):
    box(x3 + 0.012, y, w3 - 0.024, h, fc, "none", r=0.009, z=1)
    disc(x3 + 0.028, y + h - 0.024, 0.062, col)
    text(x3 + 0.028, y + h - 0.0245, "$\\checkmark$" if ok else "$\\times$", color="white", fontsize=8.0)
    text(x3 + 0.044, y + h - 0.024, title, ha="left", fontsize=8.0, fontweight="bold", color=tcol)
    for k, ln in enumerate(lines):
        text(x3 + 0.044, y + h - 0.054 - 0.026 * k, ln, ha="left", fontsize=7.2, color=INK2)


contrast(0.578, 0.108, "Offline RL", ["confined to logged data;", "cannot test unseen actions"], False, INK3, GREY_XL, INK2)
contrast(0.452, 0.108, "Online RL in the twin", ["unlimited interaction in ms;", "unreliable predictions discounted"],
         True, RL, RL_XL, RL)

text(x3 + w3 / 2, 0.408, "node reward  =  trust $w$  $\\times$  deviation score", fontsize=7.6)
axw = fig.add_axes([x3 + 0.046, 0.140, 0.104, 0.200])
sr = np.linspace(0, 3, 200)
axw.axvspan(0, 0.6, color=UNC_XL, lw=0)
axw.axvspan(1.5, 3, color="#f6e6ea", lw=0)
axw.plot(sr, 1 / (1 + sr ** 2), color=RL, lw=1.5)
axw.set_xlim(0, 3); axw.set_ylim(0, 1.05); axw.set_xticks([0, 1, 2, 3]); axw.set_yticks([0, 0.5, 1])
axw.set_xlabel("uncertainty $\\sigma/\\tau$", labelpad=1); axw.set_ylabel("trust $w$", labelpad=1)
axw.tick_params(length=1.6, pad=1.5)
for s_ in ("top", "right"):
    axw.spines[s_].set_visible(False)
axw.text(0.3, 0.12, "trusted", fontsize=6.8, color=UNC, ha="center", rotation=90, va="bottom")
axw.text(2.25, 0.62, "discounted", fontsize=6.8, color="#a23b5e", ha="center")
box(x3 + 0.166, 0.170, 0.106, 0.150, "white", HEAT, lw=0.9, r=0.010, z=2)
text(x3 + 0.219, 0.284, "Deploy", fontsize=8.3, fontweight="bold", color=HEAT)
text(x3 + 0.219, 0.246, "trained policy", fontsize=7.1, color=INK2)
text(x3 + 0.219, 0.218, "real-time control", fontsize=7.1, color=INK2)
arrow((x3 + 0.154, 0.245), (x3 + 0.164, 0.245), color=INK3, lw=0.9, ms=6)
text(x3 + w3 / 2, 0.055, "the twin makes online interaction affordable", fontsize=7.2, color=INK2, style="italic")

# ------------------------------------------------------------------ footer: key properties
props = [("Physics-structured digital twin", HEAT, HEAT_XL,
          "separate heating and cooling stages\nfor laser power and cooling time"),
         ("Two sources of uncertainty", COOL, COOL_XL,
          "epistemic flags gaps in the data;\naleatoric captures process noise"),
         ("Dynamics-aware propagation", UNC, UNC_XL,
          "Jacobian propagation across stages\nand layers tracks error growth"),
         ("Uncertainty-aware learning", RL, RL_XL,
          "$\\approx$6 ms vs. $\\approx$20 s per FE layer;\nunreliable predictions discounted")]
FT = foot.transSubfigure
fw_, gap_ = 0.2375, 0.012
for j, (h1, col, xl, h2) in enumerate(props):
    fx = 0.008 + j * (fw_ + gap_)
    box(fx, 0.08, fw_, 0.84, xl, "none", r=0.012, z=1, sf=foot)
    foot.add_artist(FancyBboxPatch((fx + 0.007, 0.22), 0.0035, 0.56, boxstyle="round,pad=0,rounding_size=0.0017",
                                   transform=FT, fc=col, ec="none", zorder=2))
    text(fx + 0.022, 0.71, h1, sf=foot, ha="left", fontsize=7.8, fontweight="bold", color=col)
    text(fx + 0.022, 0.35, h2, sf=foot, ha="left", fontsize=7.0, color=INK2, linespacing=1.3)

for sf in (fig, foot):
    for ax_ in sf.axes:                                       # keep every inset above the card backgrounds
        ax_.set_zorder(5)
out = os.path.join(ROOT, "Document/figures/fig_overview")
for ext in ("pdf", "png"):
    root.savefig(f"{out}.{ext}", dpi=300)
print("saved", out)
