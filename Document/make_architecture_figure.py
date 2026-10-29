"""
Document/make_architecture_figure.py
Architecture of the two-stage probabilistic ensemble (surrogate_model_v3/model.py), one layer:
  s_{l-1} -(heating ensemble | a_l, e(l))-> + -> u_l -(cooling ensemble | c, e(l))-> + -> s_l.
Each stage is an ensemble of K MLP members (drawn as a generic network) trained on bootstrap resamples; each member outputs a
diagonal Gaussian over the increment, which is added to the stage input (identity path).
Field maps are real (full_range checkpoint, one test build, layer 6) from
Document/figure_data/compute_architecture_fields.py; the embedding vector is the learned e(l) of layer 6.
Text is typeset with LaTeX (Computer Modern, as in the paper).

    python Document/make_architecture_figure.py   -> Document/figures/fig_architecture.{pdf,png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from baselines.common.data_utils import load_mesh_nodes

# ------------------------------------------------------------------ palette: neutral frame, muted accents
INK, INK2, INK3 = "#1f2329", "#1f2329", "#9aa1ab"            # all text black; INK3 only for thin lines
LINE, LINE2 = "#d9dbe0", "#c3c6cd"
FILL = "#f7f7f9"
SLATE = "#34495e"
LASER = "#b8323f"                                               # laser power (crimson)
CTIME = "#7d5ba6"                                               # cooling time (lavender)
EMB_C, EMB_L = "#2a9d8f", "#cdeae6"                           # layer embedding (teal green)
NET_COLS = ["#2a9d8f", "#3f7cac", "#3f7cac", "#34495e"]          # network icon: input -> hidden -> output
HEAT_T, HEAT_B, HEAT_XL = INK, "#e6b3b8", "#f9e3e5"              # heating: title / border / background
COOL_T, COOL_B, COOL_XL = INK, "#bccdea", "#e7eef9"              # cooling: title / border / background
THERMAL = LinearSegmentedColormap.from_list(
    "thermal", ["#11163a", "#2c2f6e", "#5b3a8c", "#9a4a8c", "#d4626a", "#f08f4f", "#f8c86a", "#fdf3d8"])

plt.rcParams.update({
    "text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
FW, FH = 7.2, 2.0
fig = plt.figure(figsize=(FW, FH))
F = fig.transFigure
FS = 8.5

# ------------------------------------------------------------------ data
Z = np.load(os.path.join(ROOT, "Document/figure_data/architecture_fields.npz"))
nodes = load_mesh_nodes(os.path.join(ROOT, "surrogate_model/mesh.mat"))
tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
X0, X1, H_DOM = 3.0, 9.0, 3.0                                       # crop around the heated region
VMIN, VMAX = 300.0, 5000.0
ck = torch.load(os.path.join(ROOT, "surrogate_model_v3/runs/full_range/surrogate_best.pt"), map_location="cpu",
                weights_only=False)["model_state_dict"]
EMB = ck[[k for k in ck if k.startswith("heating_transitions.0.") and "layer_embed" in k][0]][5].numpy()


# ------------------------------------------------------------------ helpers (figure coordinates)
def box(x, y, w, h, fc, ec="none", lw=0.8, r=0.010, z=1):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=F,
                                  fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(p0, p1, color=INK2, lw=0.9, ms=7, z=6, rad=0.0):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=F, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                   zorder=z, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def poly(xs_, ys_, color=INK3, lw=0.8, ls="-", z=5):
    fig.add_artist(mlines.Line2D(xs_, ys_, transform=F, color=color, lw=lw, ls=ls, zorder=z,
                                 solid_capstyle="butt"))


def text(x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    kw.setdefault("zorder", 9); kw.setdefault("fontsize", FS)
    return fig.text(x, y, s, transform=F, **kw)


def disc(x, y, r_in, fc, ec=None, lw=0.8, z=7):
    fig.add_artist(Ellipse((x, y), 2 * r_in / FW, 2 * r_in / FH, transform=F, fc=fc, ec=ec or fc, lw=lw, zorder=z))


def hmap(x, y, w, values):
    h = w * (H_DOM / (X1 - X0)) * FW / FH
    ax = fig.add_axes([x, y, w, h]); ax.set_zorder(5)
    ax.tricontourf(tri, np.clip(values, VMIN, VMAX), levels=np.linspace(VMIN, VMAX, 40), cmap=THERMAL)
    ax.set_xlim(X0, X1); ax.set_ylim(0, H_DOM); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return h


def net_icon(cx, cy, w=0.056, h=0.190, sizes=(3, 4, 4, 2)):
    """Generic fully-connected network: node columns with all connections."""
    xs_ = np.linspace(cx - w / 2, cx + w / 2, len(sizes))
    pts = [[(xx, cy + (j - (n - 1) / 2) * h / 3) for j in range(n)] for xx, n in zip(xs_, sizes)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                poly([p[0], q[0]], [p[1], q[1]], color="#b8c2cc", lw=0.45, z=8)
    for c_, col in zip(pts, NET_COLS):
        for p in c_:
            disc(p[0], p[1], 0.026, col, "white", lw=0.5, z=9)


# ------------------------------------------------------------------ layout
ya = 0.560                                         # main row
wf = 0.100
hf = wf * (H_DOM / (X1 - X0)) * FW / FH
XS0, XC1, XP1, XU, XC2, XP2, XS1 = 0.008, 0.122, 0.418, 0.434, 0.550, 0.846, 0.862
CW = 0.280                                         # stage card width
MX = 0.124                                         # member stack offset inside the card


def field(x, values, label, sub):
    hmap(x, ya - hf / 2, wf, values)
    text(x + wf / 2, ya + hf / 2 + 0.085, label, fontsize=10)
    for j, line in enumerate(sub.split("|")):
        text(x + wf / 2, ya - hf / 2 - 0.065 - 0.062 * j, line, fontsize=8.0, color=INK2)


def plus(x):
    disc(x, ya, 0.068, "white", INK, lw=0.9)
    poly([x - 0.0055, x + 0.0055], [ya, ya], color=INK, lw=0.9, z=8)
    poly([x, x], [ya - 0.0055 * FW / FH, ya + 0.0055 * FW / FH], color=INK, lw=0.9, z=8)


def stage(x0, tcol, bcol, tint, title, cond, ccol, delta, fin):
    y0, h = 0.080, 0.870
    box(x0, y0, CW, h, tint, bcol, lw=0.8, r=0.012, z=1)
    text(x0 + 0.014, y0 + h - 0.075, r"\textbf{%s}" % title, ha="left", fontsize=9.6, color=tcol)
    xbar, wbar = x0 + 0.097, 0.014                           # concatenated input vector
    seg = {"cond": (0.625, 0.660), "field": (0.395, 0.618), "emb": (0.270, 0.388)}
    mx0, my0 = x0 + MX, 0.265

    # condition: slider
    xi, wi = x0 + 0.016, 0.054
    ys = 0.705
    text(xi - 0.002, ys + 0.080, cond, fontsize=7.6, color=INK2, ha="left")
    poly([xi, xi + wi], [ys, ys], color="white", lw=2.8, z=6)
    poly([xi, xi + wi * 0.55], [ys, ys], color=ccol, lw=2.8, z=6)
    disc(xi + wi * 0.55, ys, 0.036, "white", ccol, lw=1.0, z=7)
    arrow((xi + wi + 0.004, ys), (xbar - 0.001, sum(seg["cond"]) / 2), color=INK3, lw=0.7, ms=4.5)

    # layer index l -> learned embedding e(l) (vertical vector)
    ev = (EMB - EMB.min()) / (EMB.max() - EMB.min())
    cm_ = LinearSegmentedColormap.from_list("e", ["#ffffff", EMB_L, EMB_C])
    ch, cwid = 0.034, 0.013
    xe, ye = x0 + 0.054, 0.190
    for j, v in enumerate(ev[::-1]):
        fig.add_artist(Rectangle((xe, ye + j * ch), cwid, ch, transform=F, fc=cm_(0.10 + 0.90 * v),
                                 ec=EMB_L, lw=0.5, zorder=6))
    fig.add_artist(Rectangle((xe, ye), cwid, 8 * ch, transform=F, fc="none", ec=EMB_L, lw=0.5, zorder=6))
    yem = ye + 4 * ch
    box(xi, yem - 0.050, 0.020, 0.100, "white", EMB_C, lw=0.8, r=0.004, z=6)
    text(xi + 0.010, yem, r"$l$", fontsize=8.5, color=INK)
    arrow((xi + 0.021, yem), (xe - 0.002, yem), color=INK3, lw=0.7, ms=4)
    text(xi + 0.010, yem - 0.085, "layer", fontsize=7.6, color=INK2)
    text(xe + cwid / 2, ye - 0.050, r"$e(l)$", fontsize=8.0, color=INK2)
    arrow((xe + cwid + 0.003, yem), (xbar - 0.001, sum(seg["emb"]) / 2), color=INK3, lw=0.7, ms=4.5)

    # concatenated vector with three regions: [condition | flattened field | embedding]
    fig.add_artist(Rectangle((xbar, seg["cond"][0]), wbar, seg["cond"][1] - seg["cond"][0], transform=F, fc=ccol,
                             ec="none", zorder=6))
    order = np.lexsort((nodes[:, 1], nodes[:, 0]))          # flatten the mesh column by column
    flat = fin[order][:: max(1, len(fin) // 48)]
    flat = (np.clip(flat, VMIN, VMAX) - VMIN) / (VMAX - VMIN)  # same temperature scale as the field maps
    yy = np.linspace(seg["field"][1], seg["field"][0], len(flat) + 1)
    for v, ya_, yb_ in zip(flat, yy[:-1], yy[1:]):
        fig.add_artist(Rectangle((xbar, yb_), wbar, ya_ - yb_, transform=F, fc=THERMAL(v),
                                 ec="none", zorder=6))
    yy = np.linspace(seg["emb"][1], seg["emb"][0], 9)
    for v, ya_, yb_ in zip(ev, yy[:-1], yy[1:]):
        fig.add_artist(Rectangle((xbar, yb_), wbar, ya_ - yb_, transform=F, fc=cm_(0.10 + 0.90 * v), ec="none",
                                 zorder=6))
    for a_, b_ in seg.values():
        fig.add_artist(Rectangle((xbar, a_), wbar, b_ - a_, transform=F, fc="none", ec=INK2, lw=0.6, zorder=7))
    text(xbar + wbar / 2, seg["emb"][0] - 0.045, "concat.", fontsize=7.2, color=INK2)
    text(xi + 0.030, ya + 0.040, "flatten", fontsize=7.6, color=INK2)
    arrow((xbar + wbar + 0.001, ya), (mx0 - 0.002, ya), color=INK, lw=0.9, ms=6)

    # K member cards (neutral), front card carries the network icon
    K, dx, dy = 5, 0.0055, 0.030
    mw, mh = 0.090, 0.420
    for k in range(K - 1, -1, -1):
        front = k == 0
        box(mx0 + dx * k, my0 + dy * k, mw, mh, "white" if front else FILL, INK2 if front else LINE2,
            lw=0.9 if front else 0.6, r=0.007, z=2 + (K - k))
    cxm = mx0 + mw / 2
    text(cxm, my0 + mh - 0.062, r"member $k$", fontsize=7.8)
    net_icon(cxm, my0 + mh / 2 - 0.035)
    text(mx0 + dx * (K - 1) + mw + 0.005, my0 + dy * (K - 1) + mh - 0.010, r"$\times K$", fontsize=8.2,
         ha="left", va="top")
    text(cxm + 0.011, 0.170, "bootstrap", fontsize=7.6, color=INK2)
    text(cxm + 0.011, 0.125, "resamples", fontsize=7.6, color=INK2)

    # members -> increment
    xo = mx0 + dx * (K - 1) + mw
    arrow((xo + 0.002, ya), (x0 + CW + 0.001, ya), color=INK)
    text((xo + x0 + CW) / 2 + 0.003, ya + 0.090, delta, fontsize=10)


stage(XC1, HEAT_T, HEAT_B, HEAT_XL, "Heating stage", r"laser power $a_l$", LASER, r"$\hat{\bm{\Delta}}^{h}_{l}$", Z["id_s_prev"])
stage(XC2, COOL_T, COOL_B, COOL_XL, "Cooling stage", r"cooling time $c$", CTIME, r"$\hat{\bm{\Delta}}^{c}_{l}$", Z["id_s_prev"] + Z["id_heat_mu"].mean(0))

u_hat = Z["id_s_prev"] + Z["id_heat_mu"].mean(0)
s_hat = u_hat + Z["id_cool_mu"].mean(0)
field(XS0, Z["id_s_prev"], r"$\bm{s}_{l-1}$", "layer $l-1$")
field(XU, u_hat, r"$\hat{\bm{u}}_{l}$", "End of Heating|(EoH)")
field(XS1, s_hat, r"$\hat{\bm{s}}_{l}$", "End of Cooling|(EoC)")
plus(XP1); plus(XP2)

# field -> stage (enters the member stack); stage output -> next field
for xs, xc in ((XS0, XC1), (XU, XC2)):
    arrow((xs + wf + 0.002, ya), (xc + 0.097 - 0.001, ya), color=INK)
arrow((XP1 + 0.0095, ya), (XU - 0.002, ya), color=INK)
arrow((XP2 + 0.0095, ya), (XS1 - 0.002, ya), color=INK)

# identity (residual) paths along the bottom
ysk = 0.030
for xa, xb in ((XS0 + wf / 2, XP1), (XU + wf / 2, XP2)):
    poly([xa, xa, xb, xb], [ya - hf / 2 - 0.215, ysk, ysk, ya - 0.070], color=INK3, lw=0.8, ls=(0, (2.4, 1.6)))
    arrow((xb, ya - 0.074), (xb, ya - 0.034), color=INK3, lw=0.8, ms=5)
    text((xa + xb) / 2, ysk, "identity", fontsize=7.6, color=INK2, bbox=dict(fc="white", ec="none", pad=3.0))

out = os.path.join(ROOT, "Document/figures/fig_architecture")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=300)
print("saved", out)
