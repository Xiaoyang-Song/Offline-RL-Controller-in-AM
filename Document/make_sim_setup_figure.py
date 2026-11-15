"""
Document/make_sim_setup_figure.py
Simulation setup figure for the paper:
  (a) schematic of the multilayer build (cabinet projection; layer thickness exaggerated): build plate, powder bed,
      printed part whose square scan region grows from 1.2 mm to 1.5 mm over 12 layers, laser on the current layer;
  (b) zoom on the current layer: s_l --heating (laser on, a_l)--> u_l (EoH) --cooling (laser off, c_l)--> s_{l+1},
      drawn on the FE mesh (no numerical values shown).
Fields are real (test build 0 of DatasetV2, layer 6). Geometry from LPBF-Simulation/simulation_v2 (W = 12, H = 3,
scan square side 0.4 -> 0.5 * min(W, H)).

    python Document/make_sim_setup_figure.py   -> Document/figures/fig_sim_setup.{pdf,png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from surrogate_model_v3.dataset import load_trajectories, split_trajectories
from baselines.common.data_utils import load_mesh_nodes

INK, INK3 = "#1f2329", "#9aa1ab"
LINE2 = "#c3c6cd"
HEAT, HEAT_B, HEAT_XL = "#c4352d", "#e6b3b8", "#f9e3e5"
COOL, COOL_B, COOL_XL = "#3b55c4", "#bccdea", "#e7eef9"
POWDER, POWDER_E = "#efe9dc", "#d8cfbb"
PLATE_T, PLATE_F, PLATE_S = "#9aa3ad", "#7d8792", "#68717c"
METAL_T, METAL_F, METAL_S = "#b9c0c9", "#8f98a4", "#a3abb6"
THERMAL = LinearSegmentedColormap.from_list(
    "thermal", ["#11163a", "#2c2f6e", "#5b3a8c", "#9a4a8c", "#d4626a", "#f08f4f", "#f8c86a", "#fdf3d8"])

plt.rcParams.update({
    "text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "savefig.dpi": 300, "savefig.pad_inches": 0.02,
})
FW, FH = 7.2, 1.55
fig = plt.figure(figsize=(FW, FH))
F = fig.transFigure
FS = 8.0

# ------------------------------------------------------------------ data
W_DOM, H_DOM, L = 12.0, 3.0, 12
LAYER = 6                                                        # 1-based layer shown in (b), (c)
_, _, test = split_trajectories(load_trajectories(os.path.join(ROOT, "Data/DatasetV2_layer_12_samples_5000.pkl")),
                                0.10, 0.10, 42)
traj = test[0]
s_l = traj[LAYER - 2].u_final[:, 0]
u_l = traj[LAYER - 1].u_heat_final[:, 0]
s_n = traj[LAYER - 1].u_final[:, 0]
a_l, c_b = traj[LAYER - 1].lp_action, traj[LAYER - 1].cool_time
nodes = load_mesh_nodes(os.path.join(ROOT, "surrogate_model/mesh.mat"))
tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
VMIN, VMAX = 300.0, float(np.ceil(u_l.max() / 500) * 500)
side = lambda l: min(W_DOM, H_DOM) * (0.4 + 0.1 * (l - 1) / (L - 1))   # scan square side, mm


# ------------------------------------------------------------------ helpers
def box(x, y, w, h, fc, ec="none", lw=0.8, r=0.012, z=0):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=F,
                                  fc=fc, ec=ec, lw=lw, zorder=z))


def text(x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
    kw.setdefault("zorder", 20); kw.setdefault("fontsize", FS)
    return fig.text(x, y, s, transform=F, **kw)


def farrow(p0, p1, color=INK, lw=0.9, ms=7, z=15):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=F, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                   zorder=z, shrinkA=0, shrinkB=0))


def panel(x, y, w, h, label, title):
    box(x, y, w, h, "white", LINE2, lw=0.6)
    text(x + 0.012, y + h - 0.15 / FH, r"\textbf{(%s)}\ \ \textbf{%s}" % (label, title), ha="left", fontsize=9.8)


def fieldmap(rect, values, xlim=(0, W_DOM), mesh=False, square=None, ec="white"):
    ax = fig.add_axes(rect); ax.set_zorder(5)
    ax.tricontourf(tri, np.clip(values, VMIN, VMAX), levels=np.linspace(VMIN, VMAX, 48), cmap=THERMAL)
    if mesh:
        ax.triplot(tri, color="white", lw=0.12, alpha=0.35)
    if square is not None:
        ax.add_patch(Rectangle((W_DOM / 2 - square / 2, H_DOM / 2 - square / 2), square, square, fill=False,
                               ec="white", lw=0.8, ls=(0, (2.2, 1.4))))
    ax.set_xlim(*xlim); ax.set_ylim(0, H_DOM); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ec); sp.set_linewidth(0.6)
    return ax


# ================================================================== (a) build schematic (cabinet projection)
panel(0.008, 0.025, 0.376, 0.950, "a", "Multilayer build schematic")
axa = fig.add_axes([0.010, 0.025, 0.372, 0.790]); axa.axis("off")
KX, KY = 0.62, 0.50                                              # receding (depth) axis
HZ = 0.30                                                        # drawn layer thickness (exaggerated)
P = lambda x, y, z: (x + KX * y, z + KY * y)


def face(pts, fc, ec="none", lw=0.5, alpha=1.0, z=1):
    axa.add_patch(Polygon([P(*p) for p in pts], closed=True, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z,
                          joinstyle="round"))


def cuboid(x0, x1, y0, y1, z0, z1, top, front, right, ec="none", lw=0.5, alpha=1.0, z=1):
    face([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], top, ec, lw, alpha, z)
    face([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], front, ec, lw, alpha, z)
    face([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], right, ec, lw, alpha, z)


Hb = L * HZ
face([(-0.2, -0.5, -0.75), (W_DOM + 0.9, -0.5, -0.75), (W_DOM + 0.9, H_DOM + 0.6, -0.75), (-0.2, H_DOM + 0.6, -0.75)],
     "#d9dce1", alpha=0.6, z=0)                                                                   # soft shadow
cuboid(-0.4, W_DOM + 0.4, -0.3, H_DOM + 0.3, -0.6, 0.0, PLATE_T, PLATE_F, PLATE_S, z=1)          # build plate
cuboid(0, W_DOM, 0, H_DOM, 0, Hb, POWDER, "#f3eee3", "#e6dece", ec=POWDER_E, lw=0.5, z=2)       # powder bed
_rng = np.random.default_rng(1)                                   # powder texture on the top surface
_px, _py = _rng.uniform(0, W_DOM, 900), _rng.uniform(0, H_DOM, 900)
_keep = ~((np.abs(_px - W_DOM / 2) < side(L) / 2 + 0.15) & (np.abs(_py - H_DOM / 2) < side(L) / 2 + 0.15))
_pp = np.array([P(x_, y_, Hb) for x_, y_ in zip(_px[_keep], _py[_keep])])
axa.scatter(_pp[:, 0], _pp[:, 1], s=0.25, color="#c9bda3", lw=0, zorder=3.5)
for l in range(1, L):                                             # powder layer lines
    axa.plot(*zip(P(0, 0, l * HZ), P(W_DOM, 0, l * HZ), P(W_DOM, H_DOM, l * HZ)), color=POWDER_E, lw=0.3,
             zorder=3)
for l in range(1, L + 1):                                         # printed part (cut-away view), layer by layer
    sd = side(l); x0, y0 = W_DOM / 2 - sd / 2, H_DOM / 2 - sd / 2
    cur = l == L
    cuboid(x0, x0 + sd, y0, y0 + sd, (l - 1) * HZ, l * HZ,
           "#f8c27f" if cur else METAL_T, "#e07b39" if cur else METAL_F, "#eb9a5a" if cur else METAL_S,
           ec="white", lw=0.3, z=5 + l)
sd = side(L); x0c, y0c = W_DOM / 2 - sd / 2, H_DOM / 2 - sd / 2
cx, cy = P(W_DOM / 2, H_DOM / 2, Hb)
for k_, (wtop, al) in enumerate(((0.75, 0.18), (0.55, 0.35), (0.32, 0.75))):      # graded beam
    axa.add_patch(Polygon([(cx - wtop, cy + 2.4), (cx + wtop, cy + 2.4), (cx + 0.06, cy), (cx - 0.06, cy)],
                          fc=HEAT, ec="none", alpha=al, zorder=33 + 0.1 * k_))
for r_, al in ((0.75, 0.12), (0.5, 0.2), (0.3, 0.35)):                               # melt-spot glow
    axa.add_patch(matplotlib.patches.Ellipse((cx, cy), 2 * r_, 2 * r_ * 0.55, fc="#f8c86a", ec="none", alpha=al,
                                             zorder=34.5))
axa.add_patch(FancyBboxPatch((cx - 0.85, cy + 2.4), 1.7, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                             fc="#3a3f48", ec="none", zorder=34))
axa.plot([cx], [cy], marker="o", ms=3.5, mfc="#fdf3d8", mec="#f08f4f", mew=0.8, zorder=35)
# build direction (left of the bed) and dimensions
bx, by = P(-1.1, 0, 0)
axa.annotate("", xy=(bx, by + Hb + 0.5), xytext=(bx, by), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
axa.text(bx - 0.75, by + (Hb + 0.5) / 2, "build direction", fontsize=8.0, rotation=90, ha="center", va="center")
p0, p1 = P(0, 0, -1.0), P(W_DOM, 0, -1.0)
axa.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
axa.text((p0[0] + p1[0]) / 2, p0[1] - 0.12, "12 mm", fontsize=8.0, ha="center", va="top")
q0, q1 = P(W_DOM + 0.8, -0.3, -0.6), P(W_DOM + 0.8, H_DOM + 0.3, -0.6)
axa.annotate("", xy=q1, xytext=q0, arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
axa.text(q1[0] + 0.25, (q0[1] + q1[1]) / 2 - 0.1, "3 mm", fontsize=8.0, ha="left", va="center")
# object labels: color-coded, placed next to their objects on both sides
XR = P(W_DOM, H_DOM, 0)[0] + 0.5
for txt, xy, xyt, col, ha in (
        ("laser", (cx - 0.25, cy + 1.9), (cx - 1.9, cy + 2.85), HEAT, "right"),
        ("powder", P(2.4, 2.0, Hb), (P(0, 0, 0)[0] + 0.1, cy + 1.55), "#9a8556", "left"),
        (r"current layer $l$", P(x0c + sd, y0c + sd / 2, Hb - HZ / 2), (XR, cy + 1.2), "#d0682a", "left"),
        ("solidified layers", P(x0c + sd, y0c + sd / 2, Hb * 0.4), (XR, cy - 1.2), "#4f5b69", "left")):
    axa.annotate(txt, xy=xy, xytext=xyt, fontsize=8.2, color=col, va="center", ha=ha,
                 arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=2, shrinkB=0, alpha=0.9),
                 zorder=45)
    axa.plot(*xy, marker="o", ms=2.2, color=col, zorder=46)
axa.text(*P(W_DOM / 2, 0, -0.32), "build plate", fontsize=7.2, ha="center", va="center", color="white", zorder=40)
axa.set_xlim(-2.3, W_DOM + KX * H_DOM + 7.5); axa.set_ylim(-1.6, Hb + KY * H_DOM + 2.4); axa.set_aspect("equal")
zx0, zy0 = P(x0c - 0.3, y0c, Hb - HZ - 0.12)
zx1, zy1 = P(x0c + sd + 0.3, y0c + sd, Hb + 0.15)
axa.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=INK, lw=0.7, ls=(0, (2.2, 1.4)), zorder=41))
axa.text(zx0 - 0.15, (zy0 + zy1) / 2, r"\textbf{(b)}", fontsize=8.0, ha="right", va="center", zorder=41)

# ================================================================== (b) one layer: heating and cooling phases
panel(0.394, 0.025, 0.598, 0.950, "b", r"Simulated heating and cooling of layer $l$")
wm = 0.138
hm = wm * (H_DOM / 5.0) * FW / FH
ym = 0.165
xs = [0.410, 0.626, 0.842]
for x in xs:                                                             # soft drop shadows
    fig.add_artist(Rectangle((x + 0.002, ym - 0.012), wm, hm, transform=F, fc="#000000", alpha=0.08, ec="none",
                             zorder=4))
for x, v, lab_, sub in ((xs[0], s_l, r"$\bm{s}_l$", "start of layer"),
                        (xs[1], u_l, r"$\bm{u}_l$", "end of heating (EoH)"),
                        (xs[2], s_n, r"$\bm{s}_{l+1}$", "end of cooling (EoC)")):
    fieldmap([x, ym, wm, hm], v, xlim=(3.5, 8.5), mesh=True, square=side(LAYER))
    text(x + wm / 2, ym + hm + 0.095, lab_, fontsize=11)
    text(x + wm / 2, ym - 0.090, sub, fontsize=8.4)


def phase(xa, xb, col, bcol, xl, name, lines):
    xm = (xa + xb) / 2
    box(xm - 0.033, ym + hm / 2 + 0.060, 0.066, 0.215, "white", col, lw=0.8, r=0.008, z=10)
    text(xm, ym + hm / 2 + 0.215, name, fontsize=8.4)
    text(xm, ym + hm / 2 + 0.122, lines[0], fontsize=7.8)
    farrow((xa + 0.006, ym + hm / 2), (xb - 0.006, ym + hm / 2), color=col, lw=1.2, ms=8)
    text(xm, ym + hm / 2 - 0.110, lines[1], fontsize=9.0)


phase(xs[0] + wm, xs[1], HEAT, HEAT_B, HEAT_XL, r"\textbf{heating}", ["laser on", r"$a_l$"])
phase(xs[1] + wm, xs[2], COOL, COOL_B, COOL_XL, r"\textbf{cooling}", ["laser off", r"$c_l$"])

out = os.path.join(ROOT, "Document/figures/fig_sim_setup")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=300)
print("saved", out)
