"""
surrogate_model_v3/plot_propagation_maps.py
--------------------------------------------------------
Spatial (per-node) view of what EKF-style Jacobian propagation does to the
combined two-stage uncertainty of a DIAGONAL (rank=0) checkpoint — the
final configuration (diag + propagate_uncertainty=True).

Why not the node-mean (compare_uncertainty_methods.py's per-layer curves /
jacobian_amplification_map.png): laser heating only acts on the small
middle scan square (~45 of 1053 nodes), so most nodes carry ~zero heating
uncertainty and ~zero propagation effect. Averaging over ALL nodes dilutes
the effect inside the scan region toward 1. Here everything stays per node:

  For every test trajectory, the mean-state rollout (same as model.rollout)
  is re-run and, per layer and per node, both combinations are computed from
  the SAME heat/cool ensemble outputs:
      naive      : Var_heat + Var_cool                (combine_stage_uncertainties_naive_full)
      propagated : (I+J) Var_heat (I+J)^T + Var_cool  (combine_stage_uncertainties_propagated)
  Test-set maps average variances over trajectories node-wise and take the
  ratio as sqrt(mean prop var / mean naive var) — a σ ratio per node.

Figures use the same layout conventions as evaluate.py's / plot_transitions.py's
traj_XXX/layer_XX_LP###W.png (full mesh domain, X/Y axes, faint element mesh,
one colourbar per panel, `hot` for magnitudes, 3 columns), so they can sit
next to those in the paper. σ is shown in Kelvin (normalised σ × state_std).

Outputs (--out_dir); every figure is written as .png (150 dpi) and .pdf
------------------------------------------------------------------------
  mean_layer_XX_propagation      — test-set mean, one per --detail_layers layer.
      Rows: epistemic σ, total σ. Columns: naive [K] | propagated [K] |
      σ_prop/σ_naive (blue = cooling damps heating-stage uncertainty, red =
      amplifies; log-symmetric diverging scale).
  traj_XXX/layer_XX_LP###W_propagation — same layout for single test
      trajectories (--n_examples), same trajectory indices / file naming as
      plot_transitions.py's traj_XXX folders, so they line up one-to-one.
  ratio_maps_epistemic / ratio_maps_total — test-set σ ratio, all layers.
  roi_ratio — per-layer σ ratio inside vs. outside the scan square vs. all
      nodes (epistemic | total), and the per-trajectory in-square total ratio
      (median + IQR band, split ID/OOD laser power if --id_range_* given).
  propagation_maps.npz — everything needed to redraw (see --from_npz).

Usage
-----
    python -m surrogate_model_v3.plot_propagation_maps \\
        --data_path  Data/DatasetV2_layer_12_samples_5000.pkl \\
        --checkpoint surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \\
        --id_range_min 200 --id_range_max 300 \\
        --out_dir surrogate_model_v3/results_propagation_maps_narrow

    # restyle only (no model / GPU needed):
    python -m surrogate_model_v3.plot_propagation_maps \\
        --from_npz surrogate_model_v3/results_propagation_maps_narrow/propagation_maps.npz \\
        --out_dir  surrogate_model_v3/results_propagation_maps_narrow
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.dataset import load_trajectories, split_trajectories, TwoStageTrajectoryDataset
from surrogate_model_v3.model import load_surrogate, combine_stage_uncertainties_naive_full

QUANTITIES = ("epi", "ale", "total")
Q_NAME = {"epi": "Epistemic", "ale": "Aleatoric", "total": "Total"}

# Diverging: blue (damps) <-> red (amplifies), neutral gray midpoint at ratio 1.
DIVERGING = LinearSegmentedColormap.from_list(
    "damp_amplify", ["#123f75", "#2a78d6", "#a9c8ee", "#f0efec", "#f2b0a5", "#e34948", "#8a1c1b"])
RATIO_LABEL = "σ_prop / σ_naive"
SQUARE_STYLE = dict(fill=False, linewidth=0.9, linestyle="--")
# Region lines, in the tab-colour / marker style of evaluate.py's line plots.
REGION_STYLE = {
    "inside":  dict(color="tab:blue",   marker="o", linestyle="-",  label="Inside scan square"),
    "outside": dict(color="tab:orange", marker="s", linestyle="--", label="Outside scan square"),
    "all":     dict(color="tab:green",  marker="^", linestyle=":",  label="All nodes"),
}


def _save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[propagation_maps] Saved → {path_no_ext}.{{png,pdf}}")


# =============================================================================
# Mesh / scan region
# =============================================================================

def load_mesh(mesh_path: str):
    """Returns (shading triangulation over ALL nodes, element-edge triangulation, nodes).

    Shading: each 6-node quadratic element (corners c1,c2,c3 + midside
    m12,m23,m31, rows 4-6 of `elements`) is split into its 4 standard linear
    sub-triangles, so every node's value is rendered (evaluate.py's
    corner-only triangulation uses 283 of 1053 nodes). Edges: the corner
    triangles, i.e. the real FE elements — used only for the faint mesh
    overlay, so it looks the same as in evaluate.py's plots."""
    data = loadmat(mesh_path)
    nodes = data["nodes"].astype(np.float64)
    c1, c2, c3, m12, m23, m31 = (data["elements"][:6] - 1).astype(np.int32)
    tri = np.concatenate([np.stack(t, axis=1) for t in
                          ((c1, m12, m31), (m12, c2, m23), (m31, m23, c3), (m12, m23, m31))])
    shade = mtri.Triangulation(nodes[0], nodes[1], tri)
    edges = mtri.Triangulation(nodes[0], nodes[1], np.stack([c1, c2, c3], axis=1))
    return shade, edges, nodes


def scan_squares(nodes: np.ndarray, n_layers: int, initial_fraction: float = 0.4,
                 final_fraction: float = 0.5):
    """Per-layer scan square (cx, cy, half_side) — same schedule as
    dataset.compute_roi_weights_table / simulateHeatingCooling_v2.m."""
    x, y = nodes
    w, h = x.max() - x.min(), y.max() - y.min()
    cx, cy, min_dim = x.min() + w / 2.0, y.min() + h / 2.0, min(w, h)
    return [(cx, cy, min_dim * f / 2.0) for f in np.linspace(initial_fraction, final_fraction, n_layers)]


def roi_masks(nodes: np.ndarray, squares) -> np.ndarray:
    """(n_layers, D) bool — node inside that layer's scan square."""
    x, y = nodes
    return np.stack([(np.abs(x - cx) <= hs) & (np.abs(y - cy) <= hs) for cx, cy, hs in squares])


# =============================================================================
# Per-node accumulation
# =============================================================================

def accumulate(model, loader, device, traj_len, roi, num_probes, lp_mean, lp_std, id_range, n_examples):
    """Mean-state rollout over every test trajectory. Returns per-layer,
    per-node SUMS of naive/propagated variances (epi/ale/total) for groups
    'all' (and 'id'/'ood' by that layer's laser power if id_range is given),
    their counts, per-(layer, trajectory) in-ROI σ ratios + raw laser powers,
    and the full per-node variances of the first `n_examples` trajectories."""
    D = roi.shape[1]
    groups = ["all"] + (["id", "ood"] if id_range is not None else [])
    keys = [f"{m}_{q}" for m in ("naive", "prop") for q in QUANTITIES]
    sums = {g: {k: np.zeros((traj_len, D)) for k in keys} for g in groups}
    counts = {g: np.zeros(traj_len) for g in groups}
    per_traj = {q: [[] for _ in range(traj_len)] for q in QUANTITIES}
    per_traj_lp = [[] for _ in range(traj_len)]
    examples = {k: [] for k in keys}
    roi_t = torch.as_tensor(roi, device=device)
    n_seen = 0

    model.eval()
    with torch.no_grad():
        for traj_s, _h, traj_a, traj_c, _bmask in loader:
            traj_s, traj_a, traj_c = traj_s.to(device), traj_a.to(device), traj_c.to(device)
            B = traj_s.shape[0]
            n_keep = max(0, min(B, n_examples - n_seen))
            ex_batch = {k: [] for k in keys}
            s_t = traj_s[:, 0, :]
            for t in range(traj_len):
                a_t, c_t = traj_a[:, t, :], traj_c[:, t, :]
                layer_idx = torch.full((B,), t, dtype=torch.long, device=device)
                heat_full = model.predict_heating_ensemble_full(s_t, a_t, layer_idx)
                s_heat = s_t + heat_full["mu_mean"]
                cool_full = model.predict_cooling_ensemble_full(s_heat, c_t, layer_idx)

                naive = combine_stage_uncertainties_naive_full(heat_full, cool_full)
                prop = model.combine_stage_uncertainties_propagated(
                    s_heat, c_t, layer_idx, heat_full, cool_full, num_probes=num_probes)
                var = {}
                for tag, combo in (("naive", naive), ("prop", prop)):
                    var[f"{tag}_epi"] = combo["epistemic_std"].pow(2)
                    var[f"{tag}_ale"] = combo["aleatoric_std"].pow(2)
                    var[f"{tag}_total"] = combo["total_std"].pow(2)

                # per-trajectory in-ROI σ ratio (ROI-summed variance, then ratio)
                m = roi_t[t].float()
                for q in QUANTITIES:
                    r = ((var[f"prop_{q}"] * m).sum(-1) / (var[f"naive_{q}"] * m).sum(-1).clamp_min(1e-30)).sqrt()
                    per_traj[q][t].extend(r.cpu().numpy().tolist())

                lp_raw = (a_t[:, 0] * lp_std + lp_mean).cpu().numpy()
                per_traj_lp[t].extend(lp_raw.tolist())
                sel = {"all": np.ones(B, dtype=bool)}
                if id_range is not None:
                    sel["id"] = (lp_raw >= id_range[0]) & (lp_raw <= id_range[1])
                    sel["ood"] = ~sel["id"]
                for g in groups:
                    idx = torch.as_tensor(sel[g], device=device)
                    if idx.sum() == 0:
                        continue
                    counts[g][t] += int(idx.sum())
                    for k, v in var.items():
                        sums[g][k][t] += v[idx].sum(0).cpu().numpy()
                if n_keep:
                    for k, v in var.items():
                        ex_batch[k].append(v[:n_keep].cpu().numpy())

                s_t = s_heat + cool_full["mu_mean"]
            if n_keep:
                for k in keys:
                    examples[k].append(np.stack(ex_batch[k], axis=1))       # (n_keep, T, D)
            n_seen += B
    per_traj = {q: np.array(v) for q, v in per_traj.items()}                # (T, N_traj)
    examples = {k: (np.concatenate(v) if v else np.zeros((0, traj_len, D))) for k, v in examples.items()}
    return sums, counts, per_traj, np.array(per_traj_lp), examples


# =============================================================================
# Plotting (layout conventions follow evaluate.py's per-layer field plots)
# =============================================================================

def _log_ratio_limit(*ratios, pct=99.5, floor=np.log2(1.05)):
    """Symmetric log2 colour limit shared by all given ratio fields."""
    lr = np.concatenate([np.abs(np.log2(r[np.isfinite(r) & (r > 0)])).ravel() for r in ratios])
    return max(np.percentile(lr, pct), floor)


def _ratio_colorbar(fig, mappable, ax, lim, label=RATIO_LABEL, **kw):
    """Colourbar for a log2(ratio) field, ticked in plain ratio values."""
    cb = fig.colorbar(mappable, ax=ax, label=label, **kw)
    cands = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0])
    ticks = cands[np.abs(np.log2(cands)) <= lim + 1e-9]
    cb.set_ticks(np.log2(ticks))
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    return cb


def _field_panel(ax, shade, edges, square, values, title, cmap, norm, xlabels=True, ylabels=True):
    """One mesh panel in evaluate.py's style: full domain, auto aspect,
    gouraud shading, faint element mesh, X/Y axes, small title."""
    tpc = ax.tripcolor(shade, values, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
    # same faint element mesh as evaluate.py; fainter on the light diverging maps
    ax.triplot(edges, color="k", linewidth=0.15, alpha=0.3 if cmap == "hot" else 0.12)
    cx, cy, hs = square
    ax.add_patch(Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs,
                           edgecolor="w" if cmap == "hot" else "k", **SQUARE_STYLE))
    ax.set_aspect("auto")
    ax.set_xlabel("X" if xlabels else "")
    ax.set_ylabel("Y" if ylabels else "")
    if not xlabels:
        ax.tick_params(labelbottom=False)
    if not ylabels:
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=9)
    return tpc


def _ratio(prop_var, naive_var):
    return np.sqrt(prop_var / np.maximum(naive_var, 1e-30))


def plot_layer_propagation(var, state_std, shade, edges, square, lim, suptitle, path_no_ext):
    """2x3 figure for ONE layer, same layout as evaluate.py's layer_XX_LP###W.png.
    var: dict of (D,) normalised variances (naive_/prop_ x epi/total).
    Rows: epistemic, total. Columns: naive σ [K] | propagated σ [K] | ratio.
    Each row's two σ panels share one colour scale."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(suptitle, fontsize=10)
    for r, q in enumerate(("epi", "total")):
        s_naive = np.sqrt(var[f"naive_{q}"]) * state_std
        s_prop = np.sqrt(var[f"prop_{q}"]) * state_std
        norm = Normalize(vmin=0.0, vmax=max(s_naive.max(), s_prop.max()))
        for c, (vals, lab) in enumerate(((s_naive, "naive"), (s_prop, "propagated"))):
            tpc = _field_panel(axes[r, c], shade, edges, square, vals,
                               f"{Q_NAME[q]} σ — {lab} [K]  mean={vals.mean():.3g}  max={vals.max():.3g}",
                               "hot", norm)
            fig.colorbar(tpc, ax=axes[r, c], label="σ [K]")
        ratio = _ratio(var[f"prop_{q}"], var[f"naive_{q}"])
        tpc = _field_panel(axes[r, 2], shade, edges, square, np.log2(np.clip(ratio, 1e-6, None)),
                           f"{Q_NAME[q]} {RATIO_LABEL}  min={ratio.min():.2f}  max={ratio.max():.2f}",
                           DIVERGING, Normalize(-lim, lim))
        _ratio_colorbar(fig, tpc, axes[r, 2], lim)
    fig.tight_layout()
    _save(fig, path_no_ext)


def plot_ratio_maps(ratio, shade, edges, squares, lim, q, path_no_ext):
    """ratio: (T, D) test-set σ ratio, one mesh panel per layer (4x3 grid),
    shared diverging colourbar on the right."""
    T = ratio.shape[0]
    ncols = 3
    nrows = int(np.ceil(T / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows), squeeze=False)
    fig.suptitle(f"{Q_NAME[q]} σ — effect of Jacobian propagation per node, test-set mean  "
                 f"(blue: cooling damps heating-stage uncertainty, red: amplifies; dashed: scan square)",
                 fontsize=10)
    for t in range(nrows * ncols):
        ax = axes[t // ncols, t % ncols]
        if t >= T:
            ax.axis("off")
            continue
        tpc = _field_panel(ax, shade, edges, squares[t], np.log2(np.clip(ratio[t], 1e-6, None)),
                           f"Layer {t + 1}  min={ratio[t].min():.2f}  max={ratio[t].max():.2f}",
                           DIVERGING, Normalize(-lim, lim),
                           xlabels=(t // ncols == nrows - 1), ylabels=(t % ncols == 0))
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.15, 0.012, 0.7])
    _ratio_colorbar(fig, tpc, None, lim, cax=cax)
    _save(fig, path_no_ext)


def _region_ratio(mv, roi, q, region):
    T = roi.shape[0]
    out = np.empty(T)
    for t in range(T):
        m = {"inside": roi[t], "outside": ~roi[t], "all": np.ones_like(roi[t])}[region]
        out[t] = np.sqrt(mv[f"prop_{q}"][t][m].sum() / mv[f"naive_{q}"][t][m].sum())
    return out


def plot_roi_ratio(mv, roi, per_traj, per_traj_lp, id_range, path_no_ext):
    """1x3 line plots in the style of evaluate.py's sigma_per_layer.png."""
    T = roi.shape[0]
    layers = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.6), sharey=True)

    for ax, q in ((axes[0], "epi"), (axes[1], "total")):
        for region, st in REGION_STYLE.items():
            ax.plot(layers, _region_ratio(mv, roi, q, region), linewidth=1.5, **st)
        ax.set_title(f"{Q_NAME[q]} σ — {RATIO_LABEL} by region (test-set mean)")

    ax = axes[2]
    groups = [("All test trajectories", np.ones_like(per_traj_lp, dtype=bool), "tab:purple", "o")]
    if id_range is not None:
        id_mask = (per_traj_lp >= id_range[0]) & (per_traj_lp <= id_range[1])
        groups = [(f"LP in training range [{id_range[0]:.0f}, {id_range[1]:.0f}] W", id_mask, "tab:green", "o"),
                  ("LP outside training range", ~id_mask, "tab:red", "s")]
    for lab, mask, col, mk in groups:
        med, q1, q3 = (np.array([np.percentile(per_traj["total"][t][mask[t]], p) if mask[t].any() else np.nan
                                 for t in range(T)]) for p in (50, 25, 75))
        ax.fill_between(layers, q1, q3, color=col, alpha=0.15, linewidth=0)
        ax.plot(layers, med, color=col, marker=mk, linewidth=1.5, label=f"{lab} (median, IQR)")
    ax.set_title(f"Total σ inside scan square — {RATIO_LABEL} per trajectory")

    for ax in axes:
        ax.axhline(1.0, color="k", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Layer"); ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(RATIO_LABEL)
    fig.tight_layout()
    _save(fig, path_no_ext)


def make_all_plots(mean_var, per_traj, per_traj_lp, examples, ex_lp, ex_ids, state_std,
                   shade, edges, squares, roi, id_range, detail_layers, out_dir):
    mv = mean_var["all"]
    T = roi.shape[0]
    ratios = {q: _ratio(mv[f"prop_{q}"], mv[f"naive_{q}"]) for q in ("epi", "total")}
    lim = _log_ratio_limit(*ratios.values())

    for q in ("epi", "total"):
        plot_ratio_maps(ratios[q], shade, edges, squares, lim, q,
                        os.path.join(out_dir, f"ratio_maps_{'epistemic' if q == 'epi' else 'total'}"))
    for L in detail_layers:
        if not 1 <= L <= T:
            continue
        t = L - 1
        ins = {q: _region_ratio(mv, roi, q, "inside")[t] for q in ("epi", "total")}
        plot_layer_propagation(
            {k: v[t] for k, v in mv.items()}, state_std, shade, edges, squares[t], lim,
            f"Test-set mean — Layer {L}  |  inside scan square: epistemic {RATIO_LABEL} = {ins['epi']:.2f}, "
            f"total = {ins['total']:.2f}",
            os.path.join(out_dir, f"mean_layer_{L:02d}_propagation"))
    plot_roi_ratio(mv, roi, per_traj, per_traj_lp, id_range, os.path.join(out_dir, "roi_ratio"))

    # single trajectories — same indices / naming as plot_transitions.py's traj_XXX folders
    for i in range(examples["naive_total"].shape[0]):
        traj_dir = os.path.join(out_dir, f"traj_{int(ex_ids[i]):03d}")
        os.makedirs(traj_dir, exist_ok=True)
        for t in range(T):
            var = {k: v[i, t] for k, v in examples.items()}
            r_in = {q: np.sqrt(var[f"prop_{q}"][roi[t]].sum() / var[f"naive_{q}"][roi[t]].sum())
                    for q in ("epi", "total")}
            lp = ex_lp[t, i]
            plot_layer_propagation(
                var, state_std, shade, edges, squares[t], lim,
                f"Traj {int(ex_ids[i])} — Layer {t + 1}  |  LP = {lp:.0f} W  |  inside scan square: "
                f"epistemic {RATIO_LABEL} = {r_in['epi']:.2f}, total = {r_in['total']:.2f}",
                os.path.join(traj_dir, f"layer_{t + 1:02d}_LP{lp:.0f}W_propagation"))

    print("\n[propagation_maps] σ ratio (prop/naive), node-summed variance:")
    print("  layer |  epi in   epi out  epi all | total in total out total all")
    for t in range(T):
        row = [_region_ratio(mv, roi, q, reg)[t] for q in ("epi", "total") for reg in ("inside", "outside", "all")]
        print(f"  {t + 1:5d} | " + "  ".join(f"{v:7.3f}" for v in row))


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None, help="A rank=0 (diagonal) checkpoint.")
    p.add_argument("--from_npz", type=str, default=None,
                   help="Redraw from a previous run's propagation_maps.npz (skips the model entirely).")
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_probes", type=int, default=4)
    p.add_argument("--n_examples", type=int, default=1,
                   help="Per-trajectory figures for the first N test trajectories (traj_000, ...), "
                        "matching plot_transitions.py's indices.")
    p.add_argument("--id_range_min", type=float, default=None,
                   help="Optional training laser-power range [W] — splits roi_ratio's last panel into ID/OOD.")
    p.add_argument("--id_range_max", type=float, default=None)
    p.add_argument("--detail_layers", type=int, nargs="+", default=[1, 6, 12])
    p.add_argument("--out_dir", type=str, default="surrogate_model_v3/results_propagation_maps")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    shade, edges, nodes = load_mesh(args.mesh_path)
    keys = [f"{m}_{q}" for m in ("naive", "prop") for q in QUANTITIES]

    if args.from_npz:
        z = np.load(args.from_npz)
        groups = [g for g in ("all", "id", "ood") if f"count_{g}" in z]
        mean_var = {g: {k: z[f"{g}_{k}"] for k in keys} for g in groups}
        per_traj = {q: z[f"per_traj_{q}"] for q in QUANTITIES}
        per_traj_lp = z["per_traj_lp"]
        examples = {k: z[f"example_{k}"] for k in keys}
        state_std = z["state_std"]
        id_range = tuple(z["id_range"]) if "id_range" in z else None
        traj_len = per_traj_lp.shape[0]
    else:
        if not (args.data_path and args.checkpoint):
            raise ValueError("Pass --data_path and --checkpoint (or --from_npz).")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)  # Hutchinson probes (aleatoric-diagonal part only)
        model, sm, ss, lm, ls, cm, cs = load_surrogate(args.checkpoint, device)
        if model.rank != 0:
            print(f"[propagation_maps] WARNING: checkpoint has rank={model.rank}; this script targets rank=0.")
        assert nodes.shape[1] == model.state_dim, (nodes.shape, model.state_dim)
        state_std = ss.detach().cpu().numpy().reshape(-1)

        _train, _val, test_trajs = split_trajectories(
            load_trajectories(args.data_path), val_fraction=args.val_fraction,
            test_fraction=args.test_fraction, seed=args.seed)
        traj_len = len(test_trajs[0])
        ds = TwoStageTrajectoryDataset(
            test_trajs, state_mean=sm.cpu(), state_std=ss.cpu(), lp_mean=lm, lp_std=ls,
            cool_mean=cm, cool_std=cs, initial_temp=args.initial_temp, n_ensemble=model.n_ensemble,
            bootstrap_seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        id_range = (args.id_range_min, args.id_range_max) if args.id_range_min is not None else None
        roi_tmp = roi_masks(nodes, scan_squares(nodes, traj_len))
        sums, counts, per_traj, per_traj_lp, examples = accumulate(
            model, loader, device, traj_len, roi_tmp, args.num_probes, lm, ls, id_range, args.n_examples)
        mean_var = {g: {k: v / np.maximum(counts[g], 1)[:, None] for k, v in s.items()} for g, s in sums.items()}

        np.savez(os.path.join(args.out_dir, "propagation_maps.npz"),
                 **{f"{g}_{k}": v for g, d in mean_var.items() for k, v in d.items()},
                 **{f"count_{g}": c for g, c in counts.items()},
                 **{f"per_traj_{q}": v for q, v in per_traj.items()},
                 **{f"example_{k}": v for k, v in examples.items()},
                 per_traj_lp=per_traj_lp, state_std=state_std,
                 **({"id_range": np.array(id_range)} if id_range is not None else {}))

    n_ex = examples["naive_total"].shape[0]
    ex_ids = np.arange(n_ex)                    # test-split order, as in plot_transitions.py
    ex_lp = per_traj_lp[:, :n_ex]

    squares = scan_squares(nodes, traj_len)
    roi = roi_masks(nodes, squares)
    print(f"[propagation_maps] scan-square nodes per layer: {roi.sum(1).tolist()} / {roi.shape[1]}")
    make_all_plots(mean_var, per_traj, per_traj_lp, examples, ex_lp, ex_ids, state_std,
                   shade, edges, squares, roi, id_range, args.detail_layers, args.out_dir)
    print(f"\n[propagation_maps] Complete. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
