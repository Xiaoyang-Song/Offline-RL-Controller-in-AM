"""
rl_narrow/offline_rl.py
-----------------------
Offline RL baselines trained ONLY on logged simulator data restricted to the narrow laser-power range
(200-300 W, the same transitions the narrow digital twin is trained on). No digital twin, no interaction.

Reward (method.tex Eq. reward-shaped with w = 1: offline data come from the simulator, so there is no
model uncertainty to discount):
    r_l = (1/|R_l|) sum_{j in R_l} exp(-d_{l,j} / W_T),   d = (T_L - u)^+ + (u - T_H)^+,
evaluated on the logged END-OF-HEATING field u_l, with the same ROI masks as the RL environment.

MDP (one step = one layer): s = field before the laser pass, a = laser power on the data grid
100:10:400 W (31 actions; the narrow data only contain 200..300 W), s' = post-cooling field, plus the
layer index and the build's cooling time as inputs.

Algorithms (same network, data, normalisation and optimiser):
  fqi : fitted Q-iteration / offline DQN    y = r + gamma (1 - done) max_a' Q_tgt(s', a')
  cql : fqi + alpha * E[logsumexp_a Q(s, a) - Q(s, a_data)]          (discrete CQL, Kumar et al. 2020)
  iql : expectile V, Q <- r + gamma V(s'), advantage-weighted policy  (Kostrikov et al. 2022)
Deployed greedily: argmax_a Q (fqi, cql) or argmax_a pi(a|s) (iql). The FINAL checkpoint is used
(offline RL has no return-based model selection without online access).

    python -m rl_narrow.offline_rl --algo cql --seed 1 --out_dir rl_narrow/runs/offline_cql_s1
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.dataset import load_trajectories, split_trajectories
from baselines.offline_q.model import RawQNet, ACTION_GRID
from online_RL_ucpg_v3.env import build_roi_masks, hinge_deviation


def quality_score(u, mask, T_l, T_h):
    """Method.tex reward with w = 1: mean over the ROI of exp(-d / W_T)."""
    d = hinge_deviation(u[mask] if mask is not None else u, T_l, T_h)
    return float(np.exp(-d / (T_h - T_l)).mean())


def build_transitions(trajs, masks, T_l, T_h, initial_temp=300.0, lp_filter=(200.0, 300.0)):
    """(s, a, r, s', layer, done, cool) per layer, reward recomputed from the logged EoH field.
    Chaining walks every trajectory in full; only transitions with laser power in lp_filter are kept
    (same convention as baselines.common.data_utils.build_offline_transitions)."""
    out = {k: [] for k in ("s", "a", "r", "s2", "layer", "done", "cool")}
    D = np.asarray(trajs[0][0].u_final).size
    for traj in trajs:
        prev = np.full(D, initial_temp, np.float32)
        for t, step in enumerate(traj):
            nxt = np.asarray(step.u_final, np.float32).reshape(-1)
            if lp_filter is None or lp_filter[0] <= step.lp_action <= lp_filter[1]:
                u = np.asarray(step.u_heat_final, np.float32).reshape(-1)
                out["s"].append(prev); out["a"].append(step.lp_action)
                out["r"].append(quality_score(u, masks[t], T_l, T_h)); out["s2"].append(nxt)
                out["layer"].append(t); out["done"].append(t == len(traj) - 1); out["cool"].append(step.cool_time)
            prev = nxt
    return {k: np.asarray(v) for k, v in out.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Offline FQI / CQL / IQL on narrow logged data (w = 1 reward).")
    p.add_argument("--algo", choices=["fqi", "cql", "iql"], required=True)
    p.add_argument("--data_path", default="Data/DatasetV2_layer_12_samples_5000.pkl")
    p.add_argument("--lp_min", type=float, default=200.0)
    p.add_argument("--lp_max", type=float, default=300.0)
    p.add_argument("--T_l", type=float, default=2000.0)
    p.add_argument("--T_h", type=float, default=2800.0)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
    p.add_argument("--width", type=float, default=12.0)
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--layer_embed_dim", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--target_sync_freq", type=int, default=2, help="fqi/cql: hard target sync every N epochs.")
    p.add_argument("--cql_alpha", type=float, default=1.0)
    p.add_argument("--iql_expectile", type=float, default=0.7)
    p.add_argument("--iql_beta", type=float, default=3.0, help="Advantage temperature of the IQL policy.")
    p.add_argument("--iql_tau", type=float, default=0.005, help="Polyak rate of the IQL target Q.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_transitions", type=int, default=0, help="Smoke tests only: subsample the data.")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    masks = build_roi_masks(args.mesh_path, args.n_layers, args.width, args.height,
                            args.sq_frac_start, args.sq_frac_end)
    train, val, _test = split_trajectories(load_trajectories(args.data_path), 0.10, 0.10, 42)   # surrogate split
    lp = (args.lp_min, args.lp_max)
    tr = build_transitions(train, masks, args.T_l, args.T_h, args.initial_temp, lp)
    va = build_transitions(val, masks, args.T_l, args.T_h, args.initial_temp, lp)
    print(f"[offline_rl:{args.algo}] train {len(tr['a'])} | val {len(va['a'])} transitions, "
          f"LP {np.unique(tr['a']).min():.0f}-{np.unique(tr['a']).max():.0f} W, reward {tr['r'].mean():.3f} "
          f"± {tr['r'].std():.3f}")

    sm, ss = tr["s"].mean(0), np.maximum(tr["s"].std(0), 1e-6)
    cm, cs = float(tr["cool"].mean()), max(float(tr["cool"].std()), 1e-6)
    grid = ACTION_GRID

    def tensors(d):
        T = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)
        a_idx = np.abs(d["a"][:, None] - grid[None, :]).argmin(1)
        return (T((d["s"] - sm) / ss, torch.float32), T(a_idx, torch.long), T(d["r"], torch.float32),
                T((d["s2"] - sm) / ss, torch.float32), T(d["layer"], torch.long), T(d["done"], torch.bool),
                T((d["cool"] - cm) / cs, torch.float32))

    if args.max_transitions:
        sub = lambda d: {k: v_[:args.max_transitions] for k, v_ in d.items()}
        tr, va = sub(tr), sub(va)
    S, A, R, S2, L, Dn, C = tensors(tr)
    vS, vA, vR, vS2, vL, vDn, vC = tensors(va)
    D, nA = S.shape[1], len(grid)
    cfg = dict(state_dim=D, n_actions=nA, hidden=args.hidden, depth=args.depth, n_layers=args.n_layers,
               layer_embed_dim=args.layer_embed_dim)
    q = RawQNet(**cfg).to(device)
    q_tgt = copy.deepcopy(q).eval()
    params = list(q.parameters())
    if args.algo == "iql":
        v = RawQNet(**dict(cfg, n_actions=1)).to(device)
        pi = RawQNet(**cfg).to(device)
        params += list(v.parameters()) + list(pi.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    def nxt_layer(l):
        return (l + 1).clamp(max=args.n_layers - 1)

    def losses(s, a, r, s2, l, dn, c):
        qs = q(s, l, c)
        q_sa = qs.gather(-1, a[:, None]).squeeze(-1)
        out = {}
        if args.algo in ("fqi", "cql"):
            with torch.no_grad():
                y = r + args.gamma * (~dn).float() * q_tgt(s2, nxt_layer(l), c).max(-1).values
            out["bellman"] = Fnn.mse_loss(q_sa, y)
            if args.algo == "cql":
                out["cql"] = args.cql_alpha * (torch.logsumexp(qs, -1) - q_sa).mean()
        else:                                               # IQL
            with torch.no_grad():
                q_t = q_tgt(s, l, c).gather(-1, a[:, None]).squeeze(-1)
                y = r + args.gamma * (~dn).float() * v(s2, nxt_layer(l), c).squeeze(-1)
            vs = v(s, l, c).squeeze(-1)
            u = q_t - vs
            out["v"] = (torch.abs(args.iql_expectile - (u < 0).float()) * u ** 2).mean()
            out["bellman"] = Fnn.mse_loss(q_sa, y)
            with torch.no_grad():
                w = torch.exp(args.iql_beta * (q_t - vs)).clamp(max=100.0)
            out["pi"] = (w * Fnn.cross_entropy(pi(s, l, c), a, reduction="none")).mean()
        return out

    hist, t0 = [], time.time()
    n = S.shape[0]
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        agg = {}
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            out = losses(S[idx], A[idx], R[idx], S2[idx], L[idx], Dn[idx], C[idx])
            loss = sum(out.values())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 10.0); opt.step()
            if args.algo == "iql":
                with torch.no_grad():
                    for pt, ps in zip(q_tgt.parameters(), q.parameters()):
                        pt.mul_(1 - args.iql_tau).add_(args.iql_tau * ps)
            for k, val_ in out.items():
                agg[k] = agg.get(k, 0.0) + val_.item() * len(idx) / n
        if args.algo != "iql" and ep % args.target_sync_freq == 0:
            q_tgt.load_state_dict(q.state_dict())
        with torch.no_grad():
            vout = losses(vS, vA, vR, vS2, vL, vDn, vC)
            q_all = q(S[:4096], L[:4096], C[:4096])
            greedy = (pi(S[:4096], L[:4096], C[:4096]) if args.algo == "iql" else q_all).argmax(-1)
            in_data = ((grid[greedy.cpu().numpy()] >= args.lp_min) & (grid[greedy.cpu().numpy()] <= args.lp_max)).mean()
        row = dict(epoch=ep, **{f"train_{k}": v_ for k, v_ in agg.items()},
                   **{f"val_{k}": v_.item() for k, v_ in vout.items()},
                   greedy_in_data=float(in_data), greedy_mean_W=float(grid[greedy.cpu().numpy()].mean()))
        hist.append(row)
        print(f"ep {ep:3d} | " + " | ".join(f"{k} {v_:.4f}" for k, v_ in row.items() if k != "epoch")
              + f" | {time.time() - t0:.0f}s")

    ckpt = dict(algo=args.algo, model_config=cfg, q_state_dict=q.state_dict(),
                pi_state_dict=pi.state_dict() if args.algo == "iql" else None,
                state_mean=sm, state_std=ss, cool_mean=cm, cool_std=cs, action_grid=grid,
                train_args=vars(args), history=hist)
    torch.save(ckpt, os.path.join(args.out_dir, "offline_final.pt"))
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    print(f"[offline_rl:{args.algo}] saved {os.path.join(args.out_dir, 'offline_final.pt')}")


class OfflineController:
    """act(ctx) for the real-physics evaluator: own normalisation, greedy over the action grid."""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = RawQNet(**ck["model_config"]).to(device).eval()
        self.net.load_state_dict(ck["pi_state_dict"] if ck["algo"] == "iql" else ck["q_state_dict"])
        self.sm = torch.as_tensor(ck["state_mean"], dtype=torch.float32, device=device)
        self.ss = torch.as_tensor(ck["state_std"], dtype=torch.float32, device=device)
        self.cm, self.cs, self.grid, self.device = ck["cool_mean"], ck["cool_std"], ck["action_grid"], device

    @torch.no_grad()
    def act(self, ctx):
        s = (torch.as_tensor(ctx.raw_state, dtype=torch.float32, device=self.device)[None] - self.sm) / self.ss
        out = self.net(s, torch.tensor([ctx.layer], device=self.device),
                       torch.tensor([(ctx.cool_time - self.cm) / self.cs], dtype=torch.float32, device=self.device))
        return float(self.grid[int(out.argmax())])


if __name__ == "__main__":
    main()
