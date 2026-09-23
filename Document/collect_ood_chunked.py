"""Chunked re-run of surrogate_model_v3.evaluate_ood's data collection (same functions,
same binning), to fit in limited RAM. Dumps heating-only and combined binned arrays."""
import os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_model_v3.dataset import load_trajectories, TwoStageSurrogateDataset
from surrogate_model_v3.model import load_surrogate
from surrogate_model_v3.evaluate_ood import collect_ood_samples, collect_ood_samples_combined, bin_by_action

ckpt, data_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(8)
model, sm, ss, lm, ls, cm, cs = load_surrogate(ckpt, "cpu")
trajs = load_trajectories(data_path)
heat, comb = [], []
for i in range(0, len(trajs), 500):
    ds = TwoStageSurrogateDataset(trajs[i:i+500], state_mean=sm.cpu(), state_std=ss.cpu(), lp_mean=lm, lp_std=ls,
                                  cool_mean=cm, cool_std=cs, initial_temp=300.0, n_ensemble=model.n_ensemble, bootstrap_seed=0)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    heat.append(collect_ood_samples(model, dl, sm, ss, lm, ls, "cpu"))
    comb.append(collect_ood_samples_combined(model, dl, sm, ss, lm, ls, "cpu", propagate_uncertainty=False, num_probes=4))
    print("chunk", i // 500 + 1, "done", flush=True)
cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]}
bh, bc = bin_by_action(cat(heat), 12), bin_by_action(cat(comb), 12)
dump = {f"heat/{k}": np.asarray(v) for k, v in bh.items()}
dump.update({f"comb/{k}": np.asarray(v) for k, v in bc.items()})
dump["id_ranges"] = np.array([[200.0, 300.0]])
np.savez(out, **dump); print("saved", out)
