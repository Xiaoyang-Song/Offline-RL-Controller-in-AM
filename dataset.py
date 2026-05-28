from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import pickle

def extract_single_trajectory(trajectory_id, trajectory_length=8):
    trajectory = []
    for j in range(trajectory_length):
        filename = os.path.join('..', 'LPBF-Simulation', 'RL_Dataset_lc', f'trajectory_{trajectory_id:03d}', f'layer_{j+1}_data.mat')
        data = loadmat(filename)
        # Extract s, a, r
        ss = data['SS_action'][0][0]
        lp = data['LP_action'][0][0]
        u = np.array(data['uFinal'])
        uAll = data['uAll']
        r = -data['meanDeviation'][0][0]

        # Append
        sar = [u, lp, r]
        # print(u.shape, lp.shape, ss.shape, r.shape, lp, ss, r)
        trajectory.append(sar)
    return trajectory


def gather_dataset(id_list, trajectory_length=8):
    dataset = []
    for trajectory_id in tqdm(id_list):
        traj = extract_single_trajectory(trajectory_id, trajectory_length)
        dataset.append(traj)
    return dataset


def transform_with_adj_cost(dataset, beta=0.1):
    modified_dataset = []
    for traj in tqdm(dataset):
        modified_traj = []
        for i, step in enumerate(traj):
            u, lp, r = step
            # print(f"Trajectory step: u={u.shape}, u_mean, {u.mean():.4}, lp={lp}, r={r}")
            if i == 0:
                adj_cost = 0
            else:
                lp_prev = float(traj[i-1][1])
                adj_cost = beta * np.linalg.norm(lp - lp_prev)  # Example: L2 distance to previous action
            # print(adj_cost)
            r_adj = r - adj_cost
            modified_traj.append((u, lp, r_adj))
        modified_dataset.append(modified_traj)

    return modified_dataset

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--n', type=int, default=5000, help='Number of trajectories to extract')
    args.add_argument('--step_size', type=int, default=10, help='Step size for trajectory extraction')
    args.add_argument('--trajectory_length', type=int, default=12, help='Length of each trajectory')
    args.add_argument('--beta', type=float, default=0.1, help='Cost adjustment factor')
    args.add_argument('--mode', type=str, default='read', help='read or load')
    args.add_argument('--lb', type=int, default=150, help='Lower bound laser power')
    args.add_argument('--ub', type=int, default=300, help='Upper bound laser power')
    args.add_argument('--tag', type=str, default=None, help='read or load')
    args = args.parse_args()

    n = args.n
    step_size = args.step_size
    trajectory_length = args.trajectory_length

    if args.mode == 'read':

        id_list = np.arange(1, n+1, 1)
        dataset = gather_dataset(id_list, trajectory_length=trajectory_length)

        with open(f"Data/Dataset:layer_{trajectory_length}_stepsize_{step_size}_samples_{n}_{args.lb}_{args.ub}{f'_{args.tag}' if args.tag else ''}.pkl", "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'load':

        with open(f"Data/Dataset:layer_{trajectory_length}_stepsize_{step_size}_samples_{n}_{args.lb}_{args.ub}{f'_{args.tag}' if args.tag else ''}.pkl", "rb") as f:
            dataset = pickle.load(f)

        print(f"Loaded dataset with {len(dataset)} trajectories, each of length {len(dataset[0])}")

        # Transform with adjusted cost
        modified_dataset = transform_with_adj_cost(dataset, args.beta)
        with open(f"Data/Dataset:layer_{trajectory_length}_stepsize_{step_size}_samples_{n}_{args.lb}_{args.ub}_beta_{args.beta}{f'_{args.tag}' if args.tag else ''}.pkl", "wb") as f:
            pickle.dump(modified_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

