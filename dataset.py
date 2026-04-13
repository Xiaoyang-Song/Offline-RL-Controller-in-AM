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
        filename = os.path.join('..', 'LPBF-Simulation', 'RL_Dataset', f'trajectory_{trajectory_id:03d}', f'layer_{j+1}_data.mat')
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
    for traj in dataset:
        for step in traj:
            u, lp, r = step
            print(f"Trajectory step: u={u.shape}, u_mean, {u.mean():.4}, lp={lp}, r={r}")
            if step == 0:
                adj_cost = 0
            else:
                adj_cost = beta * np.linalg.norm(u - traj[step-1][0])  # Example: L2 distance to previous action

        break

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--n', type=int, default=5000, help='Number of trajectories to extract')
    args.add_argument('--step_size', type=int, default=10, help='Step size for trajectory extraction')
    args.add_argument('--trajectory_length', type=int, default=12, help='Length of each trajectory')
    args.add_argument('--mode', type=str, default='read', help='read or load')
    args = args.parse_args()


    if args.mode == 'read':
        n = args.ns
        step_size = args.step_size
        trajectory_length = args.trajectory_length

        id_list = np.arange(1, n+1, 1)
        dataset = gather_dataset(id_list, trajectory_length=trajectory_length)

        with open(f"Data/Dataset:layer_{trajectory_length}_stepsize_{step_size}_samples_{n}.pkl", "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'load':

        with open(f"Data/Dataset:layer_{args.trajectory_length}_stepsize_{args.step_size}_samples_{args.n}.pkl", "rb") as f:
            dataset = pickle.load(f)

        print(f"Loaded dataset with {len(dataset)} trajectories, each of length {len(dataset[0])}")

        # Transform with adjusted cost
        transform_with_adj_cost(dataset)

