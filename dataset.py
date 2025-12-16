from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

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


if __name__ == '__main__':
    n = 5000
    step_size = 20
    trajectory_length = 8


    id_list = np.arange(1, n+1, 1)
    dataset = gather_dataset(id_list, trajectory_length=trajectory_length)

    import pickle

    with open(f"data/dataset_{n}_{step_size}.pkl", "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)