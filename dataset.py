from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_single_trajectory(trajectory_id, trajectory_length=8):
    trajectory = []
    for j in range(trajectory_length):
        filename = os.path.join('..', 'RL_Dataset', f'trajectory_00{trajectory_id}', f'layer_{j+1}_data.mat')
        data = loadmat(filename)
        # Extract s, a, r
        ss = data['SS_action'][0][0]
        lp = data['LP_action'][0][0]
        u = np.array(data['uFinal'])
        uAll = data['uAll']
        r = -data['meanDeviation'][0][0]
        # Append
        sar = [u, lp, r]
        print(u.shape, lp.shape, ss.shape, r.shape, lp, ss, r)
        trajectory.append(sar)
    return trajectory


def gather_dataset(id_list, trajectory_length=8):
    dataset = []
    for trajectory_id in id_list:
        traj = extract_single_trajectory(trajectory_id, trajectory_length)
        dataset.append(traj)
    return dataset