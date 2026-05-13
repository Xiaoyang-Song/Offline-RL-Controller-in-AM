import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Discrete action set 
# ==========================================
# ACTION_LIST = torch.tensor([100,150,200,250,300,350,400,450,500,550,600], dtype=torch.float32)
ACTION_LIST = torch.arange(150, 310, 10, dtype=torch.float32)
# ACTION_LIST = torch.arange(100, 601, 20, dtype=torch.float32)
NUM_ACTIONS = len(ACTION_LIST)


# ==========================================
# Map action value -> index 0..15
# ==========================================
ACTION_TO_INDEX = {float(action.item()): idx for idx, action in enumerate(ACTION_LIST)}


def action_to_index(actions):
    action_array = np.array(actions, dtype=np.float32).reshape(-1)
    indices = []

    for action in action_array:
        action_value = float(action)
        if action_value not in ACTION_TO_INDEX:
            raise ValueError(f"Action {action_value} is not in ACTION_LIST.")
        indices.append(ACTION_TO_INDEX[action_value])

    return np.array(indices, dtype=np.int64)


# ==========================================
# Q-network: Q(s) -> R^{16}
# ==========================================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

class QNet(nn.Module):
    def __init__(self, state_dim, num_actions, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, num_actions),
        )
        # self.net.apply(init_weights)
        

    def forward(self, state):
        return self.net(state)  # (batch, num_actions)

# =============================================
# Dataset loader for discrete actions
# =============================================
# The dataset stores the post-action temperature field for each layer.
# To build an MDP transition consistent with deployment, use the previous
# layer's final temperature as the current state and the current record's
# temperature field as the next state.
def flatten_dataset(trajectories, initial_temperature=300.0):
    states, actions, rewards, next_states, dones = [], [], [], [], []

    sample_state = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
    initial_state = np.full_like(sample_state, initial_temperature, dtype=np.float32)

    for traj in trajectories:
        prev_state = initial_state.copy()

        for t, (post_state, action, reward) in enumerate(traj):
            next_state = np.asarray(post_state, dtype=np.float32).reshape(-1)

            states.append(prev_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(t == len(traj) - 1))

            prev_state = next_state

    action_indices = action_to_index(actions)

    return (
        torch.tensor(np.array(states), dtype=torch.float32, device=device),
        torch.tensor(action_indices, dtype=torch.long, device=device),
        torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(-1),
        torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
        torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(-1),
    )


def build_state_normalizer(states):
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0).clamp_min(1e-6)
    return state_mean, state_std


def normalize_states(states, state_mean, state_std):
    return (states - state_mean) / state_std


def build_policy_checkpoint(qnet, state_mean, state_std, state_dim, hidden_dim, train_config):
    return {
        'qnet_state_dict': qnet.state_dict(),
        'state_mean': state_mean.detach().cpu(),
        'state_std': state_std.detach().cpu(),
        'state_dim': state_dim,
        'hidden_dim': hidden_dim,
        'num_actions': NUM_ACTIONS,
        'action_list': ACTION_LIST.detach().cpu(),
        'train_config': train_config,
    }


def load_policy(checkpoint_path, map_location=device):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, QNet):
        checkpoint.eval()
        return {
            'qnet': checkpoint.to(device),
            'state_mean': None,
            'state_std': None,
            'action_list': ACTION_LIST,
            'train_config': {},
        }

    if 'qnet_state_dict' not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    qnet = QNet(checkpoint['state_dim'], checkpoint['num_actions'], checkpoint['hidden_dim']).to(device)
    qnet.load_state_dict(checkpoint['qnet_state_dict'])
    qnet.eval()

    saved_action_list = checkpoint.get('action_list')
    if saved_action_list is not None and not torch.equal(saved_action_list.cpu(), ACTION_LIST.cpu()):
        raise ValueError('Checkpoint action list does not match ACTION_LIST in model.py.')

    return {
        'qnet': qnet,
        'state_mean': checkpoint.get('state_mean'),
        'state_std': checkpoint.get('state_std'),
        'action_list': ACTION_LIST,
        'train_config': checkpoint.get('train_config', {}),
    }


# ==========================================
# ==========================================
# Modified train_fqi to record loss
# ==========================================
def train_fqi(
    trajectories,
    state_dim=None,
    h=128,
    gamma=0.99,
    K=10000,
    lr=1e-3,
    step_size=5000,
    gamma_lr=0.1,
    batch_size=2048,
    target_update_interval=200,
    initial_temperature=300.0,
):
    S, A_idx, R, S2, done = flatten_dataset(trajectories, initial_temperature=initial_temperature)
    state_mean, state_std = build_state_normalizer(S)
    S = normalize_states(S, state_mean, state_std)
    S2 = normalize_states(S2, state_mean, state_std)

    if state_dim is None:
        state_dim = S.shape[1]

    qnet = QNet(state_dim, NUM_ACTIONS, h).to(device)
    target_qnet = copy.deepcopy(qnet).to(device)
    target_qnet.eval()
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_lr)
    mse = nn.MSELoss()

    loss_history = []
    dataset_size = S.shape[0]
    effective_batch_size = min(batch_size, dataset_size)

    for k in range(K):
        batch_indices = torch.randint(0, dataset_size, (effective_batch_size,), device=device)
        batch_s = S[batch_indices]
        batch_a = A_idx[batch_indices]
        batch_r = R[batch_indices]
        batch_s2 = S2[batch_indices]
        batch_done = done[batch_indices]

        optimizer.zero_grad()

        with torch.no_grad():
            q_next = target_qnet(batch_s2)
            max_next_q = q_next.max(dim=1, keepdim=True)[0]
            target = batch_r + gamma * (1.0 - batch_done) * max_next_q

        q_pred_all = qnet(batch_s)
        q_pred = q_pred_all.gather(1, batch_a.unsqueeze(-1))

        loss = mse(q_pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (k + 1) % target_update_interval == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        loss_history.append(loss.item())

        if k % 100 == 0:
            print(f"Iter {k}: Q Loss = {loss.item():.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

    train_config = {
        'gamma': gamma,
        'iterations': K,
        'lr': lr,
        'step_size': step_size,
        'gamma_lr': gamma_lr,
        'batch_size': effective_batch_size,
        'target_update_interval': target_update_interval,
        'initial_temperature': initial_temperature,
    }
    checkpoint = build_policy_checkpoint(qnet, state_mean, state_std, state_dim, h, train_config)

    return checkpoint, loss_history


def train_pessimistic_q(
    trajectories,
    state_dim=None,
    h=128,
    gamma=0.99,
    K=10000,
    lr=1e-3,
    step_size=5000,
    gamma_lr=0.1,
    batch_size=2048,
    target_update_interval=200,
    initial_temperature=300.0,
    cql_alpha=1.0,
    cql_temperature=1.0,
):
    """
    Pessimistic offline Q-learning via a conservative Q penalty (CQL-style):
    total_loss = bellman_loss + cql_alpha * (logsumexp(Q(s, .)) - Q(s, a_data)).
    """
    S, A_idx, R, S2, done = flatten_dataset(trajectories, initial_temperature=initial_temperature)
    state_mean, state_std = build_state_normalizer(S)
    S = normalize_states(S, state_mean, state_std)
    S2 = normalize_states(S2, state_mean, state_std)

    if state_dim is None:
        state_dim = S.shape[1]

    qnet = QNet(state_dim, NUM_ACTIONS, h).to(device)
    target_qnet = copy.deepcopy(qnet).to(device)
    target_qnet.eval()
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_lr)
    mse = nn.MSELoss()

    total_loss_history = []
    bellman_loss_history = []
    conservative_loss_history = []

    dataset_size = S.shape[0]
    effective_batch_size = min(batch_size, dataset_size)
    temperature = max(float(cql_temperature), 1e-6)

    for k in range(K):
        batch_indices = torch.randint(0, dataset_size, (effective_batch_size,), device=device)
        batch_s = S[batch_indices]
        batch_a = A_idx[batch_indices]
        batch_r = R[batch_indices]
        batch_s2 = S2[batch_indices]
        batch_done = done[batch_indices]

        optimizer.zero_grad()

        with torch.no_grad():
            q_next = target_qnet(batch_s2)
            max_next_q = q_next.max(dim=1, keepdim=True)[0]
            target = batch_r + gamma * (1.0 - batch_done) * max_next_q

        q_pred_all = qnet(batch_s)
        q_pred_data = q_pred_all.gather(1, batch_a.unsqueeze(-1))

        bellman_loss = mse(q_pred_data, target)

        # Conservative term pushes down values on unseen / OOD actions.
        q_logsumexp = torch.logsumexp(q_pred_all / temperature, dim=1, keepdim=True) * temperature
        conservative_loss = (q_logsumexp - q_pred_data).mean()

        total_loss = bellman_loss + cql_alpha * conservative_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (k + 1) % target_update_interval == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        total_loss_history.append(total_loss.item())
        bellman_loss_history.append(bellman_loss.item())
        conservative_loss_history.append(conservative_loss.item())

        if k % 100 == 0:
            print(
                f"Iter {k}: Total={total_loss.item():.4f}, "
                f"Bellman={bellman_loss.item():.4f}, "
                f"Conservative={conservative_loss.item():.4f}, "
                f"LR={scheduler.get_last_lr()[0]:.6f}"
            )

    train_config = {
        'method': 'pessimistic_cql',
        'gamma': gamma,
        'iterations': K,
        'lr': lr,
        'step_size': step_size,
        'gamma_lr': gamma_lr,
        'batch_size': effective_batch_size,
        'target_update_interval': target_update_interval,
        'initial_temperature': initial_temperature,
        'cql_alpha': cql_alpha,
        'cql_temperature': temperature,
    }
    checkpoint = build_policy_checkpoint(qnet, state_mean, state_std, state_dim, h, train_config)

    losses = {
        'total': total_loss_history,
        'bellman': bellman_loss_history,
        'conservative': conservative_loss_history,
    }
    return checkpoint, losses



# ==========================================
# Policy function (argmax Q)
# ==========================================
def select_action(qnet, state):
    if isinstance(qnet, dict):
        state_mean = qnet.get('state_mean')
        state_std = qnet.get('state_std')
        action_list = qnet.get('action_list', ACTION_LIST)
        qnet = qnet['qnet']
    else:
        state_mean = None
        state_std = None
        action_list = ACTION_LIST

    state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(-1)
    if state_mean is not None and state_std is not None:
        state_mean = state_mean.to(device)
        state_std = state_std.to(device)
        state = normalize_states(state, state_mean, state_std)

    with torch.no_grad():
        q = qnet(state.unsqueeze(0))
        idx = q.argmax(dim=1).item()
        return action_list[idx].item()

def plot_loss(loss_history, fname='q_learning_loss.png'):
    os.makedirs('checkpoints/training_log', exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label='Q Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Offline Q-learning Loss')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f'checkpoints/training_log/{fname}')

# ==========================================
# Training
# ==========================================
# id_list = np.arange(1, 11, 1)
# print(id_list)
# dataset = gather_dataset(id_list, trajectory_length=8)

if __name__ == '__main__':
    n = 5000
    step_size = 10
    trajectory_length = 12

    beta = 0.0
    if beta > 0:
        file_path = os.path.join("Data", f"Dataset:layer_{trajectory_length}_stepsize_{step_size}_samples_{n}_beta_{beta}.pkl")
    else:
        file_path = os.path.join("Data", f"Dataset:layer_{trajectory_length}_stepsize_{step_size}_samples_{n}.pkl")
    
    with open(file_path, "rb") as f:
        print("Loading processed dataset...")
        dataset = pickle.load(f)
        print("Dataset loaded successfully!")

    state_dim = 1053
    h=300
    gamma=0.99
    K=40000
    lr=1e-3
    scheduler_step_size=10000
    gamma_lr=0.1
    checkpoint, loss_history = train_fqi(
        dataset,
        state_dim=state_dim,
        h=h,
        gamma=gamma,
        K=K,
        lr=lr,
        step_size=scheduler_step_size,
        gamma_lr=gamma_lr,
        initial_temperature=300.0,
    )
    torch.save(checkpoint, f"checkpoints/qnet_offline_{n}_{step_size}.pt")
    plot_loss(loss_history, fname=f'offline_q_loss_{n}_{step_size}.png')

    # checkpoint, losses = train_pessimistic_q(
    #     dataset,
    #     state_dim=1053,
    #     h=128,
    #     gamma=0.95,
    #     K=10000,
    #     lr=1e-3,
    #     step_size=5000,
    #     gamma_lr=0.1,
    #     batch_size=2048,
    #     target_update_interval=200,
    #     initial_temperature=300.0,
    #     cql_alpha=1.0,         # stronger pessimism if increased
    #     cql_temperature=1.0,   # smoothness in logsumexp
    # )
    # torch.save(checkpoint, "checkpoints/qnet_pessimistic_5000_10.pt")
    # # plot_loss(loss_history, fname='fqi_loss.png')
    # plot_loss(losses['total'], fname='pessimistic_q_total_loss.png')
    # plot_loss(losses['bellman'], fname='pessimistic_q_bellman_loss.png')
    # plot_loss(losses['conservative'], fname='pessimistic_q_conservative_loss.png')