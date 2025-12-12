import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import *
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Discrete action set
# ==========================================
ACTION_LIST = torch.tensor([100,150,200,250,300,350,400,450,500,550,600], dtype=torch.float32)
NUM_ACTIONS = len(ACTION_LIST)


# ==========================================
# Map action value -> index 0..10
# ==========================================
def action_to_index(a):
    a = np.array(a).astype(np.float32)
    return np.array([np.where(ACTION_LIST.cpu().numpy() == x)[0][0] for x in a])


# ==========================================
# Q-network: Q(s) -> R^{11}
# ==========================================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

class QNet(nn.Module):
    def __init__(self, state_dim, num_actions, h=512):
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
        return self.net(state)  # (batch, 11)

# =============================================
# Dataset loader for discrete actions
# =============================================
def flatten_dataset(trajectories):
    states, actions, rewards, next_states = [], [], [], []

    for traj in trajectories:
        for t in range(len(traj) - 1):
            s, a, r = traj[t]
            # print(r)
            s2, _, _ = traj[t + 1]

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s2)

    A_idx = action_to_index(actions)

    return (
        torch.tensor(np.array(states), dtype=torch.float32).to(device),
        torch.tensor(A_idx, dtype=torch.long).to(device),
        torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1).to(device),
        torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
    )


# ==========================================
# Offline Q-learning (Fitted Q Iteration)
# ==========================================
import matplotlib.pyplot as plt

# ==========================================
# Modified train_fqi to record loss
# ==========================================
def train_fqi(trajectories, state_dim, gamma=0.99, K=10000, lr=1e-2, step_size=1000, gamma_lr=0.1):
    S, A_idx, R, S2 = flatten_dataset(trajectories)
    S = (S - S.mean()) / S.std()
    S2 = (S2 - S2.mean()) / S2.std()
    S = S.squeeze()
    S2 = S2.squeeze()
    
    qnet = QNet(state_dim, NUM_ACTIONS).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_lr)
    mse = nn.MSELoss()

    loss_history = []

    for k in range(K):
        optimizer.zero_grad()

        with torch.no_grad():
            q_next = qnet(S2)                        
            max_next_q = q_next.max(dim=1, keepdim=True)[0]
            target = R + gamma * max_next_q

        q_pred_all = qnet(S)                         
        q_pred = q_pred_all.gather(1, A_idx.unsqueeze(-1))

        loss = mse(q_pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if k % 20 == 0:
            print(f"Iter {k}: Q Loss = {loss.item():.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

    return qnet, loss_history



# ==========================================
# Policy function (argmax Q)
# ==========================================
def select_action(qnet, state):
    with torch.no_grad():
        q = qnet(state)
        idx = q.argmax().item()
        return ACTION_LIST[idx].item()


# ==========================================
# Training
# ==========================================
# id_list = np.arange(1, 11, 1)
# print(id_list)
# dataset = gather_dataset(id_list, trajectory_length=8)

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

state_dim = 1053

qnet, loss_history = train_fqi(dataset, state_dim)

def plot_loss(loss_history):
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label='Q Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Offline Q-learning Loss')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig('q_learning_loss.png')


plot_loss(loss_history)