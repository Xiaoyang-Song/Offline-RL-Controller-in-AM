"""
online_RL/replay_buffer.py
--------------------------
Circular (FIFO) experience replay buffer for the online DQN agent.

Stores  (state, action, reward, next_state, done)  tuples and returns
uniform random mini-batches for gradient updates.

Design notes
------------
  • States are stored as float32 numpy arrays (normalised, CPU) to minimise
    memory footprint; they are converted to tensors inside the agent.
  • The buffer pre-allocates contiguous numpy arrays when `capacity` is known
    upfront, which is faster than a Python deque for large buffers.
"""

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity FIFO experience replay buffer with pre-allocated arrays.

    Parameters
    ----------
    capacity  : int  maximum number of transitions
    state_dim : int  dimension of each state vector
    """

    def __init__(self, capacity: int = 100_000, state_dim: int = 1053) -> None:
        self.capacity  = int(capacity)
        self.state_dim = int(state_dim)

        # Pre-allocate arrays for speed
        self._states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int64)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)

        self._ptr  = 0    # write pointer
        self._size = 0    # current number of stored transitions

    # ------------------------------------------------------------------
    def push(
        self,
        state:      np.ndarray,   # (D,) normalised float32
        action:     int,
        reward:     float,
        next_state: np.ndarray,   # (D,) normalised float32
        done:       bool,
    ) -> None:
        """Store one transition.  Overwrites oldest entry when full."""
        idx = self._ptr % self.capacity
        self._states[idx]      = state
        self._next_states[idx] = next_state
        self._actions[idx]     = int(action)
        self._rewards[idx]     = float(reward)
        self._dones[idx]       = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int):
        """
        Sample a uniformly random mini-batch.

        Returns
        -------
        states      : (B, D) float32
        actions     : (B,)   int64
        rewards     : (B,)   float32
        next_states : (B, D) float32
        dones       : (B,)   float32  (1.0 = terminal)
        """
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only has {self._size} samples."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity:,}, "
            f"size={self._size:,}, state_dim={self.state_dim})"
        )
