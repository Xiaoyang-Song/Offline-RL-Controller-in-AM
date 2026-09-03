"""
baseline_surrogate/lstm/model.py
-------------------------------------
Baseline 2: recurrent (LSTM) surrogate. Tests whether carrying a learned
hidden state across the 12-layer build (sequence memory) beats an explicit
per-step Markov transition — the main surrogate (and every other baseline
here) only ever conditions on (s_t, a_t, cool_t); the LSTM additionally
gets whatever it can learn to remember about earlier layers.

    h_t, c_t   = LSTMCell([s_t, a_t, cool_t, layer_embed], h_{t-1}, c_{t-1})
    s_{t+1}    = s_t + Linear(h_t)

Exposed as an explicit per-step `.step(...)` (not a batched nn.LSTM over
the whole sequence) so the SAME call can be used for teacher-forced
training (see train.py's masked_mse_loss_fn) and for common/eval.py's
per-layer evaluator, which calls predict_fn one layer at a time and needs
somewhere to carry hidden state between calls — see LSTMPredictor below.
"""

import torch
import torch.nn as nn


class LSTMSurrogate(nn.Module):
    def __init__(
        self,
        state_dim:       int   = 1053,
        hidden:          int   = 512,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden
        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        in_dim = state_dim + 1 + 1 + layer_embed_dim
        self.cell    = nn.LSTMCell(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden, state_dim)

        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

    def init_hidden(self, batch_size: int, device: str):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def step(
        self,
        s:         torch.Tensor,  # (B, state_dim) normalised, TRUE s_t (teacher-forced)
        a:         torch.Tensor,  # (B, 1)         normalised
        cool:      torch.Tensor,  # (B, 1)         normalised
        layer_idx: torch.Tensor,  # (B,) int64
        hc,                       # (h, c) from the previous step, or init_hidden(...)
    ):
        e = self.layer_embed(layer_idx)
        x = torch.cat([s, a, cool, e], dim=-1)
        h, c = self.cell(x, hc)
        delta = self.head(self.dropout(h))
        return s + delta, (h, c)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMPredictor:
    """Stateful wrapper adapting LSTMSurrogate.step to common/eval.py's
    predict_fn(s, a, c, layer_idx) -> s2_pred interface, which calls the
    model one layer at a time with no built-in place to carry hidden
    state — on_batch_start resets it once per trajectory batch."""

    def __init__(self, model: LSTMSurrogate, device: str):
        self.model  = model
        self.device = device
        self.hc     = None

    def on_batch_start(self, batch_size: int) -> None:
        self.hc = self.model.init_hidden(batch_size, self.device)

    def predict(self, s, a, c, layer_idx) -> torch.Tensor:
        s2_pred, self.hc = self.model.step(s, a, c, layer_idx, self.hc)
        return s2_pred
