"""
baseline_surrogate/common/train_loop.py
------------------------------------------
Shared early-stopping training loop for the gradient-trained baselines
(mlp, lstm, vanilla_ensemble, ablation_no_two_stage, ablation_no_latent —
everything except kalman_filter, which is fit in closed form).
"""

import time
from typing import Callable, Optional

import torch


def run_training(
    model:      torch.nn.Module,
    train_loader,
    val_loader,
    loss_fn:    Callable[[torch.nn.Module, tuple, str], torch.Tensor],
    optimizer:  torch.optim.Optimizer,
    device:     str,
    epochs:     int = 300,
    patience:   int = 20,
    scheduler=None,
    save_best_fn: Optional[Callable[[int, float], None]] = None,
    log_prefix: str = "",
):
    best_val   = float("inf")
    no_improve = 0
    train_hist, val_hist = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tr_loss += loss.item()
            n += 1
        tr_loss /= max(n, 1)

        model.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                va_loss += loss_fn(model, batch, device).item()
                n += 1
        va_loss /= max(n, 1)

        if scheduler is not None:
            scheduler.step()

        train_hist.append(tr_loss)
        val_hist.append(va_loss)

        improved = va_loss < best_val
        if improved:
            best_val, no_improve = va_loss, 0
            if save_best_fn is not None:
                save_best_fn(epoch, best_val)
            marker = " ✓ best"
        else:
            no_improve += 1
            marker = f" (no improvement {no_improve}/{patience})"

        elapsed = time.time() - t0
        print(f"{log_prefix}Epoch {epoch:4d}/{epochs} | train {tr_loss:.5f} | "
              f"val {va_loss:.5f} | {elapsed:6.1f}s{marker}")

        if no_improve >= patience:
            print(f"{log_prefix}Early stopping at epoch {epoch}.")
            break

    return train_hist, val_hist, best_val
