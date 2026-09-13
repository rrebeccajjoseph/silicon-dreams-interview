"""Tactile + proprio state estimator for plan C. A GRU regresses the object's palm-frame
pose and size from the deploy observation stream. Trained supervised in sim."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..observations import DEPLOY
from ..rl.nets import RunningNorm, mlp

OUT_DIM = 11  # pos (3), axis outer (6), r, h


class Estimator(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.norm = RunningNorm(DEPLOY.dim)
        self.enc = mlp(DEPLOY.dim, (256,))
        self.gru = nn.GRU(256, hidden)
        self.head = nn.Linear(hidden, OUT_DIM)
        self.hidden = hidden

    def forward(self, obs: torch.Tensor, h: torch.Tensor | None = None, dones: torch.Tensor | None = None):
        obs = obs.clone()
        obs[..., DEPLOY["target"]] = 0  # the estimate must not depend on the goal
        x = self.enc(self.norm(obs))
        outs = []
        for t in range(x.shape[0]):
            if dones is not None and t > 0:
                h = h * (1.0 - dones[t - 1]).view(1, -1, 1)
            o, h = self.gru(x[t:t + 1], h)
            outs.append(o)
        return self.head(torch.cat(outs, 0)), h

    def init_hidden(self, n: int, device) -> torch.Tensor:
        return torch.zeros(1, n, self.hidden, device=device)


def decode(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Network output -> (pos, unit axis, r, h). The axis is the top eigenvector of
    the predicted outer product, so the sign ambiguity never reaches the planner."""
    M = np.array([[y[3], y[6], y[7]], [y[6], y[4], y[8]], [y[7], y[8], y[5]]])
    w, v = np.linalg.eigh(M)
    return y[:3].copy(), v[:, -1], float(y[9]), float(y[10])


def labels_from_priv(obs_p: np.ndarray) -> np.ndarray:
    from ..observations import PRIV
    return np.concatenate([obs_p[..., PRIV["obj_pos"]], obs_p[..., PRIV["obj_axis"]],
                           obs_p[..., PRIV["obj_params"]][..., :2]], -1)


def train_estimator(est: Estimator, obs_d: np.ndarray, obs_p: np.ndarray, done: np.ndarray,
                    epochs: int, device: str = "cpu", lr: float = 1e-3, batch_rows: int = 32) -> float:
    """obs arrays are [T, N, D] chunks from any policy's rollouts."""
    est.to(device)
    opt = torch.optim.Adam(est.parameters(), lr=lr)
    est.norm.update(torch.as_tensor(obs_d, device=device))
    y = torch.as_tensor(labels_from_priv(obs_p), device=device)
    x = torch.as_tensor(obs_d, device=device)
    d = torch.as_tensor(done, device=device)
    N = x.shape[1]
    last = 0.0
    for _ in range(epochs):
        for mb in torch.randperm(N).chunk(max(1, N // batch_rows)):
            pred, _ = est(x[:, mb], est.init_hidden(len(mb), device), d[:, mb])
            loss = F.mse_loss(pred, y[:, mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
    return last
