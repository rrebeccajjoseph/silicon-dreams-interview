"""Actor on deploy obs, critic on privileged obs. Both carry their own input normaliser."""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        delta = mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + var * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + 1e-5)).clamp(-self.clip, self.clip)


def mlp(inp: int, hidden: tuple[int, ...], out: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(inp, h), nn.ELU()]
        inp = h
    if out is not None:
        layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy. Optionally a GRU so it can carry shape and slip cues over time."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 256, 128), recurrent=False, gru=256):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.recurrent = recurrent
        self.gru_dim = gru if recurrent else 0
        self.enc = mlp(obs_dim, hidden)
        self.gru = nn.GRU(hidden[-1], gru) if recurrent else None
        self.head = nn.Linear(gru if recurrent else hidden[-1], act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        nn.init.zeros_(self.head.bias)
        self.head.weight.data *= 0.01

    def forward(self, obs: torch.Tensor, h: torch.Tensor | None = None, dones: torch.Tensor | None = None):
        """obs [T,B,D] -> mean [T,B,A]. Hidden state is reset wherever dones[t] is set."""
        x = self.enc(self.norm(obs))
        if self.recurrent:
            outs = []
            for t in range(x.shape[0]):
                if dones is not None and t > 0:
                    h = h * (1.0 - dones[t - 1]).view(1, -1, 1)
                o, h = self.gru(x[t:t + 1], h)
                outs.append(o)
            x = torch.cat(outs, 0)
        return self.head(x), h

    def init_hidden(self, n: int, device) -> torch.Tensor | None:
        return torch.zeros(1, n, self.gru_dim, device=device) if self.recurrent else None

    def dist(self, mean: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(512, 256, 128)):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(obs)).squeeze(-1)
