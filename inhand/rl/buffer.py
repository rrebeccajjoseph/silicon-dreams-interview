"""Fixed-length rollouts as [T, N, ...] arrays. Log-probs and values are recomputed at
update time from a frozen snapshot, which is what makes relabeled rows cheap to add."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass
class Rollout:
    obs_d: np.ndarray      # [T, N, Dd]
    obs_p: np.ndarray      # [T, N, Dp]
    act: np.ndarray        # [T, N, A]
    rew: np.ndarray        # [T, N]
    done: np.ndarray       # [T, N]
    h0: np.ndarray         # [N, G] actor hidden at t=0 (empty if not recurrent)
    last_obs_p: np.ndarray  # [N, Dp] for bootstrapping
    signals: np.ndarray    # [T, N, 10] see env.Signals
    obj_palm: np.ndarray   # [T, N, 6] object pos + axis in the palm frame
    hand_only: np.ndarray  # [T, N]
    relabeled: np.ndarray  # [N] bool, rows added by hindsight relabeling

    @property
    def T(self) -> int:
        return self.rew.shape[0]

    @property
    def N(self) -> int:
        return self.rew.shape[1]

    def rows(self, idx: np.ndarray) -> "Rollout":
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v[idx] if f.name in ("h0", "last_obs_p", "relabeled") else v[:, idx]
        return Rollout(**out)

    @staticmethod
    def concat(a: "Rollout", b: "Rollout") -> "Rollout":
        out = {}
        for f in fields(a):
            ax = 0 if f.name in ("h0", "last_obs_p", "relabeled") else 1
            out[f.name] = np.concatenate([getattr(a, f.name), getattr(b, f.name)], axis=ax)
        return Rollout(**out)
