"""Plan A grasp stage. The episode ends shortly after lift and the terminal reward is
the reorient critic's value at the lift state, averaged over targets sampled from the
prior. That is the expected downstream success given the target is still unknown."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .. import frames
from ..config import Config
from ..env import CylinderEnv, Signals
from ..observations import PRIV
from ..scene import Scene

ValueFn = Callable[[np.ndarray], np.ndarray]  # priv obs [K, Dp] -> values [K]


class GraspEnv(CylinderEnv):
    def __init__(self, scene: Scene, cfg: Config, value_fn: ValueFn | None = None, seed: int = 0,
                 settle_steps: int = 10, n_targets: int = 16, value_scale: float = 1.0, **kw):
        super().__init__(scene, cfg, mode="ground", seed=seed, horizon_s=8.0, **kw)
        self.value_fn = value_fn
        self.settle_steps = settle_steps
        self.n_targets = n_targets
        self.value_scale = value_scale
        self.held = 0

    def _on_reset(self) -> None:
        self.held = 0

    def _task_done(self, s: Signals) -> tuple[bool, float]:
        self.held = self.held + 1 if (self.ep.lifted and self.contacts.hand_only) else 0
        if self.held < self.settle_steps:
            return False, 0.0
        return True, self.value_scale * self.marginal_value()

    def marginal_value(self) -> float:
        """E_{T* ~ prior}[ V_reorient(lift state, T*) ]."""
        if self.value_fn is None:
            return 0.0
        base = self._obs()["priv"]
        batch = np.repeat(base[None], self.n_targets, 0)
        for k in range(self.n_targets):
            p, a = self._sample_target(None)
            batch[k, PRIV["true_target"]] = np.concatenate([p, frames.axis_outer(a)])
        return float(np.mean(self.value_fn(batch)))
