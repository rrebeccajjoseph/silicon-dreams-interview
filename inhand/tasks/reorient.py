"""Shared reorient stage: object already in hand, target known from t=0."""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..env import CylinderEnv
from ..scene import Scene

# the target lives in the palm frame, so the arm holds the palm-up pose and only the fingers act.
# Full-scale finger noise threw the object out before anything was learned
ACTION_SCALE = np.concatenate([np.zeros(7), np.full(16, 0.4)])


class ReorientEnv(CylinderEnv):
    def __init__(self, scene: Scene, cfg: Config, seed: int = 0, curriculum=None, horizon_s: float = 10.0, **kw):
        super().__init__(scene, cfg, mode="inhand", seed=seed, curriculum=curriculum, horizon_s=horizon_s, **kw)

    def step(self, action: np.ndarray):
        return super().step(np.clip(np.asarray(action, dtype=np.float64), -1, 1) * ACTION_SCALE)
