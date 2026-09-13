"""Shared reorient stage: object already in hand, target known from t=0."""

from __future__ import annotations

from ..config import Config
from ..env import CylinderEnv
from ..scene import Scene


class ReorientEnv(CylinderEnv):
    def __init__(self, scene: Scene, cfg: Config, seed: int = 0, curriculum=None, horizon_s: float = 10.0, **kw):
        super().__init__(scene, cfg, mode="inhand", seed=seed, curriculum=curriculum, horizon_s=horizon_s, **kw)
