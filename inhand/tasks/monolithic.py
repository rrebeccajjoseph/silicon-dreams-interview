"""Plan B: the whole episode in one env. The target block of the deploy obs is zero
until lift; the critic reads `true_target` throughout."""

from __future__ import annotations

from ..config import Config
from ..env import CylinderEnv
from ..scene import Scene


class FullTaskEnv(CylinderEnv):
    def __init__(self, scene: Scene, cfg: Config, seed: int = 0, **kw):
        super().__init__(scene, cfg, mode="ground", seed=seed, **kw)
