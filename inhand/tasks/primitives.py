"""Plan C skills: short fixed-increment reorientations, one policy conditioned on a
skill token. The planner composes them at deploy time."""

from __future__ import annotations

import numpy as np

from .. import frames
from ..config import Config
from ..scene import Scene
from .reorient import ReorientEnv

ROT = np.deg2rad(25.0)
SHIFT = 0.015
X, Y, Z = np.eye(3)
# (kind, axis in the palm frame, magnitude). The last one is "hold still".
PRIMITIVES: list[tuple[str, np.ndarray, float]] = [
    ("rot", X, ROT), ("rot", X, -ROT), ("rot", Y, ROT), ("rot", Y, -ROT), ("rot", Z, ROT), ("rot", Z, -ROT),
    ("shift", X, SHIFT), ("shift", X, -SHIFT), ("shift", Y, SHIFT), ("shift", Y, -SHIFT),
    ("hold", Z, 0.0),
]
N_PRIMITIVES = len(PRIMITIVES)
HOLD = N_PRIMITIVES - 1
SKILL_DIM = 12


def apply_primitive(k: int, p: np.ndarray, a: np.ndarray, r: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Nominal effect on a palm-frame pose. Rotations are about the object centre;
    the centre is lifted if the new orientation would otherwise cut into the palm."""
    kind, axis, mag = PRIMITIVES[k]
    a2, p2 = a.copy(), p.copy()
    if kind == "rot":
        a2 = frames.rotvec_to_mat(axis * mag) @ a
    elif kind == "shift":
        p2 = p + axis * mag
    p2[2] = max(p2[2], frames.z_clearance(a2, r, h) + 0.002)
    return p2, a2


class PrimitiveEnv(ReorientEnv):
    def __init__(self, scene: Scene, cfg: Config, seed: int = 0, skill: int | None = None, **kw):
        super().__init__(scene, cfg, seed=seed, horizon_s=2.5, **kw)
        self.fixed_skill = skill
        self.hold_steps = 10  # a skill only has to land and settle, not hold for 2 s

    def _on_reset(self) -> None:
        k = self.fixed_skill if self.fixed_skill is not None else int(self.rng.integers(N_PRIMITIVES))
        self.skill = np.zeros(SKILL_DIM, dtype=np.float32)
        self.skill[k] = 1.0
        self.set_target(*apply_primitive(k, self.obj_p, self.obj_a, self.obj.r, self.obj.h))
