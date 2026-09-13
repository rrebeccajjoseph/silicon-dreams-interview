"""Observation layouts. `DEPLOY` is the only thing a deployed policy may read."""

from __future__ import annotations

import numpy as np


class Layout:
    def __init__(self, blocks: list[tuple[str, int]]):
        self.blocks = blocks
        self.slices: dict[str, slice] = {}
        i = 0
        for name, n in blocks:
            self.slices[name] = slice(i, i + n)
            i += n
        self.dim = i

    def __getitem__(self, name: str) -> slice:
        return self.slices[name]

    def pack(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        for name, sl in self.slices.items():
            out[sl] = parts[name]
        return out


N_PADS = 5
TACTILE_DIM = N_PADS * 5

# real-hardware signals only: proprio, action history, tactile, kinematics,
# a gated vision stand-in, and the target once it has been revealed
DEPLOY = Layout([
    ("arm_q", 7), ("arm_dq", 7), ("hand_q", 16), ("hand_dq", 16),
    ("prev_action", 23), ("ctrl_target", 23),
    ("tactile", TACTILE_DIM),
    ("palm_world", 9),        # palm origin, normal, finger direction (from kinematics)
    ("vision", 12),           # obj pos (3), axis outer (6), r, h, visible flag
    ("target", 10),           # pos (3), axis outer (6), available flag
    ("skill", 12),            # one-hot primitive id, plan C only
    ("time", 1),
])

# everything above plus simulator state. Critics and teachers only.
PRIV = Layout(DEPLOY.blocks + [
    ("obj_pos", 3), ("obj_axis", 6), ("obj_linvel", 3), ("obj_angvel", 3),
    ("obj_params", 5),        # r, h, mass, mu_obj, mu_ground
    ("contacts", 4),          # hand, ground, arm, other
    ("true_target", 9),       # visible to the critic before lift too
    ("lifted", 1),
])
