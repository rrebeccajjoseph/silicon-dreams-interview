"""Plan C grasp: no learning. IK to a canonical pre-grasp, descend, wrap, lift, then
roll the wrist to the palm-up configuration the skills were trained in. Tall standing
rods are tipped over first, using the ground while that is still allowed.

Reads only the deploy observation. Robot kinematics are known by construction.

Measured on lying cylinders this wraps and holds roughly one object in three. The usual
failure is a bounce: the close squeezes the object off the floor for an instant, which
counts as lift, and the next ground contact is a violation. That is the strongest
argument in this repo for learning the grasp (plans A and B)."""

from __future__ import annotations

import mujoco
import numpy as np

from .. import ik
from ..config import Config
from ..observations import DEPLOY
from ..scene import Scene, _rot

# best of a small sweep over offsets and curls, see the report
X_OFFSET = 0.035      # object sits this far toward the fingertips from the palm origin
Z_CLEAR = 0.03        # palm face height above the object top before closing
CLOSE_RAMP = 25       # steps to ramp the finger targets
GRASP_CURL = np.array([1.1, 0.0, 1.6, 1.2] * 3 + [1.3, 0.3, 1.0, 0.8])
ROLL_SPEED = 0.3      # fraction of max arm speed for the wrist roll


def axis_from_outer(o: np.ndarray) -> np.ndarray:
    M = np.array([[o[0], o[3], o[4]], [o[3], o[1], o[5]], [o[4], o[5], o[2]]])
    return np.linalg.eigh(M)[1][:, -1]


class ScriptedGrasp:
    def __init__(self, scene: Scene, cfg: Config):
        self.scene, self.cfg = scene, cfg
        self.d = mujoco.MjData(scene.model)  # kinematics only
        self.lo, self.hi = scene.model.actuator_ctrlrange.T
        self.reset()

    def reset(self) -> None:
        self.phase = "observe"
        self.timer = 0
        self.q_arm_goal = None
        self.hand_goal = self.scene.hand_open.copy()
        self.hand_from = self.scene.hand_open.copy()
        self.obj_world = None
        self.arm_speed = 1.0

    # ------------------------------------------------------------ obs decode
    def _decode(self, obs: np.ndarray) -> None:
        self.arm_q = obs[DEPLOY["arm_q"]].astype(np.float64)
        tgt = obs[DEPLOY["ctrl_target"]].astype(np.float64)
        self.targets = self.lo + (tgt + 1) * (self.hi - self.lo) / 2
        pw = obs[DEPLOY["palm_world"]]
        p_palm, R_palm = pw[:3], _rot(pw[6:9].astype(np.float64), pw[3:6].astype(np.float64))
        v = obs[DEPLOY["vision"]]
        if v[-1] > 0.5:
            p_obj = p_palm + R_palm @ v[:3]
            a_obj = R_palm @ axis_from_outer(v[3:9])
            self.obj_world = (p_obj.astype(np.float64), a_obj / np.linalg.norm(a_obj), float(v[9]), float(v[10]))

    def _arm_ik(self, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
        self.d.qpos[self.scene.arm_qadr] = self.arm_q
        q, _ = ik.solve_ik(self.scene.model, self.d, self.scene.ids.palm_site,
                           self.scene.ids.arm_dofs, pos, R, self.arm_q, iters=100)
        return q

    def _action(self) -> np.ndarray:
        """Delta-target action that moves the current targets toward the goals."""
        q_goal = self.targets[:7] if self.q_arm_goal is None else self.q_arm_goal
        a = np.zeros(23)
        a[:7] = self.arm_speed * (q_goal - self.targets[:7]) / self.cfg.control.arm_step
        a[7:] = (self.hand_goal - self.targets[7:]) / self.cfg.control.hand_step
        return np.clip(a, -1, 1)

    def _ramp_hand(self, goal: np.ndarray, frac: float) -> None:
        self.hand_goal = self.hand_from + (goal - self.hand_from) * min(1.0, frac)

    def _arrived(self, tol: float = 0.02) -> bool:
        return self.q_arm_goal is not None and np.abs(self.arm_q - self.q_arm_goal).max() < tol

    # --------------------------------------------------------------- policy
    def act(self, obs: np.ndarray) -> np.ndarray:
        self._decode(obs)
        self.timer += 1
        if self.obj_world is None:
            return np.zeros(23)
        p, a, r, h = self.obj_world
        standing = abs(a[2]) > 0.7
        alpha = h / (2 * r)

        if self.phase == "observe":
            if standing and alpha > 1.2:
                self.phase = "tip_approach"
            else:
                self.phase = "pregrasp"
            self.timer = 0

        if self.phase == "tip_approach":
            # palm edge beside the rod, a bit above its centre of mass
            side = np.array([0.0, 1.0, 0.0])
            self.q_arm_goal = self._arm_ik(p - side * (r + 0.08) + [0, 0, 0.7 * h / 2], _rot(np.array([1.0, 0, 0]), np.array([0, 0, -1.0])))
            if self._arrived() or self.timer > 60:
                self.phase, self.timer = "tip_push", 0
        elif self.phase == "tip_push":
            side = np.array([0.0, 1.0, 0.0])
            self.q_arm_goal = self._arm_ik(p + side * (r + 0.02) + [0, 0, 0.7 * h / 2], _rot(np.array([1.0, 0, 0]), np.array([0, 0, -1.0])))
            if self._arrived() or self.timer > 60:
                self.phase, self.timer = "retreat", 0
        elif self.phase == "retreat":
            self.q_arm_goal = self.scene.q_home
            if self._arrived(0.05) or self.timer > 60:
                self.phase, self.timer = "observe", 0
        elif self.phase == "pregrasp":
            self.R_grasp = self._grasp_frame(a, standing)
            top = p[2] + (h / 2 if standing else r)
            self.base = p - self.R_grasp[:, 0] * X_OFFSET
            self.z_close = top + Z_CLEAR
            self.q_arm_goal = self._arm_ik(np.array([*self.base[:2], self.z_close + 0.10]), self.R_grasp)
            self.hand_goal = self.scene.hand_open.copy()
            if self._arrived() or self.timer > 80:
                self.phase, self.timer = "descend", 0
        elif self.phase == "descend":
            self.q_arm_goal = self._arm_ik(np.array([*self.base[:2], self.z_close]), self.R_grasp)
            if self._arrived(0.01) or self.timer > 60:
                self.phase, self.timer = "close", 0
                self.hand_from = self.targets[7:].copy()
        elif self.phase == "close":
            self._ramp_hand(GRASP_CURL, self.timer / CLOSE_RAMP)
            if self.timer > CLOSE_RAMP + 10:
                self.phase, self.timer = "lift", 0
        elif self.phase == "lift":
            z = self.z_close + min(0.15, 0.004 * self.timer)  # 8 cm/s
            self.q_arm_goal = self._arm_ik(np.array([*self.base[:2], z]), self.R_grasp)
            if self.timer > 50:
                self.phase, self.timer = "roll", 0
        elif self.phase == "roll":
            # slow joint-space move to the palm-up hold configuration
            self.arm_speed = ROLL_SPEED
            self.q_arm_goal = self.scene.q_hold
        return self._action()

    @staticmethod
    def _grasp_frame(a: np.ndarray, standing: bool) -> np.ndarray:
        """Palm down. For a lying object the fingers curl across the axis."""
        down = np.array([0.0, 0.0, -1.0])
        if standing:
            x = np.array([1.0, 0.0, 0.0])
        else:
            x = np.cross(a, down)
            x /= np.linalg.norm(x)
            if x[0] < 0:
                x = -x
        return _rot(x, down)
