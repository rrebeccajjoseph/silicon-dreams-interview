"""Plan C grasp: no learning. IK to a canonical pre-grasp, descend, wrap, lift, then
roll the wrist to the palm-up configuration the skills were trained in. Tall standing
rods are tipped over first, using the ground while that is still allowed.

Reads only the deploy observation. Robot kinematics are known by construction.

The usual failure is a pop: the closing fingers lever the rod a millimetre or two off the floor,
which counts as lift, and it settles back (a ground violation). It is easy to miss: sampled every
40 ms this script looked like 55 % handover; checked every physics step with the URDF torque limits
it was 3-9 %. What recovered ~48 %: a safe-height approach (a straight joint move from home swept
the fingers through the rod), pressing the palm into the rod, a 4 s close, catching a pop by rising
the moment the lift flag flips, and a fast lift. The remaining failures are the rod slipping back
onto the floor during the lift, which is the strongest argument for a learned grasp."""

from __future__ import annotations

import mujoco
import numpy as np

from .. import ik
from ..config import Config
from ..observations import DEPLOY
from ..scene import Scene, _rot

# best of a sweep on lying rods (r 2.2-2.8 cm, h 8-12 cm), 24 episodes per setting. Pressing the
# palm into the rod keeps it pinned while the fingers wrap; a gap lets the close pop it off the floor
X_OFFSET = 0.05       # object sits this far toward the fingertips from the palm origin
Z_CLEAR = -0.02       # palm face target relative to the object top before closing (negative = press)
CLOSE_RAMP = 100      # steps to ramp the finger targets (4 s: a fast close flicks the rod off the floor)
GRASP_CURL = np.array([1.1, 0.0, 1.6, 1.2] * 3 + [1.3, 0.3, 1.0, 0.8])
ROLL_SPEED = 0.3      # fraction of max arm speed for the wrist roll
SAFE_Z = 0.25         # palm height for the approach before dropping to the pregrasp


def axis_from_outer(o: np.ndarray) -> np.ndarray:
    M = np.array([[o[0], o[3], o[4]], [o[3], o[1], o[5]], [o[4], o[5], o[2]]])
    return np.linalg.eigh(M)[1][:, -1]


class ScriptedGrasp:
    x_offset, z_clear, close_ramp, roll_speed = X_OFFSET, Z_CLEAR, CLOSE_RAMP, ROLL_SPEED
    lift_while_closing = 0.0  # m/step the palm rises during the close, so a squeeze-pop keeps going up
    # closing everything at once lets the thumb flick the rod off the floor before the fingers
    # arrive; closing one side first gives the other side a wall to push against
    close_order = "together"  # or "fingers_first", "thumb_first"
    # a finger whose target runs this far ahead of its joint angle is blocked by the object or the
    # floor; its target stops advancing, so the grasp squeezes instead of flicking. None = off
    stall_err: float | None = None
    tip_alpha = 1.2  # standing objects taller than this are tipped onto their side first
    curl_extra = 0.0  # added to the finger flexion joints (not abduction) of GRASP_CURL
    lift_speed = 0.012  # m per control step; slower lifts let the rod slip back onto the floor
    # the first part of the lift is slow so the rod leaves the floor quasi-statically instead of
    # springing off the palm press
    lift_slow_m, lift_slow_speed = 0.0, 0.001
    # if the close pops the rod off the floor (the lift flag flips mid-close), finish the close and
    # rise at this speed so it is caught instead of falling back. Uses only the flag, never T*
    catch_speed: float | None = 0.012

    def __init__(self, scene: Scene, cfg: Config, **overrides):
        for k, v in overrides.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        self.scene, self.cfg = scene, cfg
        self.curl = GRASP_CURL + self.curl_extra * np.array([1, 0, 1, 1] * 3 + [0, 0, 1, 1])
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
        self.rise = self.lift_speed
        self._high_done = False

    # ------------------------------------------------------------ obs decode
    def _decode(self, obs: np.ndarray) -> None:
        self.arm_q = obs[DEPLOY["arm_q"]].astype(np.float64)
        self.hand_q = obs[DEPLOY["hand_q"]].astype(np.float64)
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
            if standing and alpha > self.tip_alpha:
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
            self.base = p - self.R_grasp[:, 0] * self.x_offset
            self.z_close = top + self.z_clear
            # a straight joint-space move from home dips the fingers into the object, so pass high first
            above = self.timer < 40 and not getattr(self, "_high_done", False)
            z = max(self.z_close + 0.10, SAFE_Z) if above else self.z_close + 0.10
            self.q_arm_goal = self._arm_ik(np.array([*self.base[:2], z]), self.R_grasp)
            if above and self._arrived(0.03):
                self._high_done = True
            self.hand_goal = self.scene.hand_open.copy()
            if self._arrived() or self.timer > 80:
                self.phase, self.timer = "descend", 0
        elif self.phase == "descend":
            self.q_arm_goal = self._arm_ik(np.array([*self.base[:2], self.z_close]), self.R_grasp)
            if self._arrived(0.01) or self.timer > 60:
                self.phase, self.timer = "close", 0
                self.hand_from = self.targets[7:].copy()
        elif self.phase == "close" and self.catch_speed and obs[DEPLOY["target"]][-1] > 0.5:
            self.hand_goal = self.curl.copy()
            self.phase, self.timer, self.rise = "lift", 0, self.catch_speed
            self.z_close = self.scene_palm_z(obs)
        elif self.phase == "close":
            if self.close_order == "together":
                self._ramp_hand(self.curl, self.timer / self.close_ramp)
            else:
                first = slice(0, 12) if self.close_order == "fingers_first" else slice(12, 16)
                f1 = min(1.0, self.timer / self.close_ramp)
                f2 = min(1.0, max(0.0, self.timer / self.close_ramp - 1.0))
                frac = np.full(16, f2)
                frac[first] = f1
                self.hand_goal = self.hand_from + (self.curl - self.hand_from) * frac
            if self.stall_err is not None:
                closing = np.sign(self.curl - self.hand_from)
                cap = self.hand_q + closing * self.stall_err
                self.hand_goal = np.where(closing > 0, np.minimum(self.hand_goal, cap), np.maximum(self.hand_goal, cap))
            if self.lift_while_closing:
                self.z_close += self.lift_while_closing
                self.q_arm_goal = self._arm_ik(np.array([*self.base[:2], self.z_close]), self.R_grasp)
            if self.timer > (1 if self.close_order == "together" else 2) * self.close_ramp + 10:
                self.phase, self.timer = "lift", 0
        elif self.phase == "lift":
            t_slow = self.lift_slow_m / self.lift_slow_speed
            if self.rise == self.lift_speed and self.timer < t_slow:
                dz = self.lift_slow_speed * self.timer
            else:
                t0 = t_slow if self.rise == self.lift_speed else 0.0
                dz = (self.lift_slow_m if t0 else 0.0) + self.rise * (self.timer - t0)
            z = self.z_close + min(0.15, dz)
            self.q_arm_goal = self._arm_ik(np.array([*self.base[:2], z]), self.R_grasp)
            if self.timer > 50 + self.lift_slow_m / self.lift_slow_speed:
                self.phase, self.timer = "roll", 0
        elif self.phase == "roll":
            # slow joint-space move to the palm-up hold configuration
            self.arm_speed = self.roll_speed
            self.q_arm_goal = self.scene.q_hold
        return self._action()

    @staticmethod
    def scene_palm_z(obs: np.ndarray) -> float:
        return float(obs[DEPLOY["palm_world"]][2])

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
