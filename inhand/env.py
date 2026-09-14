"""Core episode logic shared by every plan. Tasks subclass and override small hooks."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import frames
from .config import Config, Envelope, RewardWeights
from .contacts import ContactState, classify, tactile, visible_fraction
from .cylinder import (ObjectParams, apply_object, cylinder_surface_points, quat_from_z_to,
                       sample_object, sample_start_pose)
from .observations import DEPLOY, PRIV
from .scene import Scene

# fingers wrapped, used to seat the object for in-hand resets
HAND_CURL = np.array([0.9, 0.0, 0.9, 0.6] * 3 + [1.3, 0.3, 0.9, 0.6])
DROP_STEPS = 5  # consecutive steps with no hand contact after lift counts as a drop


@dataclass
class Signals:
    """Everything the reward depends on. Kept explicit so relabeling can recompute it."""
    epos: float
    eang: float
    in_tol: bool
    lifted: bool
    lifted_now: bool
    dropped: bool
    violation: bool
    success: bool
    action_sq: float
    palm_obj_dist: float

    def as_array(self) -> np.ndarray:
        return np.array([self.epos, self.eang, self.in_tol, self.lifted, self.lifted_now,
                         self.dropped, self.violation, self.success, self.action_sq,
                         self.palm_obj_dist], dtype=np.float32)


def reward_fn(w: RewardWeights, s: Signals) -> float:
    r = -w.action * s.action_sq
    if s.lifted:
        r += w.pos * np.exp(-s.epos / w.pos_scale) + w.ang * np.exp(-s.eang / w.ang_scale) + w.in_tol * float(s.in_tol)
    else:
        r += w.approach * np.exp(-s.palm_obj_dist / w.approach_scale)  # target-free by construction
    r += w.lift * float(s.lifted_now) + w.success * float(s.success)
    r += w.drop * float(s.dropped) + w.violation * float(s.violation)
    return float(r)


@dataclass
class EpisodeState:
    t: int = 0
    lifted: bool = False
    lift_step: int = -1
    hold_count: int = 0
    no_contact: int = 0
    arm_frozen: bool = False
    done_reason: str = ""
    start_pose: str = ""
    last_vision: np.ndarray = field(default_factory=lambda: np.zeros(11))


class CylinderEnv:
    """One MuJoCo world. mode='ground' starts on the floor, mode='inhand' starts held."""

    def __init__(self, scene: Scene, cfg: Config, mode: str = "ground", seed: int = 0,
                 envelope: Envelope | None = None, curriculum: tuple[float, float] | None = None,
                 horizon_s: float | None = None, bank_path: str | None = None, bank_frac: float = 0.0):
        self.scene = scene
        self.cfg = cfg
        self.ids = scene.ids
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.envelope = envelope or cfg.envelope
        self.curriculum = curriculum  # (max axis angle, max position offset) from the lift pose
        self.m = scene.new_model()
        self.d = mujoco.MjData(self.m)
        self._scratch = mujoco.MjData(self.m)
        self.horizon = int(round((horizon_s or cfg.success.horizon_s) * cfg.control.ctrl_hz))
        self.cam_every = max(1, int(round(cfg.control.ctrl_hz / cfg.sensors.camera_hz)))
        self.hold_steps = cfg.success.hold_steps(cfg.control)
        self.n_act = self.m.nu
        self.ctrl_lo, self.ctrl_hi = self.m.actuator_ctrlrange.T
        self.step_size = np.concatenate([np.full(7, cfg.control.arm_step), np.full(16, cfg.control.hand_step)])
        self.ext_cams = np.array(cfg.sensors.cameras)
        self.targets = np.zeros(self.n_act)
        self.prev_action = np.zeros(self.n_act)
        self.obj: ObjectParams
        self.target_p = np.zeros(3)
        self.target_a = np.array([0.0, 0.0, 1.0])
        self.skill = np.zeros(12, dtype=np.float32)
        self.ep = EpisodeState()
        # saved handover states from the scripted grasp; in-hand resets draw from them bank_frac of the time
        self.bank = dict(np.load(bank_path)) if bank_path else None
        self.bank_frac = bank_frac

    # ---------------------------------------------------------------- reset
    def reset(self, obj: ObjectParams | None = None, start_pose: str | None = None) -> dict:
        self.ep = EpisodeState()
        self.prev_action[:] = 0
        attempt = 0
        use_bank = self.mode == "inhand" and self.bank is not None and obj is None and self.rng.random() < self.bank_frac
        while use_bank:
            i = self.rng.integers(len(self.bank["qpos"]))
            self.obj = ObjectParams(*[float(v) for v in self.bank["obj"][i]])
            apply_object(self.m, self.ids, self.obj)
            self.pts_obj = cylinder_surface_points(self.obj.r, self.obj.h, self.cfg.sensors.n_surface_pts, self.rng)
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[:] = self.bank["qpos"][i]
            self.d.qvel[:] = self.bank["qvel"][i]
            self.targets[:] = self.bank["targets"][i]
            self.d.ctrl[:] = self.targets
            mujoco.mj_forward(self.m, self.d)
            self.ep.start_pose = "handover"
            if classify(self.m, self.d, self.ids).hand_only:
                break
        while not use_bank:
            # a few shapes refuse to seat in the hand; resample the object every few tries
            # (a pinned object is released after 40 failures rather than crashing a worker)
            if attempt % 10 == 0:
                self.obj = obj if (obj is not None and attempt < 40) else sample_object(self.envelope, self.cfg.friction, self.rng)
                apply_object(self.m, self.ids, self.obj)
                self.pts_obj = cylinder_surface_points(self.obj.r, self.obj.h, self.cfg.sensors.n_surface_pts, self.rng)
            mujoco.mj_resetData(self.m, self.d)
            if self._reset_on_ground(start_pose) if self.mode == "ground" else self._reset_in_hand():
                break
            attempt += 1
        self._update_state()
        self.ep.lifted = self.contacts.hand_only
        self.ep.lift_step = 0 if self.ep.lifted else -1
        ref = (self.obj_p, self.obj_a) if self.ep.lifted else None
        self.target_p, self.target_a = self._sample_target(ref)
        self._on_reset()
        return self._obs()

    def _set_arm(self, q: np.ndarray) -> None:
        self.d.qpos[self.scene.arm_qadr] = q
        self.d.qpos[self.scene.hand_qadr] = self.scene.hand_open
        self.targets[:7] = q
        self.targets[7:] = self.scene.hand_open
        self.d.ctrl[:] = self.targets

    def _set_obj_world(self, pos: np.ndarray, quat: np.ndarray) -> None:
        a = self.ids.obj_qadr
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = quat
        self.d.qvel[self.ids.obj_dofadr:self.ids.obj_dofadr + 6] = 0

    def _reset_on_ground(self, start_pose: str | None) -> bool:
        self._set_arm(self.scene.q_home)
        env = self.envelope if start_pose is None else Envelope(**{**self.envelope.__dict__, "start_poses": (start_pose,)})
        pos, quat, kind = sample_start_pose(env, self.obj, self.rng)
        self.ep.start_pose = kind
        self._set_obj_world(pos, quat)
        for _ in range(100):
            mujoco.mj_step(self.m, self.d)
        return self.d.qpos[self.ids.obj_qadr + 2] > 0.5 * min(self.obj.r, self.obj.h / 2)

    def _reset_in_hand(self) -> bool:
        """Drop the object onto the upturned palm and wrap the fingers around it."""
        self._set_arm(self.scene.q_hold)
        mujoco.mj_forward(self.m, self.d)
        p_palm, R_palm = self.palm_pose()
        # roughly what the scripted grasp hands over: across the fingers, but loosely placed
        a = frames.perturb_axis(np.array(self.cfg.targets.axis_nominal), np.deg2rad(60.0), self.rng)
        p = np.array([self.rng.uniform(-0.05, 0.04), self.rng.uniform(-0.03, 0.03),
                      frames.z_clearance(a, self.obj.r, self.obj.h) + self.rng.uniform(0.002, 0.01)])
        self._set_obj_world(p_palm + R_palm @ p, quat_from_z_to(R_palm @ a))
        self.ep.start_pose = "inhand"
        self.targets[7:] = HAND_CURL
        self.d.ctrl[:] = self.targets
        for _ in range(12 * self.cfg.control.substeps):
            mujoco.mj_step(self.m, self.d)
        st = classify(self.m, self.d, self.ids)
        v = np.linalg.norm(self.d.qvel[self.ids.obj_dofadr:self.ids.obj_dofadr + 3])
        return st.hand_only and v < 0.05

    def _on_reset(self) -> None:
        """Hook for tasks that need to touch the target or the skill token."""

    # --------------------------------------------------------------- target
    def _sample_target(self, ref: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
        tg = self.cfg.targets
        r, h = self.obj.r, self.obj.h
        for _ in range(40):
            if ref is not None and self.curriculum is not None:
                ang, dist = self.curriculum
                a = frames.perturb_axis(ref[1], ang, self.rng)
                p = ref[0] + frames.random_unit(self.rng) * self.rng.uniform(0, dist)
                p[2] = max(p[2], frames.z_clearance(a, r, h) + 0.002)
            else:
                a = frames.perturb_axis(np.array(tg.axis_nominal), tg.axis_cone, self.rng)
                z0 = frames.z_clearance(a, r, h)
                p = np.array([self.rng.uniform(*tg.x_range), self.rng.uniform(*tg.y_range),
                              z0 + self.rng.uniform(0.002, tg.z_clear_max)])
            if np.linalg.norm(p) > tg.max_radius or self._penetrates_palm(p, a):
                continue
            return p, a
        return p, a

    def _penetrates_palm(self, p: np.ndarray, a: np.ndarray) -> bool:
        """Would the object at this palm-frame pose sit inside the palm or wrist?"""
        s = self._scratch
        s.qpos[:] = self.d.qpos
        s.qpos[self.scene.hand_qadr] = self.scene.hand_open
        mujoco.mj_kinematics(self.m, s)
        p_palm = s.site_xpos[self.ids.palm_site]
        R_palm = s.site_xmat[self.ids.palm_site].reshape(3, 3)
        q = self.ids.obj_qadr
        s.qpos[q:q + 3] = p_palm + R_palm @ p
        s.qpos[q + 3:q + 7] = quat_from_z_to(R_palm @ a)
        mujoco.mj_forward(self.m, s)
        palm = set(self.ids.palm_geoms.tolist())
        for i in range(s.ncon):
            c = s.contact[i]
            if self.ids.obj_geom in (c.geom1, c.geom2) and c.dist < -0.002:
                other = c.geom2 if c.geom1 == self.ids.obj_geom else c.geom1
                if other in palm:
                    return True
        return False

    # ----------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        delta = action * self.step_size
        if self.ep.arm_frozen:
            delta[:7] = 0
        self.targets = np.clip(self.targets + delta, self.ctrl_lo, self.ctrl_hi)
        self.d.ctrl[:] = self.targets
        for _ in range(self.cfg.control.substeps):
            mujoco.mj_step(self.m, self.d)
        self.ep.t += 1
        self._update_state()
        sig = self._signals(action)
        reward = reward_fn(self.cfg.reward, sig)
        done, reason = self._termination(sig)
        task_done, bonus = self._task_done(sig)
        if task_done and not done:
            done, reason = True, "task"
        reward += bonus
        self.ep.done_reason = reason if done else ""
        self.prev_action = action
        info = {
            "signals": sig.as_array(), "epos": sig.epos, "eang": sig.eang,
            "lifted": self.ep.lifted, "success": sig.success, "reason": reason,
            "obj": self.obj, "start_pose": self.ep.start_pose, "t": self.ep.t,
            "obj_palm": np.concatenate([self.obj_p, self.obj_a]).astype(np.float32),
            "hand_only": self.contacts.hand_only, "robot_ground": self.contacts.robot_ground,
        }
        return self._obs(), reward, done, info

    def _update_state(self) -> None:
        p_palm, R_palm = self.palm_pose()
        b = self.ids.obj_body
        self.obj_pos_w = self.d.xpos[b].copy()
        R_obj = self.d.xmat[b].reshape(3, 3)
        self.obj_p, self.obj_a = frames.in_palm(p_palm, R_palm, self.obj_pos_w, R_obj)
        self.contacts: ContactState = classify(self.m, self.d, self.ids)
        self.tact = tactile(self.m, self.d, self.ids, self.cfg.sensors.pads)
        if self.ep.t % self.cam_every == 0:  # cameras run slower than the controller
            wrist = self.d.xpos[self.ids.palm_body] + self.d.xmat[self.ids.palm_body].reshape(3, 3) @ self.cfg.sensors.wrist_cam_pos
            cams = np.vstack([self.ext_cams, wrist])
            self.vis_frac = visible_fraction(self.m, self.d, self.ids, self.pts_obj, cams)
        self.R_palm = R_palm
        self.p_palm = p_palm

    def _signals(self, action: np.ndarray) -> Signals:
        c = self.contacts
        lifted_now = False
        if not self.ep.lifted and c.hand_only:
            self.ep.lifted = True
            self.ep.lift_step = self.ep.t
            lifted_now = True
        violation = self.ep.lifted and not lifted_now and (c.obj_ground or c.obj_arm or c.obj_other)
        self.ep.no_contact = self.ep.no_contact + 1 if (self.ep.lifted and not c.obj_hand) else 0
        dropped = self.ep.no_contact >= DROP_STEPS
        epos, eang = frames.errors(self.obj_p, self.obj_a, self.target_p, self.target_a)
        in_tol = self.ep.lifted and c.hand_only and epos <= self.cfg.success.pos_tol and eang <= self.cfg.success.ang_tol
        if in_tol:
            self.ep.hold_count += 1
            self.ep.arm_frozen = True
        else:
            self.ep.hold_count = 0
            self.ep.arm_frozen = False
        success = self.ep.hold_count >= self.hold_steps
        return Signals(
            epos=epos, eang=eang, in_tol=in_tol, lifted=self.ep.lifted, lifted_now=lifted_now,
            dropped=dropped, violation=violation, success=success,
            action_sq=float(np.sum(action**2)),
            palm_obj_dist=float(np.linalg.norm(self.obj_p - [0.02, 0.0, 0.03])),
        )

    def _termination(self, s: Signals) -> tuple[bool, str]:
        c = self.contacts
        if s.success:
            return True, "success"
        if s.violation:
            return True, "violation_ground" if c.obj_ground else ("violation_arm" if c.obj_arm else "violation_other")
        if s.dropped:
            return True, "dropped"
        # the hand touching the floor is allowed (scooping is extrinsic dexterity);
        # only the object's contacts are constrained
        if np.linalg.norm(self.obj_pos_w[:2] - [0.45, 0.0]) > 0.5:
            return True, "obj_lost"
        if self.ep.t >= self.horizon:
            return True, "timeout" if self.ep.lifted else "never_lifted"
        return False, ""

    def _task_done(self, s: Signals) -> tuple[bool, float]:
        """Extra termination and terminal reward. Overridden by tasks."""
        return False, 0.0

    # ------------------------------------------------------------------ obs
    def palm_pose(self) -> tuple[np.ndarray, np.ndarray]:
        sid = self.ids.palm_site
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def _obs(self) -> dict:
        d, ids, sc = self.d, self.ids, self.scene
        gt = np.concatenate([self.obj_p, frames.axis_outer(self.obj_a), [self.obj.r, self.obj.h]])
        if self.vis_frac >= self.cfg.sensors.visible_frac:
            self.ep.last_vision = gt
            vision = np.concatenate([gt, [1.0]])
        else:
            vision = np.concatenate([self.ep.last_vision, [0.0]])
        true_target = np.concatenate([self.target_p, frames.axis_outer(self.target_a)])
        target = np.concatenate([true_target, [1.0]]) if self.ep.lifted else np.zeros(10)
        R = self.R_palm
        deploy = {
            "arm_q": d.qpos[sc.arm_qadr], "arm_dq": d.qvel[ids.arm_dofs],
            "hand_q": d.qpos[sc.hand_qadr], "hand_dq": d.qvel[ids.hand_dofs],
            "prev_action": self.prev_action,
            "ctrl_target": 2 * (self.targets - self.ctrl_lo) / (self.ctrl_hi - self.ctrl_lo) - 1,
            "tactile": self.tact,
            "palm_world": np.concatenate([self.p_palm, R[:, 2], R[:, 0]]),
            "vision": vision, "target": target, "skill": self.skill,
            "time": [self.ep.t / self.horizon],
        }
        v = d.qvel[ids.obj_dofadr:ids.obj_dofadr + 6]
        priv = {
            **deploy,
            "obj_pos": self.obj_p, "obj_axis": frames.axis_outer(self.obj_a),
            "obj_linvel": R.T @ v[:3], "obj_angvel": v[3:],
            "obj_params": [self.obj.r, self.obj.h, self.obj.mass, self.obj.mu_obj, self.obj.mu_ground],
            "contacts": [self.contacts.obj_hand, self.contacts.obj_ground, self.contacts.obj_arm, self.contacts.obj_other],
            "true_target": true_target, "lifted": [float(self.ep.lifted)],
        }
        return {"deploy": DEPLOY.pack(deploy), "priv": PRIV.pack(priv)}

    # -------------------------------------------------------------- helpers
    def set_target(self, p: np.ndarray, a: np.ndarray) -> None:
        self.target_p, self.target_a = np.asarray(p, float), np.asarray(a, float) / np.linalg.norm(a)

    def render(self, renderer: mujoco.Renderer) -> np.ndarray:
        renderer.update_scene(self.d, camera="video")
        return renderer.render()
