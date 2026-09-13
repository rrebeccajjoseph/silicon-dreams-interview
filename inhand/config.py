"""All declared assumptions live here so the report can quote them directly."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Envelope:
    """Object geometry envelope G. Bounds are in metres and relative to the LEAP hand
    (finger span ~0.09 m across index..ring, fingertip reach ~0.12 m from the palm)."""

    r_min: float = 0.012
    r_max: float = 0.035
    h_min: float = 0.02
    h_max: float = 0.16
    density_min: float = 300.0   # kg/m^3, roughly balsa to dense plastic
    density_max: float = 1200.0
    # standing on an end face is only sampled when it is comfortably stable
    alpha_stand_max: float = 3.0
    alpha_lie_min: float = 0.15  # a thin disk on its rim just falls over
    start_poses: tuple[str, ...] = ("lying", "standing")

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        r = rng.uniform(self.r_min, self.r_max)
        h = rng.uniform(self.h_min, self.h_max)
        rho = rng.uniform(self.density_min, self.density_max)
        return r, h, rho

    def scaled(self, frac: float) -> "Envelope":
        """Shrink the box toward its centre. Used by the ADR curriculum."""
        mid_r = 0.5 * (self.r_min + self.r_max)
        mid_h = 0.5 * (self.h_min + self.h_max)
        e = Envelope(**self.__dict__)
        e.r_min = mid_r - frac * (mid_r - self.r_min)
        e.r_max = mid_r + frac * (self.r_max - mid_r)
        e.h_min = mid_h - frac * (mid_h - self.h_min)
        e.h_max = mid_h + frac * (self.h_max - mid_h)
        return e


@dataclass
class Friction:
    ground: tuple[float, float] = (0.4, 1.0)
    obj: tuple[float, float] = (0.5, 1.2)


@dataclass
class Sensors:
    """Tactile: one 3-axis force + contact-centroid pad per fingertip and one on the palm.
    Comparable to a Xela uSkin patch read at the control rate. Assumed noiseless."""

    pads: tuple[str, ...] = ("if_tip", "mf_tip", "rf_tip", "th_tip", "palm")
    # vision stand-in: ground-truth pose is exposed only when this fraction of
    # sampled surface points is visible from at least one camera
    visible_frac: float = 0.3
    n_surface_pts: int = 32
    camera_hz: float = 5.0  # pose estimate refreshes at this rate, held in between
    cameras: tuple[tuple[float, float, float], ...] = (
        (1.1, 0.7, 0.9),    # fixed external, looking at the workspace
        (1.1, -0.7, 0.9),   # second external, other side
    )
    # wrist camera on a bracket behind the wrist, in the LEAP palm body frame
    wrist_cam_pos: tuple[float, float, float] = (-0.13, -0.037, -0.09)


@dataclass
class Control:
    sim_dt: float = 0.004   # 2 ms was stable too but 1.65x slower; no NaNs or extra drops at 4 ms
    ctrl_hz: float = 25.0   # 10 substeps
    arm_step: float = 0.04    # rad per control step, delta position target
    hand_step: float = 0.25

    @property
    def substeps(self) -> int:
        return int(round(1.0 / (self.ctrl_hz * self.sim_dt)))


@dataclass
class Success:
    pos_tol: float = 0.015
    ang_tol: float = np.deg2rad(15.0)
    hold_s: float = 2.0
    horizon_s: float = 20.0

    def hold_steps(self, ctrl: Control) -> int:
        return int(round(self.hold_s * ctrl.ctrl_hz))

    def horizon_steps(self, ctrl: Control) -> int:
        return int(round(self.horizon_s * ctrl.ctrl_hz))


@dataclass
class Targets:
    """T* sampling in the palm frame. Feasible = clears the palm face, inside the
    fingertip workspace, and no penetration with the palm at open fingers."""

    max_radius: float = 0.08
    z_clear_max: float = 0.02
    # curriculum: reorient training starts with targets close to the lift pose
    ang_curriculum: tuple[float, float] = (np.deg2rad(20.0), np.pi / 2)
    pos_curriculum: tuple[float, float] = (0.01, 0.05)


@dataclass
class RewardWeights:
    """Dense terms are bounded and positive so an episode is always worth continuing.
    With negative per-step shaping the first policies learned to end episodes early by
    pushing the object into the arm or off the table."""

    pos: float = 1.0          # * exp(-epos / pos_scale)
    ang: float = 1.0          # * exp(-eang / ang_scale)
    pos_scale: float = 0.03
    ang_scale: float = 0.4
    in_tol: float = 2.0
    success: float = 50.0
    drop: float = -10.0
    violation: float = -10.0
    action: float = 0.002
    # pre-lift shaping only
    approach: float = 1.0     # * exp(-dist / approach_scale)
    approach_scale: float = 0.1
    lift: float = 10.0


@dataclass
class Config:
    envelope: Envelope = field(default_factory=Envelope)
    friction: Friction = field(default_factory=Friction)
    sensors: Sensors = field(default_factory=Sensors)
    control: Control = field(default_factory=Control)
    success: Success = field(default_factory=Success)
    targets: Targets = field(default_factory=Targets)
    reward: RewardWeights = field(default_factory=RewardWeights)
    seed: int = 0
