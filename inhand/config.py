"""All declared assumptions live here so the report can quote them directly."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Envelope:
    """Object geometry envelope G. Bounds are in metres and relative to the LEAP hand
    (finger span ~0.09 m across index..ring, fingertip reach ~0.12 m from the palm).
    Defaults are the trained scope: lying rods around r 2.5 cm, alpha ~2. WIDE is the
    envelope we measure against, not one we claim."""

    r_min: float = 0.022
    r_max: float = 0.028
    h_min: float = 0.08
    h_max: float = 0.12
    density_min: float = 500.0   # kg/m^3
    density_max: float = 900.0
    # standing on an end face is only sampled when it is comfortably stable
    alpha_stand_max: float = 3.0
    alpha_lie_min: float = 0.15  # a thin disk on its rim just falls over
    start_poses: tuple[str, ...] = ("lying",)

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


WIDE = Envelope(r_min=0.012, r_max=0.035, h_min=0.02, h_max=0.16, density_min=300.0,
                density_max=1200.0, start_poses=("lying", "standing"))


@dataclass
class Friction:
    ground: tuple[float, float] = (0.5, 0.9)
    obj: tuple[float, float] = (0.7, 1.1)


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
    camera_lookat: tuple[float, float, float] = (0.45, 0.0, 0.2)  # between the floor workspace and the palm-up hold
    # RealSense D435/D405-class RGB-D, 640x480 at 30 Hz. The frustum (87 x 58 deg) is approximated
    # by a cone with the narrow half-angle, so a point counts as in view only if it is inside both axes
    camera_resolution: tuple[int, int] = (640, 480)
    camera_half_fov_deg: float = 29.0
    # wrist camera on a bracket behind the wrist, in the LEAP palm body frame, aimed at a point 2 cm
    # above the palm-frame origin
    wrist_cam_pos: tuple[float, float, float] = (-0.13, -0.037, -0.09)
    wrist_cam_target_palm: tuple[float, float, float] = (0.0, 0.0, 0.02)


@dataclass
class Control:
    # 4 ms was stable (no NaNs) but not accurate: finger contacts flicked the object at up to
    # 0.8 m/s and the scripted grasp handed over 1 in 12 rods. 2 ms with softer object contacts: 1 in 2
    sim_dt: float = 0.002
    ctrl_hz: float = 25.0   # 20 substeps
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
    z_clear_max: float = 0.01
    # axis within a cone about palm y (across the fingers), position in a box above the palm.
    # Fixed in the palm frame, so nothing about it depends on the grasp.
    axis_nominal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_cone: float = np.deg2rad(40.0)
    # centred on where the hand holds a rod after the palm-up roll (median x -1.3, y -1.2 cm over 300
    # handovers of the final grasp). Chosen once from that distribution, never per episode
    x_range: tuple[float, float] = (-0.028, 0.002)
    y_range: tuple[float, float] = (-0.027, 0.003)
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
    # pre-lift shaping only. Kept small next to the lift bonus: at 1.0 per step the
    # policy learned to hover beside the object for the whole episode instead of lifting.
    approach: float = 0.1     # * exp(-dist / approach_scale)
    approach_scale: float = 0.1
    lift: float = 30.0


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
