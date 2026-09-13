"""Per-episode object sampling: geometry, mass, friction, and a stable start pose."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import Envelope, Friction
from .scene import Ids, WORKSPACE_X, WORKSPACE_Y


@dataclass
class ObjectParams:
    r: float
    h: float
    density: float
    mu_obj: float
    mu_ground: float

    @property
    def alpha(self) -> float:
        return self.h / (2 * self.r)

    @property
    def mass(self) -> float:
        return self.density * np.pi * self.r**2 * self.h


def sample_object(env: Envelope, fr: Friction, rng: np.random.Generator) -> ObjectParams:
    r, h, rho = env.sample(rng)
    return ObjectParams(r, h, rho, rng.uniform(*fr.obj), rng.uniform(*fr.ground))


def apply_object(m: mujoco.MjModel, ids: Ids, o: ObjectParams) -> None:
    """Writes geometry and mass into the model in place. Called at reset."""
    m.geom_size[ids.obj_geom] = [o.r, o.h / 2, 0]
    m.geom_rbound[ids.obj_geom] = np.hypot(o.r, o.h / 2)
    m.geom_friction[ids.obj_geom, 0] = o.mu_obj
    m.geom_friction[ids.floor, 0] = o.mu_ground
    mass = o.mass
    m.body_mass[ids.obj_body] = mass
    m.body_inertia[ids.obj_body] = [
        mass * (3 * o.r**2 + o.h**2) / 12,
        mass * (3 * o.r**2 + o.h**2) / 12,
        mass * o.r**2 / 2,
    ]
    mujoco.mj_setConst(m, mujoco.MjData(m))


def allowed_start_poses(env: Envelope, o: ObjectParams) -> list[str]:
    poses = []
    if "lying" in env.start_poses and o.alpha >= env.alpha_lie_min:
        poses.append("lying")
    if "standing" in env.start_poses and o.alpha <= env.alpha_stand_max:
        poses.append("standing")
    return poses or ["lying"]


def quat_from_z_to(v: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) rotating +z onto v."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = np.dot(z, v)
    if c < -1 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, v)
    q = np.array([1 + c, *axis])
    return q / np.linalg.norm(q)


def sample_start_pose(env: Envelope, o: ObjectParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (pos, quat, pose_class). Object rests on the ground, a hair above it."""
    kind = rng.choice(allowed_start_poses(env, o))
    x = rng.uniform(*WORKSPACE_X)
    y = rng.uniform(*WORKSPACE_Y)
    yaw = rng.uniform(-np.pi, np.pi)
    if kind == "lying":
        axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return np.array([x, y, o.r + 1e-3]), quat_from_z_to(axis), kind
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return np.array([x, y, o.h / 2 + 1e-3]), q, kind


def cylinder_surface_points(r: float, h: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Points on the surface in the object frame (axis along z). Area-weighted."""
    side = 2 * np.pi * r * h
    caps = 2 * np.pi * r**2
    n_side = int(round(n * side / (side + caps)))
    th = rng.uniform(0, 2 * np.pi, n)
    pts = np.zeros((n, 3))
    pts[:n_side] = np.stack([r * np.cos(th[:n_side]), r * np.sin(th[:n_side]),
                             rng.uniform(-h / 2, h / 2, n_side)], 1)
    rr = r * np.sqrt(rng.uniform(0, 1, n - n_side))
    sign = rng.choice([-1.0, 1.0], n - n_side)
    pts[n_side:] = np.stack([rr * np.cos(th[n_side:]), rr * np.sin(th[n_side:]), sign * h / 2], 1)
    return pts
