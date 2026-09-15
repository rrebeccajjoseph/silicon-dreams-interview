"""Contact bookkeeping: who touches the object, tactile pads, and the vision gate."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import Ids


@dataclass
class ContactState:
    obj_hand: bool = False
    obj_ground: bool = False
    obj_arm: bool = False
    obj_other: bool = False
    robot_ground: bool = False

    @property
    def hand_only(self) -> bool:
        return self.obj_hand and not (self.obj_ground or self.obj_arm or self.obj_other)


def _lookup(m: mujoco.MjModel, ids: Ids) -> np.ndarray:
    """Geom id -> class (0 other, 1 hand, 2 floor, 3 arm). Built once per model."""
    lut = getattr(ids, "_lut", None)
    if lut is None or len(lut) != m.ngeom:
        lut = np.zeros(m.ngeom, dtype=np.int8)
        lut[ids.hand_geoms] = 1
        lut[ids.floor] = 2
        lut[ids.arm_geoms] = 3
        ids._lut = lut
    return lut


def classify(m: mujoco.MjModel, d: mujoco.MjData, ids: Ids) -> ContactState:
    """Called every physics step, so it avoids a Python loop over contacts."""
    st = ContactState()
    if d.ncon == 0:
        return st
    lut = _lookup(m, ids)
    g = d.contact.geom[:d.ncon]
    on_obj = (g == ids.obj_geom).any(axis=1)
    robot = (lut[g] == 1) | (lut[g] == 3)
    st.robot_ground = bool(((g == ids.floor).any(axis=1) & robot.any(axis=1) & ~on_obj).any())
    if not on_obj.any():
        return st
    go = g[on_obj]
    other = np.where(go[:, 0] == ids.obj_geom, go[:, 1], go[:, 0])
    cls = lut[other]
    st.obj_hand = bool((cls == 1).any())
    st.obj_ground = bool((cls == 2).any())
    st.obj_arm = bool((cls == 3).any())
    st.obj_other = bool((cls == 0).any())
    return st


PAD_DIM = 5  # contact flag, normal force, contact centroid xyz in the pad body frame


def tactile(m: mujoco.MjModel, d: mujoco.MjData, ids: Ids, pads: tuple[str, ...]) -> np.ndarray:
    """Per pad: [touching, sum normal force, force-weighted centroid]. All contacts count,
    not just the object. A real pad does not know what it is touching."""
    out = np.zeros((len(pads), PAD_DIM))
    f6 = np.zeros(6)
    lookup = {g: k for k, pad in enumerate(pads) for g in ids.pad_geoms[pad]}
    for i in range(d.ncon):
        c = d.contact[i]
        if c.exclude:  # inside the floor's no-force band: nothing to feel
            continue
        for g in (c.geom1, c.geom2):
            if g in lookup:
                k = lookup[g]
                mujoco.mj_contactForce(m, d, i, f6)
                fn = abs(f6[0])
                out[k, 0] = 1.0
                out[k, 1] += fn
                out[k, 2:] += fn * c.pos
    for k, pad in enumerate(pads):
        if out[k, 1] > 0:
            b = ids.pad_bodies[pad]
            R = d.xmat[b].reshape(3, 3)
            out[k, 2:] = R.T @ (out[k, 2:] / out[k, 1] - d.xpos[b])
    return out.ravel()


def visible_fraction(
    m: mujoco.MjModel, d: mujoco.MjData, ids: Ids, pts_obj: np.ndarray, cams: np.ndarray,
    fwd: np.ndarray, cos_half_fov: float,
) -> float:
    """Fraction of surface sample points inside some camera's field of view and unoccluded from it."""
    R = d.xmat[ids.obj_body].reshape(3, 3)
    pts = pts_obj @ R.T + d.xpos[ids.obj_body]
    seen = np.zeros(len(pts), dtype=bool)
    geomid = np.zeros(len(pts), dtype=np.int32)
    dist = np.zeros(len(pts))
    for cam, f in zip(cams, fwd):
        vec = pts - cam
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / norm
        in_view = vec @ f >= cos_half_fov
        mujoco.mj_multiRay(m, d, cam.astype(np.float64), vec.ravel(), None, True, -1, geomid, dist, None, len(pts), 10.0)
        seen |= in_view & (geomid == ids.obj_geom) & (np.abs(dist - norm[:, 0]) < 0.005)
    return float(seen.mean())
