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


def classify(m: mujoco.MjModel, d: mujoco.MjData, ids: Ids) -> ContactState:
    hand = set(ids.hand_geoms.tolist())
    arm = set(ids.arm_geoms.tolist())
    st = ContactState()
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        if ids.obj_geom in (g1, g2):
            other = g2 if g1 == ids.obj_geom else g1
            if other in hand:
                st.obj_hand = True
            elif other == ids.floor:
                st.obj_ground = True
            elif other in arm:
                st.obj_arm = True
            else:
                st.obj_other = True
        elif ids.floor in (g1, g2):
            other = g2 if g1 == ids.floor else g1
            if other in hand or other in arm:
                st.robot_ground = True
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
    m: mujoco.MjModel, d: mujoco.MjData, ids: Ids, pts_obj: np.ndarray, cams: np.ndarray
) -> float:
    """Fraction of surface sample points seen unoccluded by at least one camera."""
    R = d.xmat[ids.obj_body].reshape(3, 3)
    pts = pts_obj @ R.T + d.xpos[ids.obj_body]
    seen = np.zeros(len(pts), dtype=bool)
    geomid = np.zeros(len(pts), dtype=np.int32)
    dist = np.zeros(len(pts))
    for cam in cams:
        vec = pts - cam
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / norm
        mujoco.mj_multiRay(m, d, cam.astype(np.float64), vec.ravel(), None, True, -1, geomid, dist, None, len(pts), 10.0)
        seen |= (geomid == ids.obj_geom) & (np.abs(dist - norm[:, 0]) < 0.005)
    return float(seen.mean())
