"""Damped least-squares IK on a site. Used for keyframes and the scripted grasp."""

from __future__ import annotations

import mujoco
import numpy as np


def site_pose(m: mujoco.MjModel, d: mujoco.MjData, sid: int) -> tuple[np.ndarray, np.ndarray]:
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def rotvec_error(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Rotation vector taking R to R_target, expressed in the world frame."""
    dR = R_target @ R.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, dR.ravel())
    rv = np.zeros(3)
    mujoco.mju_quat2Vel(rv, q, 1.0)
    return rv


def solve_ik(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    sid: int,
    dof_ids: np.ndarray,
    pos: np.ndarray,
    R: np.ndarray | None,
    q_seed: np.ndarray,
    iters: int = 200,
    damping: float = 1e-3,
    tol: float = 1e-4,
) -> tuple[np.ndarray, bool]:
    """Returns joint config for the given dofs and whether it converged.
    The rest of qpos is left as it is in `d`."""
    qadr = m.jnt_qposadr[[m.dof_jntid[i] for i in dof_ids]]
    lo, hi = m.jnt_range[m.dof_jntid[dof_ids]].T
    d.qpos[qadr] = q_seed
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, Rc = site_pose(m, d, sid)
        err = pos - p
        if R is not None:
            err = np.concatenate([err, rotvec_error(Rc, R)])
        if np.linalg.norm(err) < tol:
            return d.qpos[qadr].copy(), True
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = jacp[:, dof_ids] if R is None else np.vstack([jacp, jacr])[:, dof_ids]
        Jp = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), np.eye(J.shape[0]))
        dq = Jp @ err
        # pull toward the seed in the nullspace so the elbow stays sane
        N = np.eye(len(dof_ids)) - np.linalg.pinv(J) @ J
        dq += 0.1 * N @ (q_seed - d.qpos[qadr])
        d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)
    return d.qpos[qadr].copy(), False
