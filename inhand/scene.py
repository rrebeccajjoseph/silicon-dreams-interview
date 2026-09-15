"""Assembles xArm7 + LEAP right hand + one cylinder on a ground plane."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import Config
from . import ik

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "menagerie"
ARM_XML = ASSETS / "ufactory_xarm7" / "xarm7_nohand.xml"
HAND_XML = ASSETS / "leap_hand" / "right_hand.xml"

# the hand sits on a bracket this far out from the flange, like a real LEAP mount; it
# keeps long rods lying across the palm from touching the wrist link
MOUNT_OFFSET = 0.04
BRACKET_RADIUS = 0.032  # matches the xArm flange

# joint limits from the source URDFs (dexsuite leap_hand_right.urdf, xarm_ros xarm7.urdf).
# Torque is enforced by the actuators; speed is checked and reported by eval.py
HAND_EFFORT = 0.95      # N m
HAND_VELOCITY = 8.48    # rad/s
ARM_VELOCITY = 3.14     # rad/s

# object-ground contact counts within this distance. Soft contacts sink 1-3 mm and flicker for
# single 2 ms steps when a grasped rod is disturbed; without a band those flickers read as a lift
# followed by a ground violation. The band is stricter both ways: lift needs the object 1 mm clear,
# and coming within 1 mm of the ground after lift is a violation
CONTACT_RESOLUTION = 0.001

# palm frame in the LEAP palm body: origin on the palm face under the fingers,
# x toward the fingertips, z is the outward palm normal (object side)
PALM_SITE_POS = (-0.02, -0.037, -0.03)
PALM_SITE_QUAT = (0.0, 1.0, 0.0, 0.0)  # 180 deg about x flips y and z

# arm poses solved once at build time
HOME_POS = np.array([0.45, 0.0, 0.32])   # palm down, over the workspace
HOLD_POS = np.array([0.45, 0.0, 0.38])   # palm up, for in-hand resets
ARM_SEEDS = [
    np.array([0.0, -0.3, 0.0, 0.9, 0.0, 1.2, 0.0]),
    np.array([0.0, 0.3, 0.0, 1.5, 0.0, -1.2, 0.0]),
    np.array([0.0, -0.3, 0.0, 0.9, 0.0, 1.2, 3.1]),
    np.array([0.0, 0.5, 0.0, 1.8, 0.0, -1.3, 3.1]),
]

CAM_POS = np.array([1.35, -1.05, 0.75])
CAM_LOOKAT = np.array([0.45, 0.0, 0.18])

WORKSPACE_X = (0.36, 0.56)
WORKSPACE_Y = (-0.14, 0.14)


@dataclass
class Ids:
    obj_body: int
    obj_geom: int
    obj_qadr: int
    obj_dofadr: int
    floor: int
    palm_site: int
    palm_body: int
    arm_dofs: np.ndarray
    hand_dofs: np.ndarray
    arm_acts: np.ndarray
    hand_acts: np.ndarray
    hand_geoms: np.ndarray
    arm_geoms: np.ndarray
    pad_geoms: dict[str, np.ndarray]
    pad_bodies: dict[str, int]
    palm_geoms: np.ndarray   # palm + wrist, used for target feasibility


def _look_at(pos: np.ndarray, target: np.ndarray) -> list[float]:
    """MuJoCo camera xyaxes for a camera at `pos` looking at `target`."""
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return [*right, *up]


def _rot(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


PALM_DOWN = _rot(np.array([1.0, 0, 0]), np.array([0, 0, -1.0]))
PALM_UP = _rot(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))


def build_spec(cfg: Config) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    spec.option.timestep = cfg.control.sim_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 2

    flange = next(s for s in spec.sites if s.name == "attachment_site")
    link7 = next(b for b in spec.bodies if b.name == "link7")
    mount = link7.add_site(name="hand_mount", pos=np.asarray(flange.pos) + [0, 0, MOUNT_OFFSET], quat=flange.quat)
    spec.attach(hand, prefix="hand/", site=mount)
    # the bracket is solid: an object between flange and palm counts as touching the arm
    link7.add_geom(name="bracket", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BRACKET_RADIUS, MOUNT_OFFSET / 2, 0],
                   pos=np.asarray(flange.pos) + [0, 0, MOUNT_OFFSET / 2], quat=flange.quat, rgba=[0.6, 0.6, 0.6, 1])
    # the Menagerie LEAP model has no torque limit; the URDF it was derived from says 0.95 N m per joint
    for act in spec.actuators:
        if act.name.startswith("hand/"):
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-HAND_EFFORT, HAND_EFFORT]

    palm = next(b for b in spec.bodies if b.name == "hand/palm")
    s = palm.add_site(name="palm_frame", pos=PALM_SITE_POS, quat=PALM_SITE_QUAT, size=[0.004] * 3)
    s.group = 4

    w = spec.worldbody
    w.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    # margin == gap: pairs within CONTACT_RESOLUTION of the floor are reported as contacts but apply
    # no force, so the physics is unchanged while lift and violations see a 1 mm contact band
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
               friction=[0.7, 0.005, 0.0001], rgba=[0.3, 0.35, 0.4, 1],
               margin=CONTACT_RESOLUTION, gap=CONTACT_RESOLUTION)
    obj = w.add_body(name="obj", pos=[0.45, 0, 0.05])
    obj.add_freejoint(name="obj_free")
    obj.add_geom(name="obj", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.05, 0],
                 condim=4, friction=[0.8, 0.005, 0.0001], rgba=[0.9, 0.5, 0.1, 1],
                 solref=[0.01, 1], solimp=[0.9, 0.95, 0.002, 0.5, 2])
    w.add_camera(name="video", pos=CAM_POS, xyaxes=_look_at(CAM_POS, CAM_LOOKAT))
    return spec


class Scene:
    """Compiled model plus id tables and keyframes. One per process; envs copy the model."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.spec = build_spec(cfg)
        self.model = self.spec.compile()
        m = self.model
        jn = [m.joint(i).name for i in range(m.njnt)]
        an = [m.actuator(i).name for i in range(m.nu)]
        gn = [m.geom(i).name for i in range(m.ngeom)]
        bn = [m.body(i).name for i in range(m.nbody)]

        def geoms_of(pred):
            return np.array([g for g in range(m.ngeom) if pred(g)], dtype=int)

        pad_geoms = {}
        pad_bodies = {}
        for pad in cfg.sensors.pads:
            if pad == "palm":
                pad_geoms[pad] = geoms_of(lambda g: gn[g].startswith("hand/palm_collision"))
                pad_bodies[pad] = m.body("hand/palm").id
            else:
                pad_geoms[pad] = np.array([m.geom(f"hand/{pad}").id])
                pad_bodies[pad] = m.geom_bodyid[pad_geoms[pad][0]]

        arm_jnt = [i for i, n in enumerate(jn) if n.startswith("joint")]
        hand_jnt = [i for i, n in enumerate(jn) if n.startswith("hand/")]
        self.ids = Ids(
            obj_body=m.body("obj").id,
            obj_geom=m.geom("obj").id,
            obj_qadr=m.jnt_qposadr[m.joint("obj_free").id],
            obj_dofadr=m.jnt_dofadr[m.joint("obj_free").id],
            floor=m.geom("floor").id,
            palm_site=m.site("palm_frame").id,
            palm_body=m.body("hand/palm").id,
            arm_dofs=np.array([m.jnt_dofadr[j] for j in arm_jnt]),
            hand_dofs=np.array([m.jnt_dofadr[j] for j in hand_jnt]),
            arm_acts=np.array([i for i, n in enumerate(an) if n.startswith("act")]),
            hand_acts=np.array([i for i, n in enumerate(an) if n.startswith("hand/")]),
            hand_geoms=geoms_of(lambda g: bn[m.geom_bodyid[g]].startswith("hand/") and m.geom_contype[g] != 0),
            arm_geoms=geoms_of(lambda g: bn[m.geom_bodyid[g]].startswith("link") and m.geom_contype[g] != 0),
            pad_geoms=pad_geoms,
            pad_bodies=pad_bodies,
            palm_geoms=geoms_of(lambda g: gn[g].startswith("hand/palm_collision") or bn[m.geom_bodyid[g]] == "link7"),
        )
        self.arm_qadr = m.jnt_qposadr[arm_jnt]
        self.hand_qadr = m.jnt_qposadr[hand_jnt]
        self.hand_ctrl_range = m.actuator_ctrlrange[self.ids.hand_acts]
        self.arm_ctrl_range = m.actuator_ctrlrange[self.ids.arm_acts]
        self.hand_open = np.zeros(len(hand_jnt))
        self.q_home = self._solve_arm(HOME_POS, PALM_DOWN)
        self.q_hold = self._solve_arm(HOLD_POS, PALM_UP)

    def _solve_arm(self, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Try a few seeds, keep the most compact collision-free solution."""
        d = mujoco.MjData(self.model)
        best = None
        for seed in ARM_SEEDS:
            mujoco.mj_resetData(self.model, d)
            q, ok = ik.solve_ik(self.model, d, self.ids.palm_site, self.ids.arm_dofs, pos, R, seed)
            if not ok:
                continue
            d.qpos[self.ids.obj_qadr + 2] = 5.0  # park the object away from the robot
            mujoco.mj_forward(self.model, d)
            if d.ncon > 0:
                continue
            if best is None or np.abs(q).max() < np.abs(best).max():
                best = q
        if best is None:
            raise RuntimeError(f"arm IK found no collision-free solution for {pos}")
        return best

    def new_model(self) -> mujoco.MjModel:
        return copy.copy(self.model)
