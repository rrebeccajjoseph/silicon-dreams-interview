"""Render a few episodes of a deployed system to mp4. Same flags as eval.py."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import build  # noqa: E402

import argparse  # noqa: E402

from inhand.config import WIDE, Config  # noqa: E402
from inhand.scene import Scene  # noqa: E402
from inhand.tasks.monolithic import FullTaskEnv  # noqa: E402


def ghost_target(env: FullTaskEnv, scene: mujoco.MjvScene) -> None:
    """Translucent green cylinder at T*, drawn only once the target has been revealed."""
    if not env.ep.lifted or scene.ngeom >= scene.maxgeom:
        return
    p_palm, R_palm = env.palm_pose()
    z = R_palm @ env.target_a
    x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    mat = np.stack([x, np.cross(z, x), z], axis=1)
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([env.obj.r, env.obj.h / 2, 0]), p_palm + R_palm @ env.target_p,
                        mat.ravel(), np.array([0.1, 0.8, 0.3, 0.35], dtype=np.float32))
    scene.ngeom += 1


def frame(env: FullTaskEnv, renderer: mujoco.Renderer) -> np.ndarray:
    """Workspace view next to a close-up that follows the palm."""
    renderer.update_scene(env.d, camera="video")
    ghost_target(env, renderer.scene)
    wide = renderer.render()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = env.palm_pose()[0] if env.ep.lifted else env.obj_pos_w
    cam.distance, cam.azimuth, cam.elevation = 0.35, -45.0, -45.0
    renderer.update_scene(env.d, camera=cam)
    ghost_target(env, renderer.scene)
    return np.concatenate([wide, renderer.render()], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan", nargs="?", default="s", choices=["s", "hold", "a", "b", "c", "random"])
    p.add_argument("--grasp"); p.add_argument("--reorient"); p.add_argument("--policy")
    p.add_argument("--skills"); p.add_argument("--estimator")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="videos")
    p.add_argument("--wide", action="store_true", help="sample objects from the WIDE envelope")
    p.add_argument("--want", help="only save episodes ending this way (e.g. success, dropped, violation_arm)")
    p.add_argument("--max-tries", type=int, default=60)
    args = p.parse_args()
    cfg = Config(seed=args.seed)
    scene = Scene(cfg)
    env = FullTaskEnv(scene, cfg, seed=args.seed, envelope=WIDE if args.wide else None)
    system = build(args, scene, cfg)
    renderer = mujoco.Renderer(env.m, 480, 640)
    Path(args.out).mkdir(exist_ok=True)
    saved = tried = 0
    while saved < args.episodes and tried < args.max_tries:
        # episodes are deterministic given the env RNG, so a matching one can be replayed with rendering
        rng_state = env.rng.bit_generator.state
        if args.want:
            info = rollout(env, system)
            tried += 1
            if not (info["reason"] == args.want or (args.want == "success" and info["success"])):
                continue
            env.rng.bit_generator.state = rng_state
        frames = []
        info = rollout(env, system, lambda: frames.append(frame(env, renderer)))
        tried += 0 if args.want else 1
        name = f"plan{args.plan}_s{args.seed}_{saved}_d{200 * env.obj.r:.1f}cm_h{100 * env.obj.h:.0f}cm_alpha{env.obj.alpha:.1f}_{env.ep.start_pose}_{info['reason']}_during-{getattr(system, 'stage', 'na')}.mp4"
        imageio.mimwrite(Path(args.out) / name, np.stack(frames), fps=int(cfg.control.ctrl_hz))
        saved += 1
        print("wrote", name, flush=True)


def rollout(env: FullTaskEnv, system, on_frame=None) -> dict:
    obs = env.reset()
    system.reset()
    if on_frame:
        on_frame()
    while True:
        obs, _, done, info = env.step(system.act(obs["deploy"]))
        if on_frame:
            on_frame()
        if done:
            return info

if __name__ == "__main__":
    main()
