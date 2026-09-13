"""Render a few episodes of a deployed system to mp4. Same flags as eval.py."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import build  # noqa: E402

import argparse  # noqa: E402

from inhand.config import Config  # noqa: E402
from inhand.scene import Scene  # noqa: E402
from inhand.tasks.monolithic import FullTaskEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan", choices=["a", "b", "c", "random"])
    p.add_argument("--grasp"); p.add_argument("--reorient"); p.add_argument("--policy")
    p.add_argument("--skills"); p.add_argument("--estimator")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="videos")
    args = p.parse_args()
    cfg = Config(seed=args.seed)
    scene = Scene(cfg)
    env = FullTaskEnv(scene, cfg, seed=args.seed)
    system = build(args, scene, cfg)
    renderer = mujoco.Renderer(env.m, 480, 640)
    Path(args.out).mkdir(exist_ok=True)
    for i in range(args.episodes):
        obs = env.reset()
        system.reset()
        frames = [env.render(renderer)]
        while True:
            obs, _, done, info = env.step(system.act(obs["deploy"]))
            frames.append(env.render(renderer))
            if done:
                break
        name = f"plan{args.plan}_{i}_alpha{env.obj.alpha:.1f}_{env.ep.start_pose}_{info['reason']}.mp4"
        imageio.mimwrite(Path(args.out) / name, np.stack(frames), fps=int(cfg.control.ctrl_hz))
        print("wrote", name)


if __name__ == "__main__":
    main()
