"""Run the scripted grasp and save the simulator state at each handover (palm-up roll done,
object held). The reorient stage resets from these so it trains on the states it is deployed on.

  python scripts/collect_handovers.py --episodes 600 --procs 6 --out checkpoints/handovers.npz"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inhand.config import Config  # noqa: E402
from inhand.scene import Scene  # noqa: E402
from inhand.systems import PlanS  # noqa: E402
from inhand.tasks.monolithic import FullTaskEnv  # noqa: E402


def collect(seed: int, episodes: int) -> list[dict]:
    cfg = Config(seed=seed)
    scene = Scene(cfg)
    env = FullTaskEnv(scene, cfg, seed=seed)
    system = PlanS(scene, cfg)
    out = []
    for _ in range(episodes):
        obs = env.reset()
        system.reset()
        while True:
            obs, _, done, info = env.step(system.act(obs["deploy"]))
            if done:
                break
            if system.stage == "reorient":
                if env.contacts.hand_only:
                    o = env.obj
                    out.append({"obj": np.array([o.r, o.h, o.density, o.mu_obj, o.mu_ground]),
                                "qpos": env.d.qpos.copy(), "qvel": env.d.qvel.copy(), "targets": env.targets.copy()})
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--out", default="checkpoints/handovers.npz")
    args = p.parse_args()
    per = args.episodes // args.procs
    with get_context("spawn").Pool(args.procs) as pool:
        rows = [r for chunk in pool.starmap(collect, [(500 + i, per) for i in range(args.procs)]) for r in chunk]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    print(f"{len(rows)} handovers from {per * args.procs} episodes -> {args.out}")


if __name__ == "__main__":
    main()
