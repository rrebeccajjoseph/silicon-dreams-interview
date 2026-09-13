"""Threaded vector env. mj_step releases the GIL, so threads give real parallelism."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .env import CylinderEnv


class VecEnv:
    def __init__(self, envs: list[CylinderEnv], workers: int | None = None):
        self.envs = envs
        self.n = len(envs)
        self.pool = ThreadPoolExecutor(workers or min(self.n, 16))

    def reset(self) -> dict[str, np.ndarray]:
        obs = list(self.pool.map(lambda e: e.reset(), self.envs))
        return self._stack(obs)

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        def one(args):
            env, a = args
            obs, r, done, info = env.step(a)
            if done:
                info["final_obs"] = obs
                obs = env.reset()
            return obs, r, done, info

        out = list(self.pool.map(one, zip(self.envs, actions)))
        obs, rew, done, infos = zip(*out)
        return self._stack(obs), np.array(rew, dtype=np.float32), np.array(done), list(infos)

    @staticmethod
    def _stack(obs: list[dict]) -> dict[str, np.ndarray]:
        return {k: np.stack([o[k] for o in obs]) for k in obs[0]}

    def set_attr(self, name: str, value) -> None:
        for e in self.envs:
            setattr(e, name, value)
