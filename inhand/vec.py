"""Vector envs. `VecEnv` steps envs on threads (mj_step releases the GIL, the Python
bookkeeping does not). `SubprocVecEnv` puts groups of envs in worker processes and
scales past the GIL; it is what training uses on the cluster."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .env import CylinderEnv


def _stack(obs: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([o[k] for o in obs]) for k in obs[0]}


def _step_one(env: CylinderEnv, a: np.ndarray):
    obs, r, done, info = env.step(a)
    if done:
        info["final_obs"] = obs
        obs = env.reset()
    return obs, r, done, info


class VecEnv:
    def __init__(self, envs: list[CylinderEnv], workers: int | None = None):
        self.envs = envs
        self.n = len(envs)
        self.pool = ThreadPoolExecutor(workers or min(self.n, 16))

    def reset(self) -> dict[str, np.ndarray]:
        return _stack(list(self.pool.map(lambda e: e.reset(), self.envs)))

    def step(self, actions: np.ndarray):
        out = list(self.pool.map(lambda p: _step_one(*p), zip(self.envs, actions)))
        obs, rew, done, infos = zip(*out)
        return _stack(obs), np.array(rew, dtype=np.float32), np.array(done), list(infos)

    def set_attr(self, name: str, value) -> None:
        for e in self.envs:
            setattr(e, name, value)

    def close(self) -> None:
        self.pool.shutdown()


def _worker(conn, env_cls, cfg, kwargs_list):
    from .scene import Scene
    scene = Scene(cfg)
    envs = [env_cls(scene, cfg, **kw) for kw in kwargs_list]
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            conn.send([e.reset() for e in envs])
        elif cmd == "step":
            conn.send([_step_one(e, a) for e, a in zip(envs, arg)])
        elif cmd == "set_attr":
            for e in envs:
                setattr(e, arg[0], arg[1])
            conn.send(None)
        elif cmd == "close":
            conn.close()
            return


class SubprocVecEnv:
    """Same interface as VecEnv. Envs are built inside the workers from (class, cfg, kwargs),
    so nothing unpicklable crosses the process boundary."""

    def __init__(self, env_cls, cfg, kwargs_list: list[dict], workers: int):
        self.n = len(kwargs_list)
        ctx = mp.get_context("forkserver")
        groups = [kwargs_list[i::workers] for i in range(workers)]
        groups = [g for g in groups if g]
        self.order = np.argsort(np.concatenate([np.arange(i, self.n, workers) for i in range(len(groups))]))
        self.conns, self.procs = [], []
        for g in groups:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, env_cls, cfg, g), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.sizes = [len(g) for g in groups]

    def _gather(self, results):
        flat = [x for group in results for x in group]
        return [flat[i] for i in self.order]

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return _stack(self._gather([c.recv() for c in self.conns]))

    def step(self, actions: np.ndarray):
        chunks = [actions[i::len(self.conns)] for i in range(len(self.conns))]
        for c, a in zip(self.conns, chunks):
            c.send(("step", a))
        out = self._gather([c.recv() for c in self.conns])
        obs, rew, done, infos = zip(*out)
        return _stack(obs), np.array(rew, dtype=np.float32), np.array(done), list(infos)

    def set_attr(self, name: str, value) -> None:
        for c in self.conns:
            c.send(("set_attr", (name, value)))
        for c in self.conns:
            c.recv()

    def close(self) -> None:
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
