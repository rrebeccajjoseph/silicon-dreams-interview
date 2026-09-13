"""The deployed systems. Each one consumes the DEPLOY observation vector and nothing
else; the stage switch is the target's availability flag, which the world sets at lift."""

from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .observations import DEPLOY
from .planc.estimator import Estimator, decode
from .planc.planner import plan
from .planc.scripted_grasp import ScriptedGrasp, axis_from_outer
from .rl.nets import Actor
from .rl.ppo import load_actor
from .scene import Scene
from .tasks.primitives import HOLD, SKILL_DIM


def target_available(obs: np.ndarray) -> bool:
    return obs[DEPLOY["target"]][-1] > 0.5


class ActorRunner:
    """Deterministic actor with its own hidden state."""

    def __init__(self, actor: Actor):
        assert actor.norm.mean.shape[0] == DEPLOY.dim, "deployed actors read deploy obs only"
        self.actor, self.h = actor, None

    def reset(self) -> None:
        self.h = self.actor.init_hidden(1, "cpu")

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        mean, self.h = self.actor(torch.as_tensor(obs).view(1, 1, -1), self.h)
        return mean[0, 0].numpy()


class System:
    name = "base"

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray: ...


class PlanA(System):
    """Learned grasp (value-aware) then a learned reorient policy."""
    name = "A"

    def __init__(self, grasp_ckpt: str, reorient_ckpt: str):
        self.grasp = ActorRunner(load_actor(grasp_ckpt, deploy_only=True))
        self.reorient = ActorRunner(load_actor(reorient_ckpt, deploy_only=True))

    def reset(self) -> None:
        self.grasp.reset()
        self.reorient.reset()
        self.stage = "grasp"

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.stage == "grasp" and target_available(obs):
            self.stage = "reorient"
        return (self.grasp if self.stage == "grasp" else self.reorient).act(obs)


class PlanB(System):
    """One recurrent policy for the whole episode."""
    name = "B"

    def __init__(self, ckpt: str):
        self.pi = ActorRunner(load_actor(ckpt, deploy_only=True))

    def reset(self) -> None:
        self.pi.reset()
        self.stage = "grasp"

    def act(self, obs: np.ndarray) -> np.ndarray:
        if target_available(obs):
            self.stage = "reorient"
        return self.pi.act(obs)


class PlanC(System):
    """Scripted grasp, then estimator + planner over learned skills."""
    name = "C"

    def __init__(self, scene: Scene, cfg: Config, skills_ckpt: str, estimator_ckpt: str,
                 steps_per_skill: int = 30, grasp_ckpt: str | None = None):
        self.cfg = cfg
        # the scripted grasp is the plan C default; a learned one from plan A can be swapped in
        self.grasp = ActorRunner(load_actor(grasp_ckpt, deploy_only=True)) if grasp_ckpt else ScriptedGrasp(scene, cfg)
        self.skills = ActorRunner(load_actor(skills_ckpt, deploy_only=True))
        self.est = Estimator()
        self.est.load_state_dict(torch.load(estimator_ckpt, map_location="cpu"))
        self.est.eval()
        self.steps_per_skill = steps_per_skill

    def reset(self) -> None:
        self.grasp.reset()
        self.skills.reset()
        self.h_est = self.est.init_hidden(1, "cpu")
        self.stage = "grasp"
        self.skill, self.skill_t = HOLD, 0
        self.settle = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        y, self.h_est = self.est(torch.as_tensor(obs).view(1, 1, -1), self.h_est)
        if self.stage == "grasp":
            if target_available(obs):
                self.stage = "settle"
            else:
                return self.grasp.act(obs)
        if self.stage == "settle":
            # finish the wrist roll before handing over to the skills
            self.settle += 1
            if self.settle < 60 and isinstance(self.grasp, ScriptedGrasp):
                return self.grasp.act(obs)
            self.stage = "reorient"
        p, a, r, h = decode(y[0, 0].numpy())
        t = obs[DEPLOY["target"]]
        if self.skill_t == 0:
            self.skill = plan(p, a, r, h, t[:3], axis_from_outer(t[3:9]), self.cfg.success)
            self.skills.reset()
        self.skill_t = (self.skill_t + 1) % self.steps_per_skill
        o = obs.copy()
        o[DEPLOY["skill"]] = np.eye(SKILL_DIM, dtype=np.float32)[self.skill]
        # the skill was trained on its own local target, which is the nominal effect
        return self.skills.act(o)
