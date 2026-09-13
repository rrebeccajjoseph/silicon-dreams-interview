"""PPO with an asymmetric critic. Minibatches are whole env rows so a recurrent actor
gets proper truncated BPTT over the rollout."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..vec import VecEnv
from .buffer import Rollout
from .nets import Actor, Critic


@dataclass
class PPOConfig:
    T: int = 64
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent: float = 0.0
    vf: float = 1.0
    max_grad: float = 1.0
    target_kl: float = 0.03
    action_noise_floor: float = -2.5  # log std lower bound


class PPO:
    def __init__(self, actor: Actor, critic: Critic, cfg: PPOConfig, device: str = "cpu", actor_key: str = "deploy"):
        self.actor, self.critic, self.cfg, self.device = actor.to(device), critic.to(device), cfg, device
        self.actor_key = actor_key  # 'priv' for teachers, 'deploy' for anything that ships
        self.opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)

    # --------------------------------------------------------------- collect
    @torch.no_grad()
    def collect(self, vec: VecEnv, obs: dict, h: torch.Tensor | None) -> tuple[Rollout, dict, torch.Tensor | None, list[dict]]:
        T, N = self.cfg.T, vec.n
        A = self.actor.head.out_features
        buf = {
            "obs_d": np.zeros((T, N, obs[self.actor_key].shape[1]), np.float32),
            "obs_p": np.zeros((T, N, obs["priv"].shape[1]), np.float32),
            "act": np.zeros((T, N, A), np.float32), "rew": np.zeros((T, N), np.float32),
            "done": np.zeros((T, N), np.float32), "signals": np.zeros((T, N, 10), np.float32),
            "obj_palm": np.zeros((T, N, 6), np.float32), "hand_only": np.zeros((T, N), np.float32),
        }
        h0 = h.squeeze(0).cpu().numpy().copy() if h is not None else np.zeros((N, 0), np.float32)
        finished = []
        for t in range(T):
            oa = torch.as_tensor(obs[self.actor_key], device=self.device)
            mean, h = self.actor(oa.unsqueeze(0), h)
            act = self.actor.dist(mean).sample().squeeze(0)
            act_np = act.cpu().numpy()
            buf["obs_d"][t], buf["obs_p"][t], buf["act"][t] = obs[self.actor_key], obs["priv"], act_np
            obs, rew, done, infos = vec.step(act_np)
            buf["rew"][t], buf["done"][t] = rew, done
            for n, info in enumerate(infos):
                buf["signals"][t, n] = info["signals"]
                buf["obj_palm"][t, n] = info["obj_palm"]
                buf["hand_only"][t, n] = info["hand_only"]
                if done[n]:
                    finished.append(info)
            if h is not None:
                h = h * (1.0 - torch.as_tensor(done, dtype=torch.float32, device=self.device)).view(1, -1, 1)
        ro = Rollout(**buf, h0=h0, last_obs_p=obs["priv"].copy(), relabeled=np.zeros(N, bool))
        return ro, obs, h, finished

    # ---------------------------------------------------------------- update
    def update(self, ro: Rollout) -> dict:
        cfg, dev = self.cfg, self.device
        self.actor.norm.update(torch.as_tensor(ro.obs_d, device=dev))
        self.critic.norm.update(torch.as_tensor(ro.obs_p, device=dev))
        old_actor = copy.deepcopy(self.actor).eval()
        obs_d = torch.as_tensor(ro.obs_d, device=dev)
        obs_p = torch.as_tensor(ro.obs_p, device=dev)
        act = torch.as_tensor(ro.act, device=dev)
        rew = torch.as_tensor(ro.rew, device=dev)
        done = torch.as_tensor(ro.done, device=dev)
        h0 = torch.as_tensor(ro.h0, device=dev).unsqueeze(0) if self.actor.recurrent else None
        with torch.no_grad():
            mean, _ = old_actor(obs_d, h0, done)
            old_logp = old_actor.dist(mean).log_prob(act).sum(-1)
            val = self.critic(obs_p)
            last_val = self.critic(torch.as_tensor(ro.last_obs_p, device=dev))
            adv = self._gae(rew, done, val, last_val)
            ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = ro.N
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=dev)
            for mb in perm.chunk(cfg.minibatches):
                hmb = h0[:, mb] if h0 is not None else None
                mean, _ = self.actor(obs_d[:, mb], hmb, done[:, mb])
                dist = self.actor.dist(mean)
                logp = dist.log_prob(act[:, mb]).sum(-1)
                ratio = (logp - old_logp[:, mb]).exp()
                a = adv[:, mb]
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v_loss = F.mse_loss(self.critic(obs_p[:, mb]), ret[:, mb])
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + cfg.vf * v_loss - cfg.ent * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad)
                self.opt.step()
                with torch.no_grad():
                    self.actor.log_std.clamp_(min=cfg.action_noise_floor)
                    kl = (old_logp[:, mb] - logp).mean().item()
                stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item(); stats["kl"] += kl
                stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item(); stats["n"] += 1
            if stats["kl"] / stats["n"] > cfg.target_kl:
                break
        n = stats.pop("n")
        return {k: v / n for k, v in stats.items()} | {"ret_mean": ret.mean().item(), "std": self.actor.log_std.exp().mean().item()}

    def _gae(self, rew, done, val, last_val):
        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        gae = torch.zeros_like(last_val)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + self.cfg.gamma * nv * nonterm - val[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * nonterm * gae
            adv[t] = gae
        return adv

    # ------------------------------------------------------------ checkpoint
    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_kw": self.actor_kw, "critic_kw": self.critic_kw, "actor_key": self.actor_key}, path)


def make_ppo(act_dim: int, cfg: PPOConfig, recurrent: bool, device: str, actor_key: str = "deploy") -> PPO:
    from ..observations import DEPLOY, PRIV
    actor_kw = dict(obs_dim=(PRIV if actor_key == "priv" else DEPLOY).dim, act_dim=act_dim, recurrent=recurrent)
    critic_kw = dict(obs_dim=PRIV.dim)
    ppo = PPO(Actor(**actor_kw), Critic(**critic_kw), cfg, device, actor_key)
    ppo.actor_kw, ppo.critic_kw = actor_kw, critic_kw
    return ppo


def load_actor(path: str, device: str = "cpu", deploy_only: bool = False) -> Actor:
    ck = torch.load(path, map_location=device, weights_only=False)
    if deploy_only and ck.get("actor_key", "deploy") != "deploy":
        raise ValueError(f"{path} is a privileged teacher and cannot be deployed")
    actor = Actor(**ck["actor_kw"]).to(device)
    actor.load_state_dict(ck["actor"])
    return actor.eval()


def load_critic(path: str, device: str = "cpu") -> Critic:
    ck = torch.load(path, map_location=device, weights_only=False)
    critic = Critic(**ck["critic_kw"]).to(device)
    critic.load_state_dict(ck["critic"])
    return critic.eval()
