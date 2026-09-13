"""DAgger: a deploy-obs student imitates a privileged teacher on the student's own states."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..vec import VecEnv
from .nets import Actor


class Dagger:
    def __init__(self, teacher: Actor, student: Actor, device: str = "cpu", lr: float = 1e-3):
        self.teacher, self.student, self.device = teacher.to(device).eval(), student.to(device), device
        self.opt = torch.optim.Adam(student.parameters(), lr=lr)
        self.chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (obs_d, teacher act, done) [T,N,..]

    @torch.no_grad()
    def collect(self, vec: VecEnv, obs: dict, h, beta: float, T: int):
        """Roll out a beta-mixture of teacher and student; label every state with the teacher."""
        N = vec.n
        buf_o = np.zeros((T, N, obs["deploy"].shape[1]), np.float32)
        buf_a = np.zeros((T, N, self.teacher.head.out_features), np.float32)
        buf_d = np.zeros((T, N), np.float32)
        finished = []
        for t in range(T):
            od = torch.as_tensor(obs["deploy"], device=self.device).unsqueeze(0)
            op = torch.as_tensor(obs["priv"], device=self.device).unsqueeze(0)
            a_teacher, _ = self.teacher(op)
            a_student, h = self.student(od, h)
            use_teacher = torch.rand(N, 1, device=self.device) < beta
            act = torch.where(use_teacher, a_teacher[0], a_student[0])
            buf_o[t], buf_a[t] = obs["deploy"], a_teacher[0].cpu().numpy()
            obs, _, done, infos = vec.step(act.cpu().numpy())
            buf_d[t] = done
            finished += [i for i, d in zip(infos, done) if d]
            if h is not None:
                h = h * (1.0 - torch.as_tensor(done, dtype=torch.float32, device=self.device)).view(1, -1, 1)
        self.chunks.append((buf_o, buf_a, buf_d))
        return obs, h, finished

    def train(self, epochs: int, batch_rows: int = 32) -> float:
        obs = np.concatenate([c[0] for c in self.chunks], 1)
        act = np.concatenate([c[1] for c in self.chunks], 1)
        done = np.concatenate([c[2] for c in self.chunks], 1)
        self.student.norm.update(torch.as_tensor(obs, device=self.device))
        N = obs.shape[1]
        last = 0.0
        for _ in range(epochs):
            for mb in torch.randperm(N).chunk(max(1, N // batch_rows)):
                o = torch.as_tensor(obs[:, mb], device=self.device)
                a = torch.as_tensor(act[:, mb], device=self.device)
                d = torch.as_tensor(done[:, mb], device=self.device)
                h0 = self.student.init_hidden(len(mb), self.device)
                mean, _ = self.student(o, h0, d)
                loss = F.mse_loss(mean, a)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                last = loss.item()
        return last
