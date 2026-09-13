"""Two curricula: ADR-style widening of the geometry envelope, and target difficulty."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..config import Envelope
from ..vec import VecEnv


class SuccessGate:
    """Steps a scalar level up/down from the recent success rate."""

    def __init__(self, lo: float, hi: float, step: float, window: int = 200,
                 up_at: float = 0.6, down_at: float = 0.3, start: float | None = None):
        self.lo, self.hi, self.step = lo, hi, step
        self.level = lo if start is None else start
        self.recent: deque[float] = deque(maxlen=window)
        self.up_at, self.down_at = up_at, down_at

    def observe(self, successes: list[bool]) -> None:
        self.recent.extend(float(s) for s in successes)

    def maybe_step(self) -> bool:
        if len(self.recent) < self.recent.maxlen:
            return False
        rate = float(np.mean(self.recent))
        if rate >= self.up_at and self.level < self.hi:
            self.level = min(self.hi, self.level + self.step)
        elif rate <= self.down_at and self.level > self.lo:
            self.level = max(self.lo, self.level - self.step)
        else:
            return False
        self.recent.clear()
        return True


class EnvelopeADR:
    def __init__(self, full: Envelope, start_frac: float = 0.25, step: float = 0.1):
        self.full = full
        self.gate = SuccessGate(start_frac, 1.0, step)

    def apply(self, vec: VecEnv) -> None:
        vec.set_attr("envelope", self.full.scaled(self.gate.level))

    @property
    def frac(self) -> float:
        return self.gate.level


class TargetCurriculum:
    def __init__(self, ang: tuple[float, float], pos: tuple[float, float], step: float = 0.1):
        self.ang, self.pos = ang, pos
        self.gate = SuccessGate(0.0, 1.0, step)

    def apply(self, vec: VecEnv) -> None:
        f = self.gate.level
        vec.set_attr("curriculum", (self.ang[0] + f * (self.ang[1] - self.ang[0]),
                                    self.pos[0] + f * (self.pos[1] - self.pos[0])))

    @property
    def frac(self) -> float:
        return self.gate.level
