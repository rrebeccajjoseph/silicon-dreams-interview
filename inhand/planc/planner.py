"""Depth-limited search over primitive sequences on the nominal skill effects."""

from __future__ import annotations

import itertools


from .. import frames
from ..config import Success
from ..tasks.primitives import HOLD, N_PRIMITIVES, apply_primitive


def cost(p, a, p_t, a_t, succ: Success) -> float:
    epos, eang = frames.errors(p, a, p_t, a_t)
    return epos / succ.pos_tol + eang / succ.ang_tol


def plan(p, a, r, h, p_t, a_t, succ: Success, depth: int = 3, step_cost: float = 0.05) -> int:
    """Returns the first primitive of the cheapest sequence. HOLD once inside tolerance."""
    if cost(p, a, p_t, a_t, succ) < 1.0 and all(
        e <= tol for e, tol in zip(frames.errors(p, a, p_t, a_t), (succ.pos_tol, succ.ang_tol))):
        return HOLD
    moves = [k for k in range(N_PRIMITIVES) if k != HOLD]
    best, best_c = HOLD, cost(p, a, p_t, a_t, succ)
    for d in range(1, depth + 1):
        for seq in itertools.product(moves, repeat=d):
            q, b = p, a
            for k in seq:
                q, b = apply_primitive(k, q, b, r, h)
            c = cost(q, b, p_t, a_t, succ) + step_cost * d
            if c < best_c:
                best, best_c = seq[0], c
    return best
