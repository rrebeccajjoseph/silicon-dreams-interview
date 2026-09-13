"""Hindsight relabeling for the delayed-target task (plan B).

Before lift the policy never sees the target, so a trajectory is on-policy for any
target we pick in hindsight. After lift the actions did depend on the target, so the
relabeled rows are off-policy there; PPO's ratio clipping keeps that in check and
the old log-probs are recomputed on the relabeled observations at update time."""

from __future__ import annotations

import numpy as np

from .. import frames
from ..config import RewardWeights, Success
from ..env import Signals, reward_fn
from ..observations import DEPLOY, PRIV
from .buffer import Rollout

SIG = {n: i for i, n in enumerate(["epos", "eang", "in_tol", "lifted", "lifted_now", "dropped",
                                    "violation", "success", "action_sq", "palm_obj_dist"])}


def _segments(done: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) inclusive spans of one env row, split at done flags."""
    out, s = [], 0
    for t, d in enumerate(done):
        if d:
            out.append((s, t))
            s = t + 1
    if s < len(done):
        out.append((s, len(done) - 1))
    return out


def hindsight_relabel(ro: Rollout, w: RewardWeights, succ: Success, max_rows: int) -> Rollout | None:
    assert ro.obs_d.shape[-1] == DEPLOY.dim, "relabeling edits the deploy target block"
    rows = []
    for n in range(ro.N):
        if ro.relabeled[n]:
            continue
        segs = [(s, e) for s, e in _segments(ro.done[:, n])
                if ro.hand_only[e, n] and ro.signals[e, n, SIG["lifted"]] and not ro.signals[s:e + 1, n, SIG["success"]].any()]
        if not segs:
            continue
        row = ro.rows(np.array([n]))
        for s, e in segs:
            p_t, a_t = ro.obj_palm[e, n, :3], ro.obj_palm[e, n, 3:]
            tgt = np.concatenate([p_t, frames.axis_outer(a_t)]).astype(np.float32)
            row.obs_p[s:e + 1, 0, PRIV["true_target"]] = tgt
            for t in range(s, e + 1):
                sig = ro.signals[t, n]
                lifted = bool(sig[SIG["lifted"]])
                if lifted:
                    row.obs_d[t, 0, DEPLOY["target"]] = np.concatenate([tgt, [1.0]])
                epos, eang = frames.errors(ro.obj_palm[t, n, :3], ro.obj_palm[t, n, 3:], p_t, a_t)
                in_tol = lifted and bool(ro.hand_only[t, n]) and epos <= succ.pos_tol and eang <= succ.ang_tol
                # the hold bonus is left out: the real episode did not stop here
                new = Signals(epos, eang, in_tol, lifted, bool(sig[SIG["lifted_now"]]), bool(sig[SIG["dropped"]]),
                              bool(sig[SIG["violation"]]), False, float(sig[SIG["action_sq"]]), float(sig[SIG["palm_obj_dist"]]))
                row.rew[t, 0] = reward_fn(w, new)
            if e == ro.T - 1:
                row.last_obs_p[0, PRIV["true_target"]] = tgt
        row.relabeled[:] = True
        rows.append(row)
        if len(rows) >= max_rows:
            break
    if not rows:
        return None
    out = rows[0]
    for r in rows[1:]:
        out = Rollout.concat(out, r)
    return out
