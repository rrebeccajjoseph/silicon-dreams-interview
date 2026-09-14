"""Evaluate a deployed system on the full task and sweep the geometry envelope.

  python scripts/eval.py a --grasp ckpt/grasp_a_student.pt --reorient ckpt/reorient_student.pt
  python scripts/eval.py b --policy ckpt/mono_b.pt
  python scripts/eval.py c --skills ckpt/skills_c_student.pt --estimator ckpt/estimator_c.pt
  python scripts/eval.py c ... --grasp ckpt/grasp_a_student.pt     # C with A's learned grasp
  python scripts/eval.py s --reorient ckpt/reorient_deploy.pt   # scripted grasp + learned reorient
  python scripts/eval.py s               # scripted grasp, then hold still (baseline)
  python scripts/eval.py random          # sanity baseline

Writes results/<plan>.json with every episode and prints the envelope table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inhand.config import WIDE, Config  # noqa: E402
from inhand.cylinder import ObjectParams, allowed_start_poses  # noqa: E402
from inhand.scene import Scene  # noqa: E402
from inhand.systems import PlanA, PlanB, PlanC, PlanS, System  # noqa: E402
from inhand.tasks.monolithic import FullTaskEnv  # noqa: E402



class RandomSystem(System):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return self.rng.uniform(-1, 1, 23) * 0.3


def build(args, scene, cfg) -> System:
    if args.plan == "a":
        return PlanA(args.grasp, args.reorient)
    if args.plan == "b":
        return PlanB(args.policy)
    if args.plan == "c":
        return PlanC(scene, cfg, args.skills, args.estimator, grasp_ckpt=args.grasp)
    if args.plan == "s":
        return PlanS(scene, cfg, args.reorient)
    return RandomSystem(args.seed)


def run_episode(env: FullTaskEnv, system: System, obj: ObjectParams | None, pose: str | None) -> dict:
    obs = env.reset(obj, pose)
    system.reset()
    info = {}
    while True:
        obs, _, done, info = env.step(system.act(obs["deploy"]))
        if done:
            break
    stage = "pre_lift" if env.ep.lift_step < 0 else "post_lift"
    return {
        "r": env.obj.r, "h": env.obj.h, "alpha": env.obj.alpha, "mass": env.obj.mass,
        "start_pose": env.ep.start_pose, "success": bool(info["success"]), "reason": info["reason"],
        "stage": "success" if info["success"] else stage, "epos": info["epos"], "eang_deg": float(np.degrees(info["eang"])),
        "lift_step": env.ep.lift_step, "steps": info["t"],
        "system_stage": getattr(system, "stage", None),
    }


def grid(cfg: Config, n_r: int, n_h: int, per_cell: int, rng, e=None) -> list[tuple[ObjectParams, str]]:
    e = e or cfg.envelope
    cells = []
    for r in np.linspace(e.r_min, e.r_max, n_r):
        for h in np.linspace(e.h_min, e.h_max, n_h):
            for _ in range(per_cell):
                o = ObjectParams(r, h, rng.uniform(e.density_min, e.density_max),
                                 rng.uniform(*cfg.friction.obj), rng.uniform(*cfg.friction.ground))
                for pose in allowed_start_poses(e, o):
                    cells.append((o, pose))
    return cells


def report(rows: list[dict]) -> dict:
    succ = np.array([r["success"] for r in rows])
    out = {"n": len(rows), "success_rate": float(succ.mean())}
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    out["failure_taxonomy"] = {k: v / len(rows) for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])}
    out["per_stage"] = {s: float(np.mean([r["stage"] == s for r in rows])) for s in ("success", "pre_lift", "post_lift")}
    lifted = [r for r in rows if r["lift_step"] >= 0]
    out["lift_rate"] = len(lifted) / len(rows)
    if lifted:
        out["post_lift_success_given_lift"] = float(np.mean([r["success"] for r in lifted]))
        out["epos_cm_pct"] = {p: float(np.percentile([r["epos"] * 100 for r in lifted], p)) for p in (10, 50, 90)}
        out["eang_deg_pct"] = {p: float(np.percentile([r["eang_deg"] for r in lifted], p)) for p in (10, 50, 90)}
    # envelope: success by aspect ratio bin x radius bin x start pose
    a_bins = [0, 0.5, 1.0, 2.0, 4.0, 1e9]
    env = {}
    for pose in sorted({r["start_pose"] for r in rows}):
        for lo, hi in zip(a_bins[:-1], a_bins[1:]):
            sel = [r for r in rows if r["start_pose"] == pose and lo <= r["alpha"] < hi]
            if sel:
                env[f"{pose} alpha[{lo},{hi})"] = {"n": len(sel), "success": float(np.mean([r["success"] for r in sel]))}
    out["envelope"] = env
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", nargs="?", choices=["a", "b", "c", "s", "random"])
    p.add_argument("--grasp"); p.add_argument("--reorient"); p.add_argument("--policy")
    p.add_argument("--skills"); p.add_argument("--estimator")
    p.add_argument("--episodes", type=int, default=100, help="random episodes from the envelope")
    p.add_argument("--grid", action="store_true", help="sweep a radius x height grid instead")
    p.add_argument("--wide", action="store_true", help="grid over the WIDE envelope, both start poses")
    p.add_argument("--name", default=None, help="results file name (default plan_<plan>)")
    p.add_argument("--n-r", type=int, default=4); p.add_argument("--n-h", type=int, default=5)
    p.add_argument("--per-cell", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="results")
    p.add_argument("--merge", nargs="+", help="combine result files from parallel runs into --name instead of running")
    args = p.parse_args()
    if args.merge:
        rows = [e for f in args.merge for e in json.load(open(Path(args.out) / f"{f}.json"))["episodes"]]
        with open(Path(args.out) / f"{args.name}.json", "w") as f:
            json.dump({"summary": report(rows), "episodes": rows}, f, indent=1)
        print(json.dumps(report(rows), indent=1))
        return

    cfg = Config(seed=args.seed)
    scene = Scene(cfg)
    env = FullTaskEnv(scene, cfg, seed=args.seed)
    system = build(args, scene, cfg)
    rng = np.random.default_rng(args.seed)
    cases = grid(cfg, args.n_r, args.n_h, args.per_cell, rng, WIDE if args.wide else None) if args.grid else [(None, None)] * args.episodes
    rows = []
    for i, (obj, pose) in enumerate(cases):
        rows.append(run_episode(env, system, obj, pose))
        r = rows[-1]
        print(f"[{i + 1}/{len(cases)}] alpha={r['alpha']:.2f} r={r['r']:.3f} {r['start_pose']:8s} -> {r['reason']:16s} "
              f"epos={r['epos'] * 100:.1f}cm eang={r['eang_deg']:.0f}deg", flush=True)
    summary = report(rows)
    Path(args.out).mkdir(exist_ok=True)
    with open(Path(args.out) / f"{args.name or 'plan_' + args.plan}.json", "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
