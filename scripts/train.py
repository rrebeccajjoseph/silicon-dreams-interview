"""Training entry points. One subcommand per learned component.

  reorient-teacher   plan A/C shared: privileged PPO on the in-hand reorient task
  grasp-a            plan A: grasp policy rewarded by the reorient critic
  mono-b             plan B: whole task, masked target, hindsight relabeling
  skills-c           plan C: primitive skill policy (privileged teacher)
  distill            any privileged teacher -> deploy-obs student (DAgger)
  estimator-c        plan C: tactile/proprio state estimator
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checkpointer, Logger, base_parser  # noqa: E402

from inhand.config import Config  # noqa: E402
from inhand.observations import DEPLOY  # noqa: E402
from inhand.planc.estimator import Estimator, train_estimator  # noqa: E402
from inhand.rl.buffer import Rollout  # noqa: E402
from inhand.rl.curriculum import EnvelopeADR, TargetCurriculum  # noqa: E402
from inhand.rl.distill import Dagger  # noqa: E402
from inhand.rl.nets import Actor  # noqa: E402
from inhand.rl.ppo import PPOConfig, load_actor, make_ppo  # noqa: E402
from inhand.rl.relabel import hindsight_relabel  # noqa: E402
from inhand.scene import Scene  # noqa: E402
from inhand.tasks.grasp_value import GraspEnv  # noqa: E402
from inhand.tasks.monolithic import FullTaskEnv  # noqa: E402
from inhand.tasks.primitives import PrimitiveEnv  # noqa: E402
from inhand.tasks.reorient import ReorientEnv  # noqa: E402
from inhand.vec import SubprocVecEnv, VecEnv  # noqa: E402

N_ACT = 23


def budget(args):
    """Iteration generator that also honours --minutes."""
    t0 = time.time()
    for it in range(args.iters):
        yield it, (it == args.iters - 1) or (args.minutes > 0 and time.time() - t0 > 60 * args.minutes)
        if args.minutes > 0 and time.time() - t0 > 60 * args.minutes:
            return


def summarize(finished: list[dict]) -> dict:
    if not finished:
        return {"ep": 0}
    reasons = {}
    for f in finished:
        reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1
    return {"ep": len(finished), "succ": float(np.mean([f["success"] for f in finished])),
            "epos": float(np.median([f["epos"] for f in finished])),
            "eang_deg": float(np.degrees(np.median([f["eang"] for f in finished]))),
            "lifted": float(np.mean([f["lifted"] for f in finished])), "reasons": reasons}


def make_vec(args, cfg, env_cls, kwargs_fn):
    """kwargs_fn(i) -> constructor kwargs for env i. Processes if --workers, else threads."""
    kws = [dict(seed=1000 * args.seed + i, **kwargs_fn(i)) for i in range(args.envs)]
    if args.workers > 0:
        return SubprocVecEnv(env_cls, cfg, kws, args.workers)
    scene = Scene(cfg)
    return VecEnv([env_cls(scene, cfg, **kw) for kw in kws], args.threads)


def run_ppo(args, env_cls, kwargs_fn, name: str, actor_key: str, recurrent: bool, adr=None, tcur=None, relabel=False):
    cfg = Config(seed=args.seed)
    vec = make_vec(args, cfg, env_cls, kwargs_fn)
    if adr:
        adr.apply(vec)
    if tcur:
        tcur.apply(vec)
    ppo = make_ppo(N_ACT, PPOConfig(T=args.T), recurrent, args.device, actor_key)
    log = Logger(Path(args.out), name, args, cfg)
    ckpt = Checkpointer(Path(args.out), name, ppo.save, args.save_every)
    obs = vec.reset()
    h = ppo.actor.init_hidden(args.envs, args.device)
    steps = 0
    for it, last in budget(args):
        ro, obs, h, fin = ppo.collect(vec, obs, h)
        steps += ro.T * ro.N
        if relabel:
            extra = hindsight_relabel(ro, cfg.reward, cfg.success, max_rows=ro.N // 2)
            if extra is not None:
                ro = Rollout.concat(ro, extra)
        st = ppo.update(ro)
        s = summarize(fin)
        succ = [f["success"] for f in fin]
        for cur in (adr, tcur):
            if cur:
                cur.gate.observe(succ)
                if cur.gate.maybe_step():
                    cur.apply(vec)
        log.log(it, steps=steps, **st, **s, adr=adr.frac if adr else None, tcur=tcur.frac if tcur else None)
        # plan A's grasp has no success signal of its own; rank it by the value it earns
        score = s.get("succ") if "succ" in s else (st["ret_mean"] if fin else None)
        ckpt.step(it, score, last=last)
    log.finish()
    vec.close()
    return ppo


def cmd_reorient_teacher(args):
    cfg = Config()
    if args.fixed_targets:
        # train on the deployed target distribution directly; the deploy actor needs no distill step
        name = "reorient_teacher" if args.actor == "priv" else "reorient_deploy"
        kw = dict(bank_path=args.bank, bank_frac=args.bank_frac) if args.bank else {}
        run_ppo(args, ReorientEnv, lambda i: kw, name, actor_key=args.actor, recurrent=args.actor == "deploy")
        return
    run_ppo(args, ReorientEnv, lambda i: {}, "reorient_teacher", actor_key="priv", recurrent=False,
            adr=EnvelopeADR(cfg.envelope), tcur=TargetCurriculum(cfg.targets.ang_curriculum, cfg.targets.pos_curriculum))


def cmd_grasp_a(args):
    # the critic is loaded inside each env (or worker) so the reward works across processes
    run_ppo(args, GraspEnv, lambda i: dict(critic_path=args.critic, value_scale=args.value_scale),
            "grasp_a_teacher", actor_key="priv", recurrent=False)


def cmd_mono_b(args):
    cfg = Config()
    run_ppo(args, FullTaskEnv, lambda i: {}, "mono_b", actor_key="deploy", recurrent=True,
            adr=EnvelopeADR(cfg.envelope), relabel=not args.no_relabel)


def cmd_skills_c(args):
    cfg = Config()
    run_ppo(args, PrimitiveEnv, lambda i: {}, "skills_c_teacher", actor_key="priv", recurrent=False,
            adr=EnvelopeADR(cfg.envelope))


ENVS = {"reorient": ReorientEnv, "grasp": GraspEnv, "prims": PrimitiveEnv, "full": FullTaskEnv}


def cmd_distill(args):
    cfg = Config(seed=args.seed)
    teacher = load_actor(args.teacher, args.device)
    student = Actor(DEPLOY.dim, N_ACT, recurrent=True)
    vec = make_vec(args, cfg, ENVS[args.env], lambda i: {})
    dg = Dagger(teacher, student, args.device)
    log = Logger(Path(args.out), args.name, args, cfg)

    def save(path):
        torch.save({"actor": student.state_dict(), "actor_kw": dict(obs_dim=DEPLOY.dim, act_dim=N_ACT, recurrent=True),
                    "critic": {}, "critic_kw": {}, "actor_key": "deploy"}, path)

    ckpt = Checkpointer(Path(args.out), args.name, save, args.save_every)
    obs = vec.reset()
    h = student.init_hidden(args.envs, args.device)
    for it, last in budget(args):
        beta = max(0.0, 1.0 - it / max(1, args.iters // 2))  # teacher fades out over the first half
        obs, h, fin = dg.collect(vec, obs, h, beta, args.T)
        dg.chunks = dg.chunks[-args.keep_chunks:]
        loss = dg.train(epochs=args.epochs)
        s = summarize(fin)
        log.log(it, beta=beta, loss=loss, **s)
        # only student-driven episodes say anything about the student
        ckpt.step(it, s.get("succ") if beta == 0.0 else None, last=last)
    log.finish()
    vec.close()


def cmd_estimator_c(args):
    cfg = Config(seed=args.seed)
    policy = load_actor(args.policy, args.device)
    vec = make_vec(args, cfg, PrimitiveEnv, lambda i: {})
    est = Estimator()
    log = Logger(Path(args.out), "estimator_c", args, cfg)
    obs = vec.reset()
    h = policy.init_hidden(args.envs, args.device)
    chunks = []
    for it, last in budget(args):
        bo, bp, bd = [], [], []
        with torch.no_grad():
            for _ in range(args.T):
                key = "priv" if policy.norm.mean.shape[0] != DEPLOY.dim else "deploy"
                mean, h = policy(torch.as_tensor(obs[key], device=args.device).unsqueeze(0), h)
                act = (mean[0] + 0.2 * torch.randn_like(mean[0])).cpu().numpy()
                bo.append(obs["deploy"]); bp.append(obs["priv"])
                obs, _, done, _ = vec.step(act)
                bd.append(done.astype(np.float32))
                if h is not None:
                    h = h * (1.0 - torch.as_tensor(done, dtype=torch.float32, device=args.device)).view(1, -1, 1)
        chunks.append((np.stack(bo), np.stack(bp), np.stack(bd)))
        chunks = chunks[-args.keep_chunks:]
        loss = train_estimator(est, *[np.concatenate([c[k] for c in chunks], 1) for k in range(3)], epochs=args.epochs, device=args.device)
        log.log(it, loss=loss)
        torch.save(est.state_dict(), Path(args.out) / "estimator_c.pt")
    log.finish()
    vec.close()


def main():
    p = base_parser(__doc__)
    p.add_argument("cmd", choices=["reorient-teacher", "grasp-a", "mono-b", "skills-c", "distill", "estimator-c"])
    p.add_argument("--critic", help="grasp-a: reorient teacher checkpoint (its critic is used)")
    p.add_argument("--value-scale", type=float, default=1.0)
    p.add_argument("--actor", choices=["priv", "deploy"], default="priv", help="reorient-teacher --fixed-targets")
    p.add_argument("--fixed-targets", action="store_true", help="reorient-teacher: no curricula, deployed target prior")
    p.add_argument("--bank", help="handover states from scripts/collect_handovers.py for in-hand resets")
    p.add_argument("--bank-frac", type=float, default=0.7)
    p.add_argument("--no-relabel", action="store_true", help="mono-b ablation")
    p.add_argument("--teacher", help="distill: privileged teacher checkpoint")
    p.add_argument("--policy", help="estimator-c: policy used to generate data")
    p.add_argument("--env", choices=list(ENVS), default="reorient")
    p.add_argument("--name", default="student")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--keep-chunks", type=int, default=20)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    {"reorient-teacher": cmd_reorient_teacher, "grasp-a": cmd_grasp_a, "mono-b": cmd_mono_b,
     "skills-c": cmd_skills_c, "distill": cmd_distill, "estimator-c": cmd_estimator_c}[args.cmd](args)


if __name__ == "__main__":
    main()
