# In-hand cylinder reorientation with a delayed target

Pick a cylinder of random aspect ratio off the ground, lift it, receive a palm-frame
target only after lift, reorient in hand, hold 2 s. Three decompositions on one MuJoCo
core so they can be compared with the same evaluator.

| Plan | Grasp | Reorient | Idea |
|---|---|---|---|
| A | learned, reward = reorient critic's value averaged over the target prior | teacher-student PPO | grasp maximises E[V] since T* is unknown at grasp time |
| B | one recurrent policy, target masked until lift | same policy | hindsight relabeling is exact pre-lift; critic sees T* throughout |
| C | scripted IK + tip-over | 11 learned primitives + planner + tactile GRU estimator | options over an SMDP, explicit estimation |

Robot: xArm7 + LEAP right hand (MuJoCo Menagerie, vendored in `assets/`). MuJoCo 3.13, PyTorch, custom PPO.

## Setup

```bash
uv sync && uv run pytest -q
```

## Commands run

```bash
# laptop smoke (2 iters each) and the cluster matrix (scripts/remote/delta_submit.sh, one GPU share per job, time-boxed)
uv run python scripts/train.py reorient-teacher --envs 64 --workers 16 --minutes 45 --wandb
uv run python scripts/train.py skills-c         --envs 64 --workers 16 --minutes 35 --wandb
uv run python scripts/train.py mono-b           --envs 64 --workers 16 --minutes 80 --wandb
uv run python scripts/train.py mono-b --no-relabel ... --out checkpoints/ablation
uv run python scripts/train.py grasp-a  --critic checkpoints/reorient_teacher.pt --minutes 30
uv run python scripts/train.py distill  --teacher checkpoints/reorient_teacher.pt --env reorient --name reorient_student --minutes 15
uv run python scripts/train.py distill  --teacher checkpoints/grasp_a_teacher.pt  --env grasp    --name grasp_a_student  --minutes 15
uv run python scripts/train.py distill  --teacher checkpoints/skills_c_teacher.pt --env prims    --name skills_c_student --minutes 15
uv run python scripts/train.py estimator-c --policy checkpoints/skills_c_student.pt --minutes 10

uv run python scripts/eval.py a --grasp checkpoints/grasp_a_student.pt --reorient checkpoints/reorient_student.pt --grid
uv run python scripts/eval.py b --policy checkpoints/mono_b.pt --grid
uv run python scripts/eval.py c --skills checkpoints/skills_c_student.pt --estimator checkpoints/estimator_c.pt --grid
uv run python scripts/eval.py c ... --grasp checkpoints/grasp_a_student.pt   # C with A's grasp
uv run python scripts/eval.py random --grid
uv run python scripts/record_video.py a|b|c ... --episodes 4
```

`eval.py` writes `results/plan_<x>.json`: every episode, success rate, error percentiles,
failure taxonomy with pre/post-lift attribution, success by aspect-ratio bin and start pose.
W&B: `ftn-ebm-lab/cylinder-reorient-rl`.

## Layout

```
inhand/config.py        all declared assumptions (below)
inhand/scene.py         arm + hand + cylinder, palm frame, keyframes
inhand/cylinder.py      object sampling, stable start poses
inhand/frames.py        palm-frame math, sign-invariant axis encoding, metrics
inhand/contacts.py      contact taxonomy, tactile pads, visibility gate
inhand/observations.py  DEPLOY (ships) vs PRIV (critics/teachers) layouts
inhand/env.py           lift detector, target reveal, hold phase, reward
inhand/vec.py           thread and subprocess vector envs
inhand/tasks/           reorient (shared), grasp_value (A), monolithic (B), primitives (C)
inhand/rl/              PPO (asymmetric critic, GRU), relabeling, DAgger, curricula
inhand/planc/           estimator, planner, scripted grasp
inhand/systems.py       deployed systems; DEPLOY input only, stage switch = target flag
scripts/                train.py, eval.py, record_video.py, remote/ (cluster)
```

Train/deploy split: checkpoints carry `actor_key`; `load_actor(deploy_only=True)` refuses
privileged teachers; `systems.py` asserts input width == `DEPLOY.dim`.

## Decomposition and interfaces

- Stage switch: the world sets the target-available flag at the lift instant; every system switches on that flag.
- Lift: object touches at least one hand geom and nothing else, checked at the control rate.
- A: grasp episode ends 10 steps after a stable lift; terminal reward = mean over 16 sampled T* of the privileged reorient critic.
- B: deploy target block is zeros pre-lift; critic reads `true_target` always; rollouts ending in-hand are relabeled with their final pose, rewards recomputed through the pure `reward_fn`.
- C: scripted grasp -> 60-step wrist roll to palm-up -> planner picks a primitive every 30 steps from the estimator's pose using nominal primitive effects (depth-3 search).

## MDPs

Obs (deploy): arm/hand q, dq, previous action, targets, 5 tactile pads x (contact, normal force, centroid), palm pose from kinematics, gated vision (pos, axis outer product, r, h, flag), target (pos, axis outer, flag), skill one-hot, time.
Obs (privileged, train only): all of the above + object pose/velocity in palm frame, r, h, mass, frictions, contact flags, true target, lifted.
Action: 23 delta joint-position targets in [-1, 1], arm 0.04 rad/step, hand 0.25 rad/step.
Reward: post-lift exp(-epos/3cm) + exp(-eang/0.4rad) + 2·in_tol + 50·success; pre-lift 0.1·exp(-dist/10cm) + 30·lift; -10 drop or violation; small action penalty.
Termination: success, non-hand contact after lift, 0.25 s without hand contact, object >0.5 m away, horizon.
Curricula: envelope widens (ADR) and target offset widens on 60 % success over 200 episodes.

## Assumptions

- Palm frame: origin on the palm face under the fingers, x to fingertips, y across fingers, z outward normal. Hand on a 40 mm bracket past the flange (without it long rods hit the wrist link).
- Envelope G: r 1.2-3.5 cm, h 2-16 cm independent (alpha 0.3-6.7); finger span ~9 cm, reach ~12 cm. Density 300-1200 kg/m3. Friction: object 0.5-1.2, ground 0.4-1.0. Standing starts for alpha <= 3, lying for alpha >= 0.15. Start position x 0.36-0.56 m, y +-0.14 m.
- Tactile: 4 fingertips + palm; contact flag, summed normal force, force-weighted centroid in pad frame; control rate; noiseless; all contacts count.
- Vision stand-in: ground-truth pose + (r, h) only when >= 30 % of 32 surface samples are unoccluded from any of two fixed cameras at (1.1, +-0.7, 0.9) m or a wrist camera, ray-cast at 5 Hz; otherwise last value with flag 0.
- Control: 4 ms physics, 25 Hz policy, Menagerie position actuators, no latency. Elliptic cones, impratio 100, implicit-fast, cylinder condim 4.
- T*: axis uniform on the sphere; position x -2..5 cm, y +-3 cm, z from palm clearance to +2 cm; feasible if within 8 cm of the palm origin and no palm/wrist penetration with fingers open.
- Success: 1.5 cm, 15 deg, 2 s hold with arm targets frozen. Hand-floor contact allowed and logged.
- Estimator (C): GRU on the deploy stream with the target zeroed, outputs pose + (r, h); trained supervised on skill rollouts.

## Compute

CPU MuJoCo, ~520 env-steps/s per job with 64 envs across 16 worker processes on a Delta
A100 node share. Matrix: 10 dependent Slurm jobs, ~3 h end to end, ~8 GPU-hours.

## Status and known failures

- Reward sign matters: negative per-step error made early termination optimal (97 % arm contacts); positive bounded shaping fixed it. An approach reward of 1/step made the plan B policy hover instead of lift; 0.1/step with a 30 lift bonus fixed that.
- Scripted grasp (C) lifts ~1 in 3 lying cylinders; failure = squeeze bounce that counts as lift then ground contact.
- No standing-disk scripted grasp; in-hand resets bias toward poses resting on the palm; timeouts terminal in GAE.
- Next: longer budgets (curricula never advanced in 45 min), MJX port for 100x envs, learned grasp for C.

## References

MuJoCo Menagerie (LEAP, xArm7). Chen et al. Visual Dexterity 2023; Handa et al. DeXtreme 2023;
Andrychowicz et al. HER 2017; Pinto et al. Asymmetric Actor Critic 2017; Ross et al. DAgger 2011;
Sutton et al. Options 1999.
