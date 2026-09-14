# In-hand cylinder reorientation with a delayed target

Pick a lying rod off the ground, lift it, receive a palm-frame target only after lift,
reorient it in the hand and hold it for 2 s. xArm7 + LEAP hand in MuJoCo 3.13.

**Submitted system (`PlanS`)**: scripted grasp, then a learned in-hand reorient policy.
One shape band, one start pose, measured end to end. The wider envelope is swept in eval only.
Plans A, B and C in the repo share the same core but are not part of the submission.

## Status

The reorient policy is still training (local run, 80 min, ends around 11:30 on 2026-09-14).
The numbers below come from the scripted grasp, the hold-still baseline, and a 2M-step snapshot
of the policy. Final 200-episode evals, videos and the PDF report follow once training finishes.

## Results so far

| System | episodes | handover | success (95 % CI) | median e_pos / e_ang |
|---|---|---|---|---|
| scripted grasp alone | 480 | 266 (55 %) | - | - |
| scripted grasp + hold still | 100 | - | 0 % (0-3.7) | 3.6 cm / 34 deg |
| scripted grasp + learned reorient, 2M steps | 60 | - | 5 % (1.7-13.7) | 4.0 cm / 33 deg |
| hold still, wide envelope grid | 105 | - | 1 % | 5.9 cm / 36 deg |

Success means e_pos <= 1.5 cm and e_ang <= 15 deg held for 2 s, with hand-only contact after lift.
Hold-still is the baseline the policy has to beat: it shows how much of the target box the grasp
already covers by luck (none of it).

Where episodes end, 2M-step policy: timeout out of tolerance 38 %, ground contact after lift 28 %
(mostly the grasp popping the rod), drop 15 %, arm contact 12 %, success 5 %.

## Scope and assumptions

- Object: lying cylinders, r 2.2-2.8 cm, h 8-12 cm (alpha 1.4-2.7), density 500-900 kg/m^3.
- Friction: object 0.7-1.1, ground 0.5-0.9, sampled per episode.
- Target T*: axis within 40 deg of palm y, position in a 3 x 3 cm box above the palm. Fixed in the
  palm frame and sampled independently of the grasp.
- Physics: 2 ms step, implicit-fast, elliptic cones, impratio 10, softer object contact.
- Wide envelope (eval only): r 1.2-3.5 cm, h 2-16 cm, density 300-1200, lying and standing (`config.WIDE`).

All of it is declared in `inhand/config.py`.

## What moved the numbers

**Grasp, 1 in 12 -> 55 % handover.** The old script's usual failure was a pop: the closing fingers
flicked the rod off the floor, which counts as lift, and it landed as a violation. Most of that was
the 4 ms physics step, which was stable but not accurate. Going to 2 ms with softer contacts,
pressing the palm into the rod before closing, and rising fast when the lift flag flips mid-close
got it to 266 handovers in 480 episodes.

**Reorient training.** The actor is recurrent and sees only deploy observations; the critic is
privileged. There is no distill step. 70 % of resets come from the bank of real handover states,
so the policy trains on the poses it is actually deployed on. Only the fingers act, at 0.4x
scale, since the target is palm-frame and full-scale finger noise threw the rod out. All shaping is
positive and bounded: with negative per-step error, ending the episode early was the best policy.

**The grasp caps the system.** 45 % of episodes never reach reorientation. A learned,
value-aware grasp (plan A) is the change with the biggest expected gain.

## Setup

```bash
uv sync && uv run pytest -q
```

## Reproduce

```bash
# 1. handover bank from the scripted grasp
uv run python scripts/collect_handovers.py --episodes 480 --procs 5 --out checkpoints/handovers.npz

# 2. reorient policy
uv run python scripts/train.py reorient-teacher --fixed-targets --actor deploy \
    --bank checkpoints/handovers.npz --bank-frac 0.7 --envs 56 --workers 14 --minutes 80 --out checkpoints/local

# 3. eval (same seeds give the same objects and targets for every system)
uv run python eval.py s --episodes 100 --name s_hold
bash scripts/eval_parallel.sh s_policy 4 50 s --reorient checkpoints/local/reorient_deploy.best.pt
uv run python eval.py s --grid --wide --n-r 4 --n-h 5 --per-cell 3 --name s_grid_wide_hold

# 4. figures and videos
uv run python scripts/report_figs.py --policy results/s_policy.json --hold results/s_hold.json \
    --grid results/s_grid_wide.json --out report/figs
uv run python scripts/record_video.py s --reorient checkpoints/local/reorient_deploy.best.pt --episodes 6
```

`eval.py` writes every episode (geometry, start pose, termination reason, the stage the system
was in, final errors) plus a summary with the failure taxonomy to `results/`.

## Layout

```
inhand/config.py                declared assumptions: envelope, friction, physics, targets, reward
inhand/scene.py                 arm + hand + cylinder, palm frame, solver settings
inhand/env.py                   lift detector, target reveal, hold phase, reward, handover-bank resets
inhand/observations.py          DEPLOY (what ships) vs PRIV (critic only)
inhand/contacts.py              contact classes, tactile pads, camera visibility gate
inhand/tasks/reorient.py        reorient stage: arm held, fingers scaled
inhand/planc/scripted_grasp.py  the grasp script
inhand/systems.py               deployed systems; PlanS is the submission
inhand/rl/                      PPO (asymmetric critic, GRU), DAgger, relabeling, curricula
scripts/                        train, eval, eval_parallel, collect_handovers, report_figs, record_video
report/report.html              report draft
```

Train/deploy split: checkpoints carry `actor_key`. `load_actor(deploy_only=True)` refuses
privileged actors, and `ActorRunner` asserts the input width equals `DEPLOY.dim`. The grasp script
decodes only the deploy vector and never reads the target block.

## Also in the repo, not evaluated

Plans A (value-aware learned grasp), B (monolithic recurrent policy with hindsight relabeling)
and C (primitives + planner + tactile estimator). They were built and smoke tested but not trained
to convergence, so no claims are made about them.

## Compute

One laptop (Apple M5 Max, 18 cores), CPU MuJoCo. Reorient training runs at about 1.7k env
steps/s with 14 worker processes, so roughly 8M steps in 80 min. Handover collection took about
a minute on 5 processes. Delta jobs were queued as a second seed but never started in time.

## References

MuJoCo Menagerie (LEAP, xArm7). Pinto et al. Asymmetric Actor Critic 2017; Chen et al. Visual
Dexterity 2023; Handa et al. DeXtreme 2023; Andrychowicz et al. HER 2017; Ross et al. DAgger 2011.
