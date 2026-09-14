# In-hand cylinder reorientation with a delayed target

Pick a lying rod off the ground, lift it, receive a palm-frame target only after lift,
reorient it in the hand and hold it for 2 s. xArm7 + LEAP hand in MuJoCo 3.13.

**Submitted system (`PlanS`)**: scripted grasp, then a learned in-hand reorient policy.
One shape band, one start pose, measured end to end. The wider envelope is swept in eval only.
Plans A, B and C in the repo share the same core but are not part of the submission.

## Results

Report: `report/report.pdf`. Policy and baseline run on the same 200 objects and targets (seeds 123-126).

| System | episodes | handover | success (95 % CI) | success given handover | median e_pos / e_ang |
|---|---|---|---|---|---|
| scripted grasp + hold still | 200 | 120 (60 %) | 0.5 % (0.1-2.8) | 0 / 120 | 3.6 cm / 35 deg |
| scripted grasp + learned reorient | 200 | 120 (60 %) | **8.5 %** (5.4-13.2) | 16 / 120 | 3.3 cm / 25 deg |
| learned reorient, wide envelope grid | 105 | 25 (24 %) | 1.9 % (0.5-6.7) | 1 / 25 | 5.2 cm / 37 deg |

Success means e_pos <= 1.5 cm and e_ang <= 15 deg held for 2 s, with hand-only contact after lift.
Below the 50 % bar. Hold-still shows how much of the target box the grasp covers by luck: none of it.

Where policy episodes end: grasp pops the rod onto the floor 37.5 %, timeout out of tolerance 30 %,
arm contact 12.5 %, in-hand drop 9.5 %, success 8.5 %. Arm contacts and drops are the policy's own
(2 % and 4 % when holding still). Wide grid: only 4-5.5 cm lying rods get past the grasp; 2.4 cm
rods never hand over, 7 cm rods pop.

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

**The grasp caps the system.** 40 % of episodes never reach reorientation. A learned,
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
SEED0=900 bash scripts/eval_parallel.sh val_best  4 30 s --reorient checkpoints/local/cand_best.pt   # checkpoint pick,
SEED0=900 bash scripts/eval_parallel.sh val_final 4 30 s --reorient checkpoints/local/cand_final.pt  # held-out seeds
cp checkpoints/local/cand_best.pt checkpoints/reorient_deploy.pt
bash scripts/eval_parallel.sh s_hold   4 50 s
bash scripts/eval_parallel.sh s_policy 4 50 s --reorient checkpoints/reorient_deploy.pt
uv run python eval.py s --reorient checkpoints/reorient_deploy.pt --grid --wide --n-r 4 --n-h 5 --per-cell 3 --name s_grid_wide

# 4. figures, videos, report
uv run python scripts/report_figs.py --policy results/s_policy.json --hold results/s_hold.json \
    --grid results/s_grid_wide.json --out report/figs
uv run python scripts/record_video.py s --reorient checkpoints/reorient_deploy.pt --episodes 8 --seed 123
uv run python scripts/record_video.py s --reorient checkpoints/reorient_deploy.pt --episodes 3 --seed 124 --want success
uv run python scripts/record_video.py s --reorient checkpoints/reorient_deploy.pt --episodes 1 --seed 125 --want dropped
uv run python scripts/record_video.py s --reorient checkpoints/reorient_deploy.pt --episodes 6 --seed 11 --wide
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
    --print-to-pdf=report/report.pdf "file://$PWD/report/report.html"
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
report/                         report.html -> report.pdf, figs/
```

Train/deploy split: checkpoints carry `actor_key`. `load_actor(deploy_only=True)` refuses
privileged actors, and `ActorRunner` asserts the input width equals `DEPLOY.dim`. The grasp script
decodes only the deploy vector and never reads the target block.

## Also in the repo, not evaluated

Plans A (value-aware learned grasp), B (monolithic recurrent policy with hindsight relabeling)
and C (primitives + planner + tactile estimator). They were built and smoke tested but not trained
to convergence, so no claims are made about them.

## Compute

One laptop (Apple M5 Max, 18 cores), CPU MuJoCo, no GPU. Reorient training: 80 min, 7.55M env steps
at about 1.57k steps/s on 14 worker processes. The submitted checkpoint is the 5.53M-step snapshot,
picked over the final one on held-out seeds 900-903 (10.8 % vs 7.5 % end to end). Grasp sweeps
about 25 min, handover collection about a minute, all evals about 5 min on 13 processes. Delta
jobs were queued as a second seed but never started in time.

## References

MuJoCo Menagerie (LEAP, xArm7). Pinto et al. Asymmetric Actor Critic 2017; Chen et al. Visual
Dexterity 2023; Handa et al. DeXtreme 2023; Andrychowicz et al. HER 2017; Ross et al. DAgger 2011.
