# In-hand cylinder reorientation with a delayed target

Pick a lying rod off the ground, lift it, receive a palm-frame target only after lift,
reorient it in the hand and hold it for 2 s. xArm7 + LEAP hand in MuJoCo 3.13.

**Submitted system (`PlanS`)**: scripted grasp, then a learned in-hand reorient policy.
One shape band, one start pose, measured end to end under the exact rules of the task.
The wider envelope is swept in eval only. Plans A, B and C in the repo share the same core
but are not part of the submission.

Report: `report/report.pdf` (assumptions, decomposition, MDP, grasp trade-off, envelope,
failure taxonomy, limitations, compute, and a requirement-by-requirement checklist).

## Results

Policy and baseline run on the same 200 objects and targets (seeds 123-126).

| System | episodes | handover | success (95 % CI) | success after handover | median e_pos / e_ang |
|---|---|---|---|---|---|
| scripted grasp + hold still | 200 | 99 (49.5 %) | 4.0 % (2.0-7.7) | 0 / 99 | 2.9 cm / 26 deg |
| scripted grasp + learned reorient | 200 | 99 (49.5 %) | **5.5 %** (3.1-9.6) | 3 / 99 | 2.9 cm / 25 deg |
| learned reorient, wide envelope grid | 105 | 13 (12 %) | 1.9 % (0.5-6.7) | 1 / 13 | 2.8 cm / 43 deg |

Success means e_pos <= 1.5 cm and e_ang <= 15 deg held for 2 s with the arm commanded fixed and
hand-only contact after lift. Far below the 50 % bar.

- 8 of the 11 learned successes, and all 8 baseline ones, happen during the grasp's wrist roll: the
  rod passes through tolerance, the arm freezes, it holds. Learned reorientation adds 3 (10/64 on
  the validation seeds).
- Where policy episodes end: the grasp pops the rod and it settles back onto the floor 44 %,
  timeout in hand 37.5 %, in-hand drop 7.5 %, arm contact 3 %, lost during the grasp 2.5 %.
- Hand joint speed exceeds the URDF 8.48 rad/s in 3.6 % of control steps. In a diagnostic every such
  sample had the hand in contact (back-driven transients); commanded speeds stay under the limits.

**Correction.** An earlier commit reported 8.5 % end to end. That version sampled contacts every
40 ms and had no hand torque limit; millimetre pops of the rod during the grasp fell between
samples. Its results are kept in `results/v1_loose/` for comparison only.

## What makes it spec-exact

- Lift and violations are checked after every 2 ms physics step (`inhand/env.py`), with a 1 mm
  object-ground contact band that is stricter in both directions (`scene.CONTACT_RESOLUTION`).
- Hand torque limited to the URDF's 0.95 N m, arm to Menagerie's 50/30/20 N m; setpoints ramp within
  each control period so commanded speeds stay under the URDF velocity limits; `eval.py` records
  measured peaks. The hand bracket is a solid collision geom.
- T* is zeros in the deploy observation until lift. The grasp and stage switch use only the lift
  flag, never the target values.
- Vision stand-in: 3 RGB-D cameras (640x480, 87x58 deg field of view), pose estimate at 5 Hz, only
  when >= 30 % of the surface is in view and unoccluded.
- Train/deploy split: checkpoints carry `actor_key`, `load_actor(deploy_only=True)` refuses privileged
  actors, `ActorRunner` asserts the input width equals `DEPLOY.dim`.

All declared assumptions live in `inhand/config.py` and `inhand/scene.py`.

## Setup

```bash
uv sync && uv run pytest -q
```

## Evaluate

```bash
uv run python eval.py            # submitted system, loads checkpoints/reorient_deploy.pt
uv run python eval.py hold       # same grasp, then hold still
```

`eval.py` writes every episode (geometry, start pose, termination reason, the stage the system was
in, final errors, peak joint speeds) plus a summary to `results/`.

## Reproduce (the commands run for the submitted results)

```bash
# 1. handover bank from the final scripted grasp (1440 episodes, 765 handovers)
uv run python scripts/collect_handovers.py --episodes 1440 --procs 16 --out checkpoints/v2/handovers.npz

# 2. reorient policy: deploy-obs recurrent actor, privileged critic, 70 % of resets from the bank
uv run python scripts/train.py reorient-teacher --fixed-targets --actor deploy \
    --bank checkpoints/v2/handovers.npz --bank-frac 0.7 --envs 56 --workers 14 --minutes 80 --out checkpoints/v2/train

# 3. pick the checkpoint on held-out seeds, then evaluate on the test seeds
cp checkpoints/v2/train/reorient_deploy.best.pt checkpoints/v2/cand_best.pt
cp checkpoints/v2/train/reorient_deploy.pt checkpoints/v2/cand_final.pt
SEED0=900 bash scripts/eval_parallel.sh val_best  4 30 s --reorient checkpoints/v2/cand_best.pt    # 12/120
SEED0=900 bash scripts/eval_parallel.sh val_final 4 30 s --reorient checkpoints/v2/cand_final.pt   # 8/120
cp checkpoints/v2/cand_best.pt checkpoints/reorient_deploy.pt
cp checkpoints/v2/handovers.npz checkpoints/handovers.npz
bash scripts/eval_parallel.sh s_hold   4 50 hold
bash scripts/eval_parallel.sh s_policy 4 50 s
uv run python eval.py s --grid --wide --n-r 4 --n-h 5 --per-cell 3 --name s_grid_wide

# 4. figures, videos, report
uv run python scripts/report_figs.py --policy results/s_policy.json --hold results/s_hold.json \
    --grid results/s_grid_wide.json --out report/figs
uv run python scripts/record_video.py s --episodes 6 --seed 123
uv run python scripts/record_video.py s --episodes 3 --seed 126 --want success
uv run python scripts/record_video.py s --episodes 2 --seed 125 --want success
uv run python scripts/record_video.py s --episodes 1 --seed 125 --want dropped
uv run python scripts/record_video.py s --episodes 1 --seed 126 --want violation_arm
uv run python scripts/record_video.py s --episodes 5 --seed 11 --wide
```

Videos show the workspace and a close-up with the target as a green ghost once it is revealed.
File names give diameter, height, start pose, outcome and, where recorded, the stage it happened in.

## Layout

```
eval.py                         the single evaluation entry point
inhand/config.py                declared assumptions: envelope, friction, sensors, physics, targets, reward
inhand/scene.py                 arm + hand + cylinder, palm frame, solver settings, joint limits, contact band
inhand/env.py                   per-physics-step lift detector, target reveal, hold phase, reward, bank resets
inhand/observations.py          DEPLOY (what ships) vs PRIV (critic only)
inhand/contacts.py              contact classes, tactile pads, camera visibility gate
inhand/tasks/reorient.py        reorient stage: arm held, fingers scaled
inhand/planc/scripted_grasp.py  the grasp script
inhand/systems.py               deployed systems; PlanS is the submission
inhand/rl/                      PPO (asymmetric critic, GRU), DAgger, relabeling, curricula
scripts/                        train, eval_parallel, collect_handovers, report_figs, record_video, remote/
report/                         report.pdf and its figures (figs/ from scripts/report_figs.py)
results/                        submitted results; v1_loose/ holds the superseded 40 ms results
```

## Compute

One laptop (Apple M5 Max, 18 cores), CPU MuJoCo, no GPU. Submitted policy: 80 min, 8.61M env steps
at about 1.8k steps/s on 14 worker processes; the checkpoint is the 5.59M-step snapshot. Superseded
runs about 8.8M steps. Grasp sweeps about 1 h, handover banks about 3 min, evals about 5 min.
NCSA Delta jobs were queued as a hedge, never started, and were cancelled.

## References

MuJoCo 3.13 and MuJoCo Menagerie (LEAP hand, xArm7); dexsuite `leap_hand_right.urdf` for the hand's
joint limits; PyTorch. Pinto et al. Asymmetric Actor Critic 2017; Chen et al. Visual Dexterity 2023;
Handa et al. DeXtreme 2023; Andrychowicz et al. HER 2017; Ross et al. DAgger 2011. LLM assistance
(Claude) for code and debugging.
