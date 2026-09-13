# In-hand cylinder reorientation with a delayed target

Pick a cylinder of unknown aspect ratio off the ground, lift it, and only then learn
the commanded palm-frame pose. Reorient in hand and hold. Three decompositions share
one simulator core and one evaluation harness so they can be compared on equal terms.

| Plan | Grasp | Reorient | The RL idea it leans on |
|---|---|---|---|
| A | learned, rewarded by the reorient critic | learned, teacher-student | option value: the grasp maximises E over the target prior of V_reorient at lift |
| B | one recurrent policy | same policy | hindsight relabeling is exact before lift; the critic sees the target the actor cannot |
| C | scripted IK + extrinsic tip-over | primitive skills + planner + tactile estimator | semi-MDP over learned options, explicit state estimation |

Robot: UFactory xArm7 with a LEAP right hand (MuJoCo Menagerie). Simulator: MuJoCo 3.13.

## Setup

```bash
uv sync                                   # python 3.12, mujoco, torch, numpy, imageio
uv run pytest -q                          # 10 tests, ~2 s
```

Robot assets are a sparse checkout of `google-deepmind/mujoco_menagerie` under
`assets/menagerie` (`leap_hand`, `ufactory_xarm7`). Re-fetch with:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie assets/menagerie
cd assets/menagerie && git sparse-checkout set leap_hand ufactory_xarm7
```

## Layout

```
inhand/
  config.py          every declared assumption: envelope, sensors, rates, tolerances, rewards
  scene.py           xArm7 + LEAP + cylinder + floor; palm frame; home and palm-up keyframes
  cylinder.py        object sampling, mass/inertia, stable start poses, surface points
  frames.py          palm-frame transforms, sign-invariant axis encoding, error metrics
  contacts.py        contact taxonomy, tactile pads, camera-visibility gate
  observations.py    DEPLOY (ships) and PRIV (critics and teachers) layouts
  env.py             episode logic: lift detector, target reveal, hold phase, pure reward fn
  vec.py             threaded vector env
  tasks/             reorient (shared), grasp_value (A), monolithic (B), primitives (C)
  rl/                PPO with asymmetric critic, GRU actor, hindsight relabeling, DAgger, curricula
  planc/             estimator, planner, scripted grasp
  systems.py         the three deployed systems; they read DEPLOY vectors and nothing else
scripts/
  train.py           one subcommand per learned component
  eval.py            full task, envelope sweep, failure taxonomy with per-stage attribution
  record_video.py    mp4 clips
```

The train/deploy split is enforced in code: `Actor` checkpoints carry an `actor_key`,
and `load_actor(..., deploy_only=True)` refuses privileged teachers. `systems.py`
asserts every deployed actor's input width equals `DEPLOY.dim`.

## Commands

Everything below was run for a two-iteration smoke test on a laptop. On the cluster the
same commands run with `--workers 64 --minutes <budget>`; see `scripts/remote/`.

```bash
# shared: privileged reorient teacher (plans A and C reuse its critic / structure)
uv run python scripts/train.py reorient-teacher --envs 64 --iters 1500 --out checkpoints
uv run python scripts/train.py distill --teacher checkpoints/reorient_teacher.pt --env reorient --name reorient_student --iters 200

# plan A
uv run python scripts/train.py grasp-a --critic checkpoints/reorient_teacher.pt --envs 64 --iters 800
uv run python scripts/train.py distill --teacher checkpoints/grasp_a_teacher.pt --env grasp --name grasp_a_student --iters 200
uv run python scripts/eval.py a --grasp checkpoints/grasp_a_student.pt --reorient checkpoints/reorient_student.pt --grid

# plan B (add --no-relabel for the ablation)
uv run python scripts/train.py mono-b --envs 64 --iters 3000
uv run python scripts/eval.py b --policy checkpoints/mono_b.pt --grid

# plan C
uv run python scripts/train.py skills-c --envs 64 --iters 800
uv run python scripts/train.py distill --teacher checkpoints/skills_c_teacher.pt --env prims --name skills_c_student --iters 200
uv run python scripts/train.py estimator-c --policy checkpoints/skills_c_student.pt --iters 100
uv run python scripts/eval.py c --skills checkpoints/skills_c_student.pt --estimator checkpoints/estimator_c.pt --grid
uv run python scripts/eval.py c ... --grasp checkpoints/grasp_a_student.pt     # C with A's learned grasp

# baseline and clips
uv run python scripts/eval.py random --grid
uv run python scripts/record_video.py a --grasp ... --reorient ... --episodes 5
```

`eval.py` writes `results/plan_<x>.json` with every episode and prints success rate,
error percentiles over lifted episodes, the failure taxonomy, pre/post-lift attribution,
and success by aspect-ratio bin and start pose.

## Assumptions

Everything here is a field in `inhand/config.py` unless noted.

**Palm frame.** Site `palm_frame` on the LEAP palm body: origin on the palm face under
the fingers, x toward the fingertips, y across the fingers, z the outward palm normal.
The hand sits on a 40 mm bracket beyond the xArm7 flange, like a real LEAP mount; without
it, long rods lying across the palm touch the wrist link and every episode fails.

**Envelope G.** Radius 1.2 to 3.5 cm, height 2 to 16 cm, sampled independently, so
aspect ratio spans about 0.3 to 6.7. Finger span across index to ring is about 9 cm and
fingertip reach about 12 cm. Density 300 to 1200 kg/m3. Object friction 0.5 to 1.2, ground
0.4 to 1.0. Standing starts are sampled only for alpha <= 3, lying starts only for
alpha >= 0.15. Position uniform in x 0.36 to 0.56 m, y -0.14 to 0.14 m from the arm base.

**Tactile.** Five pads: four fingertips and the palm. Each reports a contact flag, summed
normal force, and the force-weighted contact centroid in the pad body frame, at the
control rate. All contacts count, not only the object. This is a uSkin-class patch with
centroid extraction, assumed noiseless.

**Vision stand-in.** Ground-truth palm-frame pose plus radius and height are exposed
only when at least 30 percent of 32 area-weighted surface samples are unoccluded from
one of three cameras: two fixed at (1.1, +-0.7, 0.9) m and a wrist camera on a bracket
behind the palm. Occlusion is a ray cast against the full scene, refreshed at 5 Hz. When gated off, the
last visible estimate is passed with a zero flag.

**Control.** 4 ms physics, 25 Hz policy (10 substeps), delta joint-position targets
clipped to the URDF ranges (arm 0.04 rad per step, hand 0.25 rad per step). 2 ms was
checked to behave the same and is 1.65x slower. Position actuators as
shipped in Menagerie. No latency modeled. Contact: elliptic cones, impratio 100,
implicit-fast integrator, cylinder with condim 4.

**Lift detector.** Evaluated at each control step: the object touches at least one hand
geom and nothing else. That instant unhides the target. A momentary bounce during a
squeeze counts as lift, so a grasp that lets the object touch down again is a violation.

**Target T*.** Axis uniform on the sphere; position with x in -2 to 5 cm, y in -3 to 3 cm,
and z between just-clearing-the-palm and 2 cm above that. Feasible if within 8 cm of the
palm origin and the object at that pose does not penetrate the palm or wrist with the
fingers open. Reorient training starts with targets within 20 degrees and 1 cm of the
lift pose and widens on success.

**Reward.** Dense terms are bounded and positive: exp(-epos/3 cm) + exp(-eang/0.4 rad)
after lift, exp(-dist/10 cm) before, plus bonuses for lift, tolerance, and the hold, and
a one-off penalty for drops and violations. The first runs used negative per-step error
penalties and the policies learned to end episodes early by pushing the object into the
arm; with the per-step term negative, termination is the best action.

**Success.** 1.5 cm and 15 degrees, then a 2 s hold with the arm targets frozen. Any
non-hand contact after lift, or 0.25 s without hand contact, ends the episode as a
failure. Hand-to-floor contact is allowed and logged.

**Compute.** CPU MuJoCo. The in-hand stage is contact-rich and runs about 700
env-steps/s on 8 laptop cores; on Delta one A100 node share (16 EPYC cores) gave about
285 env-steps/s with 16 worker processes and the 2 ms step. The cluster script
(`scripts/remote/delta_submit.sh`) takes whole 64-core nodes and time-boxes each stage
with `--minutes`; the default schedule is a 3 hour end-to-end run, and `scale` stretches
it. Every stage logs to W&B project `cylinder-reorient-rl`.

## Status

- The simulator core, the three decompositions, training, evaluation, and video are
  implemented and smoke-tested end to end. Checkpoints in this repo are two-iteration
  smoke artifacts, not trained policies.
- The plan C scripted grasp lifts about one lying cylinder in three (measured on a
  3 x 3 grid). Its main failure is the lift-then-touch-down bounce described above. This
  is the baseline plans A and B are meant to beat.
- Known gaps: no standing-disk grasp in the scripted controller; in-hand resets bias
  toward poses that rest on the upturned palm; timeouts are treated as terminal in GAE.

## References

- MuJoCo Menagerie: LEAP hand and xArm7 models.
- Chen et al., Visual Dexterity (2023); Handa et al., DeXtreme (2023): teacher-student
  and domain randomization for in-hand reorientation.
- Andrychowicz et al., Hindsight Experience Replay (2017).
- Pinto et al., Asymmetric Actor Critic (2017).
- Ross et al., DAgger (2011).
- Sutton et al., Options framework (1999).
