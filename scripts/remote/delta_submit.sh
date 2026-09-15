#!/bin/bash
# Submit the training matrix to Delta as dependent Slurm jobs. Stages are time-boxed
# with --minutes so the chain finishes on a schedule regardless of throughput.
#   bash scripts/remote/delta_submit.sh [envs] [scale] [--wandb]
# scale multiplies every stage's minutes (1.0 = the ~3 h schedule below).
# Node shape via env vars: GPUS=4 CPUS=64 MEM=200g WORKERS=64 takes whole nodes (4x
# faster per job, but whole nodes can queue for hours); the default is one GPU's share.
set -euo pipefail
WORK=/work/nvme/beig/rjoseph2/silicon-dreams-interview
ENVS=${1:-64}; shift || true
SCALE=${1:-1.0}; shift || true
WB=${*:-}
GPUS=${GPUS:-1}; CPUS=${CPUS:-16}; MEM=${MEM:-64g}; WORKERS=${WORKERS:-16}
CK=checkpoints
cd "$WORK"
mkdir -p $CK slurm results

mins() { python3 -c "print(int($1 * $SCALE))"; }
job() {  # name, dependency ids (colon-separated or empty), minutes of walltime, command
  local name=$1 dep=$2 wall=$3 cmd=$4
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterany:$dep"
  sbatch --parsable $depflag <<EOS
#!/bin/bash
#SBATCH -A beig-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH -t 00:$(printf %02d $wall):00
#SBATCH -J $name
#SBATCH -o slurm/%x-%j.out
cd $WORK
export PATH="\$HOME/.local/bin:\$PATH" OMP_NUM_THREADS=1 UV_CACHE_DIR=/work/nvme/beig/rjoseph2/.uv-cache UV_PYTHON_INSTALL_DIR=/work/nvme/beig/rjoseph2/.uv-python
$cmd
EOS
}

PY="uv run python scripts/train.py"
COMMON="--envs $ENVS --workers $WORKERS --iters 100000 --out $CK $WB"

# stage budgets in minutes; walltime gets a few minutes of slack for setup and saving
R=$(job reorient  ""   $(mins 55) "$PY reorient-teacher --minutes $(mins 45) $COMMON")
S=$(job skills_c  ""   $(mins 45) "$PY skills-c --minutes $(mins 35) $COMMON")
B=$(job mono_b    ""   $(mins 110) "$PY mono-b --minutes $(mins 100) $COMMON")
BN=$(job mono_b_norelabel "" $(mins 110) "$PY mono-b --no-relabel --minutes $(mins 100) --envs $ENVS --workers $WORKERS --iters 100000 --out $CK/ablation --run-name mono_b_norelabel $WB")
G=$(job grasp_a   "$R" $(mins 40) "$PY grasp-a --critic $CK/reorient_teacher.pt --minutes $(mins 30) $COMMON")
DR=$(job distill_reorient "$R" $(mins 25) "$PY distill --teacher $CK/reorient_teacher.pt --env reorient --name reorient_student --minutes $(mins 15) $COMMON")
DG=$(job distill_grasp    "$G" $(mins 25) "$PY distill --teacher $CK/grasp_a_teacher.pt --env grasp --name grasp_a_student --minutes $(mins 15) $COMMON")
DS=$(job distill_skills   "$S" $(mins 25) "$PY distill --teacher $CK/skills_c_teacher.pt --env prims --name skills_c_student --minutes $(mins 15) $COMMON")
E=$(job estimator_c       "$DS" $(mins 20) "$PY estimator-c --policy $CK/skills_c_student.pt --minutes $(mins 10) $COMMON")
EVAL="uv run python eval.py"
GRID="--grid --n-r 3 --n-h 4 --per-cell 1"
EV=$(job eval_all "$DR:$DG:$DS:$E:$B" 45 "$EVAL a --grasp $CK/grasp_a_student.pt --reorient $CK/reorient_student.pt $GRID; $EVAL b --policy $CK/mono_b.pt $GRID; $EVAL c --skills $CK/skills_c_student.pt --estimator $CK/estimator_c.pt $GRID; $EVAL c --skills $CK/skills_c_student.pt --estimator $CK/estimator_c.pt --grasp $CK/grasp_a_student.pt $GRID --out results/c_hybrid; $EVAL random $GRID; uv run python scripts/record_video.py a --grasp $CK/grasp_a_student.pt --reorient $CK/reorient_student.pt --episodes 4; uv run python scripts/record_video.py b --policy $CK/mono_b.pt --episodes 4; uv run python scripts/record_video.py c --skills $CK/skills_c_student.pt --estimator $CK/estimator_c.pt --episodes 4")

echo "submitted: reorient=$R skills_c=$S mono_b=$B mono_b_norelabel=$BN grasp_a=$G distill=$DR,$DG,$DS estimator=$E eval=$EV"
squeue -u rjoseph2 -o "%.10i %.20j %.8T %.10M %.6D %R"
