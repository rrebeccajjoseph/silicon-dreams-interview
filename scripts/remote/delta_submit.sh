#!/bin/bash
# Submit the whole training matrix to Delta as dependent Slurm jobs.
#   bash scripts/remote/delta_submit.sh [envs] [--wandb]
# MuJoCo is CPU-bound, so every job asks for one GPU plus the 16 cores that come with it.
set -euo pipefail
WORK=/work/nvme/beig/rjoseph2/silicon-dreams-interview
ENVS=${1:-64}; shift || true
WB=${*:-}
CK=checkpoints
cd "$WORK"
mkdir -p $CK slurm

job() {  # name, dependency job ids (comma or empty), command
  local name=$1 dep=$2 cmd=$3
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
  sbatch --parsable $depflag <<EOS
#!/bin/bash
#SBATCH -A beig-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 12:00:00
#SBATCH -J $name
#SBATCH -o slurm/%x-%j.out
cd $WORK
export PATH="\$HOME/.local/bin:\$PATH" OMP_NUM_THREADS=4 UV_CACHE_DIR=/work/nvme/beig/rjoseph2/.uv-cache UV_PYTHON_INSTALL_DIR=/work/nvme/beig/rjoseph2/.uv-python
$cmd
EOS
}

PY="uv run python scripts/train.py"
COMMON="--envs $ENVS --threads 16 --out $CK $WB"

R=$(job reorient  "" "$PY reorient-teacher --iters 1500 $COMMON")
S=$(job skills_c  "" "$PY skills-c --iters 800 $COMMON")
B=$(job mono_b    "" "$PY mono-b --iters 3000 $COMMON")
BN=$(job mono_b_norelabel "" "$PY mono-b --no-relabel --iters 3000 --envs $ENVS --threads 16 --out $CK/ablation --run-name mono_b_norelabel $WB")
G=$(job grasp_a   "$R" "$PY grasp-a --critic $CK/reorient_teacher.pt --iters 800 $COMMON")
DR=$(job distill_reorient "$R" "$PY distill --teacher $CK/reorient_teacher.pt --env reorient --name reorient_student --iters 200 $COMMON")
DG=$(job distill_grasp    "$G" "$PY distill --teacher $CK/grasp_a_teacher.pt --env grasp --name grasp_a_student --iters 200 $COMMON")
DS=$(job distill_skills   "$S" "$PY distill --teacher $CK/skills_c_teacher.pt --env prims --name skills_c_student --iters 200 $COMMON")
E=$(job estimator_c       "$DS" "$PY estimator-c --policy $CK/skills_c_student.pt --iters 100 $COMMON")
EV=$(job eval_all "$DR:$DG:$DS:$E:$B" "uv run python scripts/eval.py a --grasp $CK/grasp_a_student.pt --reorient $CK/reorient_student.pt --grid; uv run python scripts/eval.py b --policy $CK/mono_b.pt --grid; uv run python scripts/eval.py c --skills $CK/skills_c_student.pt --estimator $CK/estimator_c.pt --grid; uv run python scripts/eval.py c --skills $CK/skills_c_student.pt --estimator $CK/estimator_c.pt --grasp $CK/grasp_a_student.pt --grid --out results/c_hybrid; uv run python scripts/eval.py random --grid")

echo "submitted: reorient=$R skills_c=$S mono_b=$B mono_b_norelabel=$BN grasp_a=$G distill=$DR,$DG,$DS estimator=$E eval=$EV"
squeue -u rjoseph2 -o "%.10i %.20j %.8T %.10M %.6D %R"
