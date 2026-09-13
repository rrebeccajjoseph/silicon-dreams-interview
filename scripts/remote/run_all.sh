#!/bin/bash
# Full training matrix on one node, each stage in its own tmux window so it survives
# disconnects. Stages that depend on a checkpoint wait for it.
#   bash scripts/remote/run_all.sh [envs] [--wandb]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENVS=${1:-64}; shift || true
WB=${*:-}
CK=checkpoints
export OMP_NUM_THREADS=1
PY="uv run python"
S=inhand

wait_for() { while [ ! -f "$1" ]; do sleep 60; done; }
win() { tmux new-window -t $S -n "$1"; tmux send-keys -t "$S:$1" "$2 2>&1 | tee -a $CK/$1.log" C-m; }

mkdir -p $CK
tmux kill-session -t $S 2>/dev/null || true
tmux new-session -d -s $S -n main

# independent teachers
win reorient "$PY scripts/train.py reorient-teacher --envs $ENVS --iters 1500 --workers 16 --out $CK $WB"
win skills_c "$PY scripts/train.py skills-c --envs $ENVS --iters 800 --workers 16 --out $CK $WB"
win mono_b   "$PY scripts/train.py mono-b --envs $ENVS --iters 3000 --workers 16 --out $CK $WB"
win mono_b_norelabel "$PY scripts/train.py mono-b --no-relabel --envs $ENVS --iters 3000 --workers 16 --out $CK/ablation --run-name mono_b_norelabel $WB"

# dependents; each waits on the teacher's final checkpoint sentinel written below
win grasp_a "$(declare -f wait_for); wait_for $CK/reorient_teacher.done; $PY scripts/train.py grasp-a --critic $CK/reorient_teacher.pt --envs $ENVS --iters 800 --workers 16 --out $CK $WB; touch $CK/grasp_a_teacher.done"
win distill_reorient "$(declare -f wait_for); wait_for $CK/reorient_teacher.done; $PY scripts/train.py distill --teacher $CK/reorient_teacher.pt --env reorient --name reorient_student --envs $ENVS --iters 200 --workers 16 --out $CK $WB"
win distill_grasp "$(declare -f wait_for); wait_for $CK/grasp_a_teacher.done; $PY scripts/train.py distill --teacher $CK/grasp_a_teacher.pt --env grasp --name grasp_a_student --envs $ENVS --iters 200 --workers 16 --out $CK $WB"
win distill_skills "$(declare -f wait_for); wait_for $CK/skills_c_teacher.done; $PY scripts/train.py distill --teacher $CK/skills_c_teacher.pt --env prims --name skills_c_student --envs $ENVS --iters 200 --workers 16 --out $CK $WB; touch $CK/skills_c_student.done"
win estimator "$(declare -f wait_for); wait_for $CK/skills_c_student.done; $PY scripts/train.py estimator-c --policy $CK/skills_c_student.pt --envs $ENVS --iters 100 --workers 16 --out $CK $WB"

# sentinels for the two teachers
tmux send-keys -t $S:reorient "; touch $CK/reorient_teacher.done" C-m
tmux send-keys -t $S:skills_c "; touch $CK/skills_c_teacher.done" C-m
echo "launched tmux session '$S'. Attach: tmux attach -t $S   List: tmux list-windows -t $S"
