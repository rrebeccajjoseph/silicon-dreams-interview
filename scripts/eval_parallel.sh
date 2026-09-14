#!/bin/bash
# Run eval.py in N processes with different seeds, then merge. Same seeds give the same
# objects and targets across systems, since the grasp and world RNG do not depend on the policy.
#   bash scripts/eval_parallel.sh <name> <procs> <episodes-per-proc> <eval.py args...>
set -euo pipefail
NAME=$1; PROCS=$2; EPS=$3; shift 3
parts=()
for i in $(seq 0 $((PROCS - 1))); do
  uv run python scripts/eval.py "$@" --episodes "$EPS" --seed $((123 + i)) --name "${NAME}_part$i" > "results/${NAME}_part$i.log" 2>&1 &
  parts+=("${NAME}_part$i")
done
wait
uv run python scripts/eval.py --merge "${parts[@]}" --name "$NAME" | head -40
