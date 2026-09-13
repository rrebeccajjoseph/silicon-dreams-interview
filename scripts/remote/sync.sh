#!/bin/bash
# Push the repo to a node (default nc-1) and pull results back. Usage: sync.sh [host] [push|pull]
set -euo pipefail
HOST=${1:-nc-1}; DIR=${2:-push}
REMOTE=/home/ubuntu/silicon-dreams-interview
cd "$(dirname "$0")/../.."
if [ "$DIR" = push ]; then
  rsync -az --info=progress2 --exclude .venv --exclude .git --exclude __pycache__ --exclude checkpoints \
        --exclude ckpt_smoke --exclude results --exclude videos --exclude wandb ./ "$HOST:$REMOTE/"
else
  rsync -az --info=progress2 "$HOST:$REMOTE/checkpoints/" checkpoints/remote/
  rsync -az "$HOST:$REMOTE/results/" results/remote/ 2>/dev/null || true
fi
