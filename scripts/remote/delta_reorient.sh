#!/bin/bash
# Reorient-stage runs on Delta, one GPU share each. Needs checkpoints/handovers.npz synced first.
#   bash scripts/remote/delta_reorient.sh <partition> <minutes> <name> <extra train.py args...>
set -euo pipefail
WORK=/work/nvme/beig/rjoseph2/silicon-dreams-interview
PART=$1; MIN=$2; NAME=$3; shift 3
cd "$WORK"
mkdir -p slurm checkpoints/$NAME
sbatch --parsable <<EOS
#!/bin/bash
#SBATCH -A beig-delta-gpu
#SBATCH -p $PART
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 00:$((MIN + 8)):00
#SBATCH -J $NAME
#SBATCH -o slurm/%x-%j.out
cd $WORK
export PATH="\$HOME/.local/bin:\$PATH" OMP_NUM_THREADS=1 UV_CACHE_DIR=/work/nvme/beig/rjoseph2/.uv-cache UV_PYTHON_INSTALL_DIR=/work/nvme/beig/rjoseph2/.uv-python
uv run python scripts/train.py reorient-teacher --fixed-targets --actor deploy --bank checkpoints/handovers.npz \
    --envs 64 --workers 16 --iters 100000 --minutes $MIN --out checkpoints/$NAME $*
EOS
