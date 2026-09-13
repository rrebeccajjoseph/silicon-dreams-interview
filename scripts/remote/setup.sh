#!/bin/bash
# One-time setup on a bare Ubuntu GPU node. Installs uv, the project, and a ROCm or CUDA torch.
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
if command -v rocm-smi >/dev/null 2>&1; then
  ROCM=$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1 | sed 's#.*/rocm-##' | cut -d. -f1,2)
  echo "ROCm ${ROCM:-unknown} detected, installing the matching torch wheel"
  uv pip install --reinstall torch --index-url "https://download.pytorch.org/whl/rocm${ROCM:-6.3}"
fi
uv run python -c "import torch, mujoco; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), 'mujoco', mujoco.__version__)"
uv run pytest -q
