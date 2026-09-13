#!/bin/bash
# One-time setup on NCSA Delta (login node is fine: it is only installs).
set -euo pipefail
WORK=/work/nvme/beig/rjoseph2/silicon-dreams-interview
cd "$WORK"
export UV_CACHE_DIR=/work/nvme/beig/rjoseph2/.uv-cache UV_PYTHON_INSTALL_DIR=/work/nvme/beig/rjoseph2/.uv-python
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync
uv run python -c "import torch, mujoco; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'mujoco', mujoco.__version__)"
uv run pytest -q
