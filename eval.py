"""Single entry point for evaluation. Same flags as scripts/eval.py."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "eval.py"), run_name="__main__")
