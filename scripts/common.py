"""Shared CLI plumbing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--threads", type=int, default=8)
    return p


class Logger:
    def __init__(self, out: Path, name: str):
        out.mkdir(parents=True, exist_ok=True)
        self.f = open(out / f"{name}.jsonl", "a")
        self.t0 = time.time()

    def log(self, it: int, **kw) -> None:
        row = {"it": it, "wall_s": round(time.time() - self.t0, 1), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}}
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
