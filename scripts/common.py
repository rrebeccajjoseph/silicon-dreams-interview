"""Shared CLI plumbing: args, jsonl + optional W&B logging, checkpoint bookkeeping."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--threads", type=int, default=8, help="thread pool size when --workers is 0")
    p.add_argument("--workers", type=int, default=0, help="env worker processes (0 = threads in-process)")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--project", default="cylinder-reorient-rl")
    p.add_argument("--run-name", default=None)
    return p


class Logger:
    """Writes <out>/<name>.jsonl, prints one line per iter, mirrors to W&B if asked."""

    def __init__(self, out: Path, name: str, args=None, config=None):
        out.mkdir(parents=True, exist_ok=True)
        self.f = open(out / f"{name}.jsonl", "a")
        self.t0 = time.time()
        self.wb = None
        if config is not None:
            with open(out / f"{name}.config.json", "w") as f:
                json.dump({"args": vars(args) if args else {}, "config": asdict(config)}, f, indent=1, default=str)
        if args is not None and args.wandb:
            import wandb
            self.wb = wandb.init(project=args.project, name=args.run_name or f"{name}-s{args.seed}",
                                 config={**vars(args), **({"cfg": asdict(config)} if config else {})},
                                 dir=str(out), resume="allow")

    def log(self, it: int, **kw) -> None:
        row = {"it": it, "wall_s": round(time.time() - self.t0, 1),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}}
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        if self.wb is not None:
            flat = {k: v for k, v in row.items() if isinstance(v, (int, float)) and v is not None}
            for k, v in row.items():  # nested dicts such as the failure reasons
                if isinstance(v, dict):
                    flat.update({f"{k}/{kk}": vv for kk, vv in v.items() if isinstance(vv, (int, float))})
            self.wb.log(flat, step=it)

    def finish(self) -> None:
        self.f.close()
        if self.wb is not None:
            self.wb.finish()


class Checkpointer:
    """Keeps <name>.pt (latest) and <name>.best.pt (best smoothed success)."""

    def __init__(self, out: Path, name: str, save_fn, every: int):
        self.path = out / f"{name}.pt"
        self.best_path = out / f"{name}.best.pt"
        self.save_fn, self.every = save_fn, every
        self.best = -1.0
        self.ema = None

    def step(self, it: int, score: float | None, last: bool = False) -> None:
        if score is not None:
            self.ema = score if self.ema is None else 0.9 * self.ema + 0.1 * score
        if it % self.every == 0 or last:
            self.save_fn(str(self.path))
        if self.ema is not None and self.ema > self.best and it > 0:
            self.best = self.ema
            self.save_fn(str(self.path))
            shutil.copy(self.path, self.best_path)
