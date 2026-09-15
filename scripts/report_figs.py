"""Figures for the report from eval.py result files.

  python scripts/report_figs.py --policy results/s_policy.json --hold results/s_hold.json \
      --grid results/s_grid_wide.json --out report/figs"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
POLICY, HOLD = "#2a78d6", "#eb6834"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def load(path):
    return json.load(open(path))["episodes"] if path else None


def cdf(ax, vals, color, label):
    v = np.sort(vals)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=2, label=label)


def errors(policy, hold, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, key, scale, tol, unit in ((axes[0], "epos", 100, 1.5, "cm"), (axes[1], "eang_deg", 1, 15, "deg")):
        for eps, color, name in ((policy, POLICY, "learned reorient"), (hold, HOLD, "hold still")):
            sel = [e[key] * scale for e in eps if e["system_stage"] == "reorient"]
            if sel:
                cdf(ax, sel, color, f"{name} (n={len(sel)})")
        ax.axvline(tol, color=MUTED, lw=1, ls="--")
        ax.text(tol, 1.02, f" tol {tol:g} {unit}", color=INK2, fontsize=8, va="bottom")
        ax.set_xlabel(f"final {'position' if key == 'epos' else 'axis'} error ({unit})")
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("fraction of handed-over episodes")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "errors_cdf.png", dpi=200)


def taxonomy(policy, hold, out):
    def rates(eps):
        c = Counter("success" if e["success"] else f"{e['system_stage']}: {e['reason']}" for e in eps)
        return {k: v / len(eps) for k, v in c.items()}
    rp, rh = rates(policy), rates(hold)
    keys = sorted(set(rp) | set(rh), key=lambda k: (k != "success", -max(rp.get(k, 0), rh.get(k, 0))))
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 0.35 * len(keys) + 0.9))
    h = 0.38
    for off, r, color, name in ((-h / 2 - 0.01, rp, POLICY, "learned reorient"), (h / 2 + 0.01, rh, HOLD, "hold still")):
        vals = [r.get(k, 0) * 100 for k in keys]
        ax.barh(y + off, vals, height=h, color=color, label=name)
        for yi, v in zip(y + off, vals):
            if v > 0:
                ax.text(v + 0.5, yi, f"{v:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, keys)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of episodes (%)  -  stage is where the system was when the episode ended")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "failure_taxonomy.png", dpi=200)


def envelope(grid, out):
    poses = sorted({e["start_pose"] for e in grid})
    rs = sorted({round(e["r"], 4) for e in grid})
    hs = sorted({round(e["h"], 4) for e in grid})
    fig, axes = plt.subplots(len(poses), 2, figsize=(7.2, 2.9 * len(poses)), squeeze=False)
    for i, pose in enumerate(poses):
        for j, (what, title) in enumerate((("handover", "reached handover (grasp stage)"), ("success", "end-to-end success"))):
            ax = axes[i, j]
            M = np.full((len(hs), len(rs)), np.nan)
            lab = [["" for _ in rs] for _ in hs]
            for a, h in enumerate(hs):
                for b, r in enumerate(rs):
                    sel = [e for e in grid if e["start_pose"] == pose and round(e["r"], 4) == r and round(e["h"], 4) == h]
                    if sel:
                        k = sum((e["system_stage"] == "reorient") if what == "handover" else e["success"] for e in sel)
                        M[a, b] = k / len(sel)
                        lab[a][b] = f"{k}/{len(sel)}"
            ax.imshow(M, origin="lower", cmap=BLUES, vmin=0, vmax=1, aspect="auto")
            for a in range(len(hs)):
                for b in range(len(rs)):
                    if lab[a][b]:
                        ax.text(b, a, lab[a][b], ha="center", va="center", fontsize=7,
                                color="white" if M[a, b] > 0.55 else INK)
            ax.set_xticks(range(len(rs)), [f"{2 * r * 100:.1f}" for r in rs])
            ax.set_yticks(range(len(hs)), [f"{h * 100:.0f}" for h in hs])
            ax.grid(False)
            ax.set_xlabel("diameter 2r (cm)")
            ax.set_ylabel("height h (cm)")
            ax.set_title(f"{pose}: {title}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "envelope.png", dpi=200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--hold", required=True)
    p.add_argument("--grid")
    p.add_argument("--out", default="report/figs")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy, hold = load(args.policy), load(args.hold)
    errors(policy, hold, out)
    taxonomy(policy, hold, out)
    if args.grid:
        envelope(load(args.grid), out)
    print("wrote", sorted(f.name for f in out.iterdir()))


if __name__ == "__main__":
    main()
