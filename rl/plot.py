"""Generate diagnostic plots from PPO metrics.jsonl files.

Usage:
    python -m rl.plot rl_runs/<run_name>           # single run, writes PNGs into the run dir
    python -m rl.plot rl_runs/run1 rl_runs/run2    # overlay comparison

Produces:
    summary.png — 4-panel grid: returns, eval winrates, losses, PPO diagnostics
    eval.png    — eval winrates only (publication-friendly)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> dict[str, list]:
    """Load metrics.jsonl into a dict-of-lists keyed by metric name.

    Pads with None for keys that don't appear in every row (e.g. eval keys
    only present on eval steps), so all column lengths match the row count.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    all_keys = set()
    for r in rows:
        all_keys.update(r)
    cols: dict[str, list] = {k: [r.get(k) for r in rows] for k in all_keys}
    return cols


def _label_from_path(path: str) -> str:
    base = path.rstrip("/")
    if os.path.isdir(base):
        return os.path.basename(base)
    return os.path.basename(os.path.dirname(base))


def _xs(cols: dict, prefer: str = "global_step") -> tuple[np.ndarray, str]:
    if prefer in cols:
        return np.asarray(cols[prefer]), prefer
    return np.asarray(cols["update"]), "update"


def _eval_keys(cols: dict) -> list[str]:
    keys = [k for k in cols if k.startswith("eval/winrate_vs_")]
    return sorted(keys)


def _plot_returns(ax, runs: list[tuple[str, dict]]) -> None:
    for label, cols in runs:
        if "rollout/mean_return_50" not in cols:
            continue
        x_all, _ = _xs(cols)
        y_all = cols["rollout/mean_return_50"]
        mask = np.array([v is not None for v in y_all])
        x = x_all[mask]
        y = np.asarray([v for v in y_all if v is not None], dtype=float)
        ax.plot(x, y, label=label, linewidth=1.6)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("rollout: mean return (last 50 episodes)")
    ax.set_xlabel("step")
    ax.set_ylabel("return  (zero-sum, +1 win / -1 loss)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    if len(runs) > 1:
        ax.legend(loc="lower right", fontsize=8)


def _plot_eval(ax, runs: list[tuple[str, dict]]) -> None:
    palette = {"random": "#1f77b4", "aggressive": "#d62728", "conservative": "#2ca02c"}
    line_styles = ["-", "--", ":", "-."]
    for run_idx, (label, cols) in enumerate(runs):
        keys = _eval_keys(cols)
        if not keys:
            continue
        # Eval rows are sparse — use only entries that have these fields populated
        eval_mask = np.array([v is not None for v in cols[keys[0]]])
        x_all, _ = _xs(cols)
        x = x_all[eval_mask]
        for k in keys:
            opp = k.replace("eval/winrate_vs_", "")
            y = np.asarray([v for v in cols[k] if v is not None])
            color = palette.get(opp, None)
            style = line_styles[run_idx % len(line_styles)] if len(runs) > 1 else "-"
            ax.plot(
                x,
                y,
                style,
                color=color,
                label=f"vs {opp}" + (f" [{label}]" if len(runs) > 1 else ""),
                marker="o",
                markersize=3,
                linewidth=1.5,
            )
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("eval: win-rate vs baseline opponents")
    ax.set_xlabel("step")
    ax.set_ylabel("win rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def _plot_losses(ax, runs: list[tuple[str, dict]]) -> None:
    for label, cols in runs:
        x_all, _ = _xs(cols)
        for key, color in [
            ("loss/policy", "tab:red"),
            ("loss/value", "tab:blue"),
            ("loss/entropy", "tab:green"),
        ]:
            if key not in cols:
                continue
            mask = np.array([v is not None for v in cols[key]])
            x = x_all[mask]
            y = np.asarray([v for v in cols[key] if v is not None], dtype=float)
            run_label = key.split("/", 1)[1] + (f" [{label}]" if len(runs) > 1 else "")
            ax.plot(x, y, label=run_label, color=color, linewidth=1.2, alpha=0.85)
    ax.set_title("losses (policy / value / entropy)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def _plot_ppo_diagnostics(ax, runs: list[tuple[str, dict]]) -> None:
    for label, cols in runs:
        x_all, _ = _xs(cols)
        suffix = f" [{label}]" if len(runs) > 1 else ""
        for key, color in [("approx_kl", "tab:purple"), ("clipfrac", "tab:orange")]:
            if key not in cols:
                continue
            mask = np.array([v is not None for v in cols[key]])
            x = x_all[mask]
            y = np.asarray([v for v in cols[key] if v is not None], dtype=float)
            ax.plot(x, y, label=f"{key}{suffix}", color=color, linewidth=1.2)
    ax.set_title("PPO diagnostics")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def make_plots(paths: list[str], out_dir: Optional[str] = None) -> dict[str, str]:
    runs = [(_label_from_path(p), _load(p)) for p in paths]
    out_dir = out_dir or paths[0]
    if not os.path.isdir(out_dir):
        out_dir = os.path.dirname(out_dir) or "."
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _plot_returns(axes[0, 0], runs)
    _plot_eval(axes[0, 1], runs)
    _plot_losses(axes[1, 0], runs)
    _plot_ppo_diagnostics(axes[1, 1], runs)
    fig.suptitle(
        "PPO training summary  —  " + ", ".join(label for label, _ in runs),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    summary_path = os.path.join(out_dir, "summary.png")
    fig.savefig(summary_path, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_eval(ax, runs)
    fig.tight_layout()
    eval_path = os.path.join(out_dir, "eval.png")
    fig.savefig(eval_path, dpi=130)
    plt.close(fig)

    return {"summary": summary_path, "eval": eval_path}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="One or more run directories or metrics.jsonl files")
    p.add_argument("--out", default=None, help="Output directory (default: first run's dir)")
    args = p.parse_args()
    out = make_plots(args.runs, args.out)
    for name, path in out.items():
        print(f"wrote {name} -> {path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
