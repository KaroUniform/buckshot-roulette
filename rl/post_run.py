"""Post-training analysis bundle.

After a PPO training run finishes, run this on its directory to produce
the standard analysis pack:
  - PNG plots from the metrics file
  - Behavioral analysis on the final checkpoint
  - Extended evaluation (more episodes per baseline) for tighter CIs
  - A short text summary printed to stdout

Usage:
    python -m rl.post_run rl_runs/<run_name>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from rl.analyze_policy import (
    SCENARIOS,
    load_policy,
    probe_policy,
    render_markdown as render_analysis_md,
)
from rl.engine import NUM_ACTIONS
from rl.eval import evaluate_policy
from rl.opponents import NAMED_OPPONENTS
from rl.plot import make_plots


def _read_last_metric(run_dir: str, key: str):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if key in entry and entry[key] is not None:
                last = entry[key]
    return last


def _infer_obs_dim(checkpoint: str) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # body.0 is the first Linear; weight shape = [hidden, obs_dim]
    return state["body.0.weight"].shape[1]


def _infer_hidden(checkpoint: str) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return state["body.0.weight"].shape[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="Directory containing policy_final.pt and metrics.jsonl")
    p.add_argument("--eval-episodes", type=int, default=1000,
                   help="Episodes per baseline for extended eval (default 1000 → ±3% CI)")
    p.add_argument("--seed", type=int, default=20260420)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (default: <run_dir>/policy_final.pt)")
    a = p.parse_args()

    ckpt = a.checkpoint or os.path.join(a.run_dir, "policy_final.pt")
    if not os.path.exists(ckpt):
        print(f"checkpoint not found: {ckpt}")
        return 1

    print(f"=== post-run analysis of {a.run_dir} ===\n")

    # 1) Plots
    print("[1/3] generating plots ...")
    out = make_plots([a.run_dir])
    print(f"  wrote {out}")

    # 2) Behavioral analysis on final policy
    print("\n[2/3] behavioral probe on final policy ...")
    obs_dim = _infer_obs_dim(ckpt)
    hidden = _infer_hidden(ckpt)
    policy = load_policy(ckpt, obs_dim=obs_dim, hidden=hidden, device=a.device)
    rows = probe_policy(policy, SCENARIOS, device=a.device)
    md = render_analysis_md(rows)
    md_path = os.path.join(a.run_dir, "analysis.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  wrote {md_path}")

    # 3) Extended eval with tight CI
    print(f"\n[3/3] extended eval ({a.eval_episodes} eps per baseline) ...")
    wr = evaluate_policy(
        policy, opponents=NAMED_OPPONENTS,
        n_episodes=a.eval_episodes, seed=a.seed, device=a.device,
    )
    p_ = a.eval_episodes
    summary = {
        "run_dir": a.run_dir,
        "eval_episodes_per_opp": p_,
        "winrates": {k: float(v) for k, v in wr.items()},
        "mean_winrate": float(np.mean(list(wr.values()))),
        # 95% CI half-width for a Bernoulli at p with n trials
        "ci95_halfwidth": {
            k: float(1.96 * np.sqrt(max(v * (1 - v), 1e-9) / p_)) for k, v in wr.items()
        },
        "training_final_return50": _read_last_metric(a.run_dir, "rollout/mean_return_50"),
        "training_final_global_step": _read_last_metric(a.run_dir, "global_step"),
    }
    with open(os.path.join(a.run_dir, "post_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"FINAL EVAL ({p_} episodes per opponent)")
    print("=" * 60)
    for opp, rate in wr.items():
        ci = summary["ci95_halfwidth"][opp]
        print(f"  vs {opp:<14}  {rate:6.3f}  ±{ci:5.3f}  (95% CI)")
    print(f"  {'mean':<17}  {summary['mean_winrate']:6.3f}")
    print(f"\nFinal return50 during training: {summary['training_final_return50']}")
    print(f"Total training steps: {summary['training_final_global_step']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
