"""Generate a final markdown report aggregating results across experiment phases.

Pulls from:
  - rl_runs/phase1_baseline_5M/{metrics.jsonl,post_run_summary.json,analysis.md}
  - rl_runs/sweep_phase2/{ranking.json,manifest.json}
  - rl_runs/phase3_final_50M/{metrics.jsonl,post_run_summary.json,analysis.md,round_robin.json}
  - rl_runs/cfr/{cfr_metrics.jsonl} (simplified-game ground truth)

Missing sections are skipped so the script works at any phase boundary.

Usage:
    python -m rl.final_report --out reports/final_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _section_phase_summary(label: str, run_dir: str) -> list[str]:
    summary_path = os.path.join(run_dir, "post_run_summary.json")
    summary = _load_json(summary_path)
    if summary is None:
        return [f"_{label}: `post_run_summary.json` not found at {run_dir}._", ""]
    lines = [f"### {label}", f"**Run dir:** `{run_dir}`"]
    lines.append(f"**Training steps:** {summary.get('training_final_global_step', '?')}")
    lines.append(f"**Final return50:** {summary.get('training_final_return50', '?')}")
    lines.append(f"**Eval episodes per opponent:** {summary.get('eval_episodes_per_opp', '?')}")
    lines.append("")
    lines.append("| Opponent | Win rate | 95% CI |")
    lines.append("|---|---|---|")
    wr = summary.get("winrates", {})
    ci = summary.get("ci95_halfwidth", {})
    for opp, rate in wr.items():
        lines.append(f"| {opp} | {rate:.3f} | ±{ci.get(opp, 0):.3f} |")
    lines.append(f"| **mean** | **{summary.get('mean_winrate', 0):.3f}** | |")
    lines.append("")
    return lines


def _section_sweep(sweep_dir: str) -> list[str]:
    ranking_path = os.path.join(sweep_dir, "ranking.json")
    ranking = _load_json(ranking_path)
    if not ranking:
        return [f"_Sweep ranking not found at {ranking_path}._", ""]
    lines = ["### Phase 2: hyperparameter sweep results", ""]
    lines.append(f"**Sweep dir:** `{sweep_dir}`")
    lines.append(f"**Configs:** {len(ranking)}")
    lines.append("")
    lines.append("| Rank | Score | Config | vs_random | vs_aggr | vs_cons | Elapsed |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranking[:10]):  # top 10
        cfg = r.get("cfg", {})
        m = r.get("metrics", {})
        lines.append(
            f"| {i+1} | {r.get('score', 0):.3f} | "
            f"`lr={cfg.get('lr')}` `ent={cfg.get('ent_coef')}` `h={cfg.get('hidden')}` | "
            f"{m.get('eval/winrate_vs_random', 0):.3f} | "
            f"{m.get('eval/winrate_vs_aggressive', 0):.3f} | "
            f"{m.get('eval/winrate_vs_conservative', 0):.3f} | "
            f"{r.get('elapsed_s', 0):.0f}s |"
        )
    lines.append("")
    return lines


def _section_cfr(cfr_dir: str) -> list[str]:
    path = os.path.join(cfr_dir, "cfr_metrics.jsonl")
    rows = _load_jsonl(path)
    if not rows:
        return [f"_CFR metrics not found at {path}._", ""]
    lines = ["### Phase 4 baseline: CFR on simplified game",
             f"**Directory:** `{cfr_dir}`"]
    final = rows[-1]
    lines.append(f"**Final iteration:** {final['iter']}")
    lines.append(f"**Final nash_conv:** {final['nash_conv']:.6f} "
                 f"(≤0.01 = near-Nash; ≤0.001 = effectively Nash)")
    lines.append(f"**Convergence elapsed:** {final['elapsed_s']:.1f}s")
    lines.append("")
    return lines


def _section_round_robin(rr_path: str) -> list[str]:
    data = _load_json(rr_path)
    if not data:
        return [f"_Round-robin results not found at {rr_path}._", ""]
    lines = ["### Phase 4: round-robin across training trajectory",
             f"**Input:** `{rr_path}`",
             f"**Episodes per match:** {data.get('episodes_per_match', '?')}", ""]
    lines.append("Mean win-rate ranking (descending):")
    for i, name in enumerate(data.get("ranking", [])):
        wr = data.get("mean_winrate", {}).get(name, 0)
        lines.append(f"  {i+1}. `{name}` — {wr:.3f}")
    cycles = data.get("cycles", [])
    if cycles:
        lines.append("")
        lines.append(f"**⚠ Cycle warning:** {len(cycles)} rock-paper-scissors triples detected.")
    else:
        lines.append("")
        lines.append("**✓** No cycles detected — training produced a monotone dominance ordering.")
    lines.append("")
    return lines


def _maybe_append_analysis(run_dir: str, heading: str) -> list[str]:
    """Copy-paste the per-phase behavioral analysis markdown if present."""
    p = os.path.join(run_dir, "analysis.md")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        content = f.read()
    # strip the top-level "# Policy behavioral analysis" header — we nest it
    lines = [heading, ""]
    for line in content.splitlines():
        if line.startswith("# "):
            continue
        lines.append(line)
    lines.append("")
    return lines


def build_report(args) -> str:
    lines = [
        "# Buckshot Roulette RL — final report",
        "",
        f"_Generated from {os.getcwd()}_",
        "",
        "## Experiment plan",
        "",
        "See commit history and README in [rl/](../rl/) for the full plan.",
        "In short: engine+env → PPO + self-play league → hyperparameter sweep →",
        "long final run → validation via CFR baseline + round-robin across",
        "training-trajectory checkpoints.",
        "",
    ]

    lines.append("## Phase 1: 5M baseline")
    lines.extend(_section_phase_summary("Phase 1 (5M default config)", args.phase1))
    lines.extend(_maybe_append_analysis(args.phase1, "**Phase 1 behavioral analysis:**"))

    lines.append("## Phase 2: hyperparameter sweep")
    lines.extend(_section_sweep(args.sweep))

    lines.append("## Phase 3: 50M final run")
    lines.extend(_section_phase_summary("Phase 3 (best-config 50M)", args.phase3))
    lines.extend(_maybe_append_analysis(args.phase3, "**Phase 3 behavioral analysis:**"))

    lines.append("## Phase 4: validation")
    lines.extend(_section_cfr(args.cfr))
    lines.extend(_section_round_robin(os.path.join(args.phase3, "round_robin.json")))

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1", default="rl_runs/phase1_baseline_5M")
    p.add_argument("--sweep", default="rl_runs/sweep_phase2")
    p.add_argument("--phase3", default="rl_runs/phase3_final_50M")
    p.add_argument("--cfr", default="rl_runs/cfr")
    p.add_argument("--out", default="reports/final_report.md")
    a = p.parse_args()

    md = build_report(a)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print(f"wrote {a.out}")
    print("---")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
