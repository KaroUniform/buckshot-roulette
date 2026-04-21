"""Round-robin tournament between checkpoints from a training run.

For Phase 4 validation: does the final policy dominate all earlier ones?
If yes, self-play is genuinely improving; if there are cycles (A beats B
beats C beats A), the league has rock-paper-scissors dynamics and the
"final" snapshot is not strictly the strongest.

Usage:
    python -m rl.round_robin rl_runs/<run>/checkpoints --episodes 500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from rl.opponents import (
    NAMED_OPPONENTS,
    OpponentFn,
    OpponentPool,
    make_frozen_policy_opponent,
)
from rl.policy import ActorCritic, load_policy
from rl.engine import NUM_ACTIONS
from rl.single_agent_env import SingleAgentBuckshotEnv


def _load_checkpoint(path: str, device: str = "cpu") -> ActorCritic:
    return load_policy(path, n_actions=NUM_ACTIONS, device=device)


def _ckpt_step(path: str) -> int:
    m = re.search(r"u(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _to_opponent_fn(competitor, device: str) -> OpponentFn:
    """Coerce a round-robin competitor to an OpponentFn.

    `competitor` is either:
      * a torch ActorCritic policy → wrapped via make_frozen_policy_opponent
      * an OpponentFn (callable taking obs, mask, rng) → returned as-is
    """
    if hasattr(competitor, "act") and hasattr(competitor, "forward"):
        return make_frozen_policy_opponent(competitor, device=device)
    return competitor


def _agent_action(competitor, obs, mask, rng, device: str) -> int:
    """Run competitor as the agent: torch policies use .act(), OpponentFn
    callables use the (obs, mask, rng) signature directly."""
    if hasattr(competitor, "act") and hasattr(competitor, "forward"):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)
            return int(competitor.act(obs_t, mask_t).item())
    return int(competitor(obs, mask, rng))


def play_match(a, b, n_episodes: int, seed: int, device: str) -> float:
    """Return P(a wins) across n_episodes games where `a` is the agent and
    `b` is the opponent in SingleAgentBuckshotEnv. Either may be a torch
    policy (ActorCritic) or an OpponentFn (rule-based baseline)."""
    opp_fn = _to_opponent_fn(b, device=device)
    pool = OpponentPool({"opp": opp_fn})
    env = SingleAgentBuckshotEnv(opponent_pool=pool)
    rng = np.random.default_rng(seed)
    wins = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
        if info.get("_terminated_in_reset"):
            # opponent won in reset
            tr = info.get("_terminal_reward", 0)
            if tr > 0:
                wins += 1
            continue
        done = False
        while not done:
            action = _agent_action(a, obs["observation"], obs["action_mask"], rng, device)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if done and reward > 0:
                wins += 1
    return wins / n_episodes


def round_robin(ckpts_dir: str, episodes: int, seed: int, device: str,
                max_ckpts: int = 8,
                explicit_paths: Optional[list[str]] = None,
                explicit_names: Optional[list[str]] = None,
                include_baselines: Optional[list[str]] = None) -> dict:
    if explicit_paths:
        paths = list(explicit_paths)
        names = list(explicit_names or [os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in paths])
    else:
        paths = sorted(glob.glob(os.path.join(ckpts_dir, "snapshot_*.pt")), key=_ckpt_step)
        if not paths:
            raise FileNotFoundError(f"No snapshots found under {ckpts_dir}")
        # If too many, sub-sample evenly so runtime is bounded
        if len(paths) > max_ckpts:
            idxs = np.linspace(0, len(paths) - 1, max_ckpts).astype(int)
            paths = [paths[i] for i in idxs]
        # Also add policy_final.pt if present
        parent = os.path.dirname(ckpts_dir)
        final_path = os.path.join(parent, "policy_final.pt")
        if os.path.exists(final_path):
            paths.append(final_path)
        names = [os.path.basename(p).replace(".pt", "") for p in paths]
    n_neural = len(paths)
    baseline_names = list(include_baselines or [])
    # Validate baselines up front
    for bname in baseline_names:
        if bname not in NAMED_OPPONENTS:
            raise ValueError(
                f"Unknown baseline '{bname}'. Available: {list(NAMED_OPPONENTS)}"
            )
    print(
        f"[round_robin] {n_neural} checkpoints + {len(baseline_names)} baselines "
        f"× {n_neural + len(baseline_names) - 1} opponents × {episodes} eps"
    )
    policies = [_load_checkpoint(p, device=device) for p in paths]
    # Append baselines as additional competitors (they coerce to OpponentFn).
    competitor_names = list(names) + list(baseline_names)
    competitors = list(policies) + [NAMED_OPPONENTS[b] for b in baseline_names]
    # Keep `names` and `policies` aliases for the rest of the function so
    # existing print/cycle code reads cleanly.
    names = competitor_names
    policies = competitors

    # Pairwise win rates: wins[i][j] = P(i beats j)
    wins: dict[tuple[int, int], float] = {}
    for i in range(len(policies)):
        for j in range(len(policies)):
            if i == j:
                continue
            wr = play_match(policies[i], policies[j], episodes, seed + i * 31 + j, device)
            wins[(i, j)] = wr
            print(f"  {names[i]:<20} vs {names[j]:<20}  {wr:.3f}")

    # Aggregate: mean row score (how often i beats someone else)
    scores = []
    for i in range(len(policies)):
        others = [wins[(i, j)] for j in range(len(policies)) if j != i]
        scores.append(float(np.mean(others)))

    ranking = sorted(range(len(policies)), key=lambda i: -scores[i])

    out = {
        "checkpoints": names,
        "pairwise_wr": {f"{names[i]}_vs_{names[j]}": v for (i, j), v in wins.items()},
        "mean_winrate": {names[i]: scores[i] for i in range(len(policies))},
        "ranking": [names[i] for i in ranking],
        "episodes_per_match": episodes,
    }

    print("\n=== ranking (by mean win rate across round-robin) ===")
    for rank, i in enumerate(ranking):
        print(f"{rank+1:>2}. {names[i]:<25} mean_wr={scores[i]:.3f}")

    # Cycle detection over UNORDERED triples so each rock-paper-scissors
    # pattern is counted once. For {a,b,c}, only one rotation at most
    # satisfies a→b→c→a given pairwise win rates, so we fix i<j<k and try
    # both orientations (i→j→k→i or i→k→j→i).
    cycles: list[tuple[str, str, str]] = []
    n = len(policies)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (wins[(i, j)] > 0.55 and wins[(j, k)] > 0.55 and wins[(k, i)] > 0.55):
                    cycles.append((names[i], names[j], names[k]))
                elif (wins[(i, k)] > 0.55 and wins[(k, j)] > 0.55 and wins[(j, i)] > 0.55):
                    cycles.append((names[i], names[k], names[j]))
    if cycles:
        print(f"\n[!] Detected {len(cycles)} rock-paper-scissors triples (threshold 0.55). First 3:")
        for c in cycles[:3]:
            print(f"  {c[0]} > {c[1]} > {c[2]} > {c[0]}")
    else:
        print("\nNo dominance cycles above the 0.55 threshold — training is monotone.")
    out["cycles"] = cycles
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpts_dir", nargs="?", default=None,
                   help="Directory containing snapshot_*.pt files (alternative: --sweep-dir)")
    p.add_argument("--sweep-dir", default=None,
                   help="Sweep directory; auto-collects sweep_*/policy_final.pt from all subdirs")
    p.add_argument("--league-dir", default=None,
                   help="League sweep directory; auto-collects *_gen*/policy_final.pt (league_train.py format)")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-ckpts", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--include-baselines",
        nargs="*",
        default=None,
        help="Names of rule-based opponents from rl.opponents.NAMED_OPPONENTS "
             "to include as round-robin competitors (e.g. strong_baseline "
             "aggressive). Useful for grounding neural results against an "
             "absolute reference. Without this flag the round-robin is "
             "neural-only.",
    )
    a = p.parse_args()
    baselines = a.include_baselines or []

    if a.league_dir:
        league_paths = sorted(glob.glob(os.path.join(a.league_dir, "*_gen*/policy_final.pt")))
        if not league_paths:
            print(f"ERROR: no *_gen*/policy_final.pt under {a.league_dir}")
            return 1
        league_names = [os.path.basename(os.path.dirname(p)) for p in league_paths]
        result = round_robin(
            a.league_dir, a.episodes, a.seed, a.device,
            max_ckpts=a.max_ckpts,
            explicit_paths=league_paths,
            explicit_names=league_names,
            include_baselines=baselines,
        )
        out = a.out or os.path.join(a.league_dir, "round_robin.json")
    elif a.sweep_dir:
        sweep_paths = sorted(glob.glob(os.path.join(a.sweep_dir, "sweep_*/policy_final.pt")))
        if not sweep_paths:
            print(f"ERROR: no sweep_*/policy_final.pt under {a.sweep_dir}")
            return 1
        sweep_names = [os.path.basename(os.path.dirname(p)).replace("sweep_", "") for p in sweep_paths]
        result = round_robin(
            a.sweep_dir, a.episodes, a.seed, a.device,
            max_ckpts=a.max_ckpts,
            explicit_paths=sweep_paths,
            explicit_names=sweep_names,
            include_baselines=baselines,
        )
        out = a.out or os.path.join(a.sweep_dir, "round_robin.json")
    else:
        if not a.ckpts_dir:
            print("ERROR: must pass ckpts_dir, --sweep-dir, or --league-dir")
            return 1
        result = round_robin(
            a.ckpts_dir, a.episodes, a.seed, a.device,
            max_ckpts=a.max_ckpts,
            include_baselines=baselines,
        )
        out = a.out or os.path.join(os.path.dirname(a.ckpts_dir) or ".", "round_robin.json")

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
