"""Run CFR / CFR+ on the simplified Buckshot Roulette and report
exploitability over time. Saves the final average policy to disk so the
PPO agent can be evaluated against it (Phase 4 of the experiment plan).

Usage:
    python -m rl.cfr_baseline --iterations 2000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Optional

from open_spiel.python.algorithms import cfr, exploitability
import pyspiel

import rl.openspiel_game  # noqa: F401  — registers the game


def run_cfr(
    iterations: int = 2000,
    log_every: int = 50,
    out_dir: str = "rl_runs/cfr",
    game_params: Optional[dict] = None,
    plus: bool = True,
) -> dict:
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if log_every < 1:
        raise ValueError(f"log_every must be >= 1, got {log_every}")
    os.makedirs(out_dir, exist_ok=True)
    game = pyspiel.load_game("simple_buckshot", game_params or {})
    solver = (
        cfr.CFRPlusSolver(game) if plus else cfr.CFRSolver(game)
    )
    print(
        f"[cfr{'plus' if plus else ''}] game=simple_buckshot params={game_params or {}} "
        f"iterations={iterations}"
    )

    metrics_path = os.path.join(out_dir, "cfr_metrics.jsonl")
    history: list = []
    t_start = time.time()
    last_nash = None

    with open(metrics_path, "w") as metrics_f:
        for i in range(1, iterations + 1):
            solver.evaluate_and_update_policy()
            if i % log_every == 0 or i == 1 or i == iterations:
                avg = solver.average_policy()
                nash = exploitability.nash_conv(game, avg)
                elapsed = time.time() - t_start
                entry = {
                    "iter": i,
                    "nash_conv": float(nash),
                    "elapsed_s": float(elapsed),
                }
                metrics_f.write(json.dumps(entry) + "\n")
                metrics_f.flush()
                history.append(entry)
                print(
                    f"iter {i:>5} nash_conv={nash:.6f}  elapsed={elapsed:6.1f}s"
                )
                last_nash = nash

    # Save the average policy as a tabular dict[info_state_str] -> dict[action]->prob
    avg = solver.average_policy()
    table: dict = {}
    for state_str, action_probs in _enumerate_avg_policy(game, avg).items():
        table[state_str] = action_probs

    policy_path = os.path.join(out_dir, "cfr_avg_policy.pkl")
    with open(policy_path, "wb") as f:
        pickle.dump(
            {
                "table": table,
                "game_params": game_params or {},
                "iterations": iterations,
                "final_nash_conv": last_nash,
            },
            f,
        )
    print(f"saved avg policy table to {policy_path}  ({len(table)} info states)")
    return {
        "metrics_path": metrics_path,
        "policy_path": policy_path,
        "final_nash_conv": last_nash,
        "n_info_states": len(table),
        "history": history,
    }


def _enumerate_avg_policy(game: pyspiel.Game, avg_policy) -> dict[str, dict[int, float]]:
    """Walk the entire game tree and snapshot the avg policy at each
    info state. Tractable only for the simplified game.
    """
    out: dict[str, dict[int, float]] = {}

    def visit(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                child = state.clone()
                child.apply_action(a)
                visit(child)
            return
        player = state.current_player()
        info = state.information_state_string(player)
        if info not in out:
            probs = avg_policy.action_probabilities(state)
            out[info] = {int(a): float(p) for a, p in probs.items()}
        for a in state.legal_actions():
            child = state.clone()
            child.apply_action(a)
            visit(child)

    visit(game.new_initial_state())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="rl_runs/cfr")
    p.add_argument("--no-plus", action="store_true", help="Use vanilla CFR instead of CFR+")
    p.add_argument("--hp", type=int, default=2)
    p.add_argument("--n-live", type=int, default=2)
    p.add_argument("--n-blank", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_cfr(
        iterations=a.iterations,
        log_every=a.log_every,
        out_dir=a.out_dir,
        plus=not a.no_plus,
        game_params={"hp": a.hp, "n_live": a.n_live, "n_blank": a.n_blank},
    )
