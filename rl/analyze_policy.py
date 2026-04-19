"""Behavioral analysis of a trained policy.

Loads a checkpoint and probes its decisions across a battery of crafted
game states designed to reveal what strategy the agent learned. Output is
a markdown-style table to stdout (also written to a file).

Crafted scenarios are built directly via BuckshotEngine state mutation —
this skips the question "what's the agent's policy in random_state_X" and
instead asks "what does the agent do when the choice is theoretically
clear?"

Usage:
    python -m rl.analyze_policy rl_runs/<run>/policy_final.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from rl.engine import NUM_ACTIONS, Action, BuckshotEngine, Item, NUM_ITEMS
from rl.policy import ActorCritic


@dataclass
class Scenario:
    name: str
    description: str
    setup: callable  # takes engine, mutates state in place
    optimal_action_hint: str  # human-readable expected behavior


def _empty_inventory() -> np.ndarray:
    return np.zeros(NUM_ITEMS, dtype=np.int32)


def _scenario_pure_blank_majority(e: BuckshotEngine) -> None:
    """1 live + 3 blanks → P(blank)=75%; naive math says shoot self."""
    e.state.shells = [False, True, False, False]
    e.state.players[e.state.current_player].inventory = _empty_inventory()
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_pure_live_majority(e: BuckshotEngine) -> None:
    """3 live + 1 blank → P(live)=75%; naive math says shoot opponent."""
    e.state.shells = [True, False, True, True]
    e.state.players[e.state.current_player].inventory = _empty_inventory()
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_pure_50_50(e: BuckshotEngine) -> None:
    """1 live + 1 blank → equal EV; agent should be ~indifferent."""
    e.state.shells = [True, False]
    e.state.players[e.state.current_player].inventory = _empty_inventory()
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_glass_available_uncertain(e: BuckshotEngine) -> None:
    """Mixed shells, agent has Glass — it should USE it for free info."""
    e.state.shells = [True, False, True, False]
    inv = _empty_inventory()
    inv[int(Item.GLASS)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_handsaw_lethal(e: BuckshotEngine) -> None:
    """Opponent at 2 HP, 3 live shells → use saw, shoot, opponent dies in one."""
    e.state.players[1 - e.state.current_player].hp = 2
    e.state.shells = [True, True, True, False]
    inv = _empty_inventory()
    inv[int(Item.HANDSAW)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_smoke_when_low_hp(e: BuckshotEngine) -> None:
    """Agent at 1 HP (max 4), has Smoke — should heal."""
    me = e.state.players[e.state.current_player]
    me.hp = 1
    me.max_hp = 4
    inv = _empty_inventory()
    inv[int(Item.SMOKE)] = 1
    me.inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()
    e.state.shells = [True, False, True]


def _scenario_handcuff_then_lethal(e: BuckshotEngine) -> None:
    """Opponent at 1 HP, agent has handcuff + good odds — cuff first to
    guarantee a free turn after the kill chance."""
    e.state.players[1 - e.state.current_player].hp = 1
    e.state.shells = [True, False, True]
    inv = _empty_inventory()
    inv[int(Item.HANDCUFF)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_inverter_known_blank(e: BuckshotEngine) -> None:
    """Glass-revealed next shell is BLANK (recorded in known_shells), agent
    has inverter. Optimal: invert (now next is live) and shoot opponent."""
    e.state.shells = [False, True, False]
    e.state.known_shells[e.state.current_player][0] = False
    inv = _empty_inventory()
    inv[int(Item.INVERTER)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


SCENARIOS: list[Scenario] = [
    Scenario(
        "blank_majority_no_items",
        "1L + 3B (75% blank), no items.",
        _scenario_pure_blank_majority,
        "Naive EV: shoot self (75% blank → keep turn).",
    ),
    Scenario(
        "live_majority_no_items",
        "3L + 1B (75% live), no items.",
        _scenario_pure_live_majority,
        "Naive EV: shoot opponent (likely-live shot lands on them).",
    ),
    Scenario(
        "50_50_no_items",
        "1L + 1B (50/50), no items.",
        _scenario_pure_50_50,
        "Indifferent / slight edge — observe what agent prefers.",
    ),
    Scenario(
        "glass_when_uncertain",
        "Mixed shells, agent has Glass.",
        _scenario_glass_available_uncertain,
        "Use Glass for free information before deciding to shoot.",
    ),
    Scenario(
        "handsaw_lethal",
        "Opp at 2 HP, 3L+1B, agent has Handsaw.",
        _scenario_handsaw_lethal,
        "Use Handsaw → shoot opponent (one-shot kill).",
    ),
    Scenario(
        "smoke_when_low_hp",
        "Agent 1/4 HP, has Smoke, mixed shells.",
        _scenario_smoke_when_low_hp,
        "Heal first to gain HP buffer.",
    ),
    Scenario(
        "handcuff_then_lethal",
        "Opp at 1 HP, mixed shells, agent has Handcuff.",
        _scenario_handcuff_then_lethal,
        "Cuff opponent for safety, then aim for the kill.",
    ),
    Scenario(
        "inverter_known_blank",
        "Glass said next is BLANK; agent has Inverter.",
        _scenario_inverter_known_blank,
        "Invert the chamber to make it live, then shoot opponent.",
    ),
]


def load_policy(checkpoint: str, obs_dim: int = 47, hidden: int = 256, device: str = "cpu") -> ActorCritic:
    p = ActorCritic(obs_dim, NUM_ACTIONS, hidden=hidden).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    p.load_state_dict(state)
    p.eval()
    return p


def _format_action(a: int) -> str:
    return Action(a).name


def probe_policy(policy: ActorCritic, scenarios: list[Scenario], device: str = "cpu") -> list[dict]:
    rows: list[dict] = []
    for sc in scenarios:
        e = BuckshotEngine()
        e.reset(seed=0)
        sc.setup(e)
        obs = e.observation(e.state.current_player).astype(np.float32)
        mask = e.legal_actions().astype(np.int8)
        if not mask.any():
            rows.append({
                "scenario": sc.name, "error": "no legal actions after setup",
                "description": sc.description, "expected": sc.optimal_action_hint,
            })
            continue
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)
            logits, value = policy.forward(obs_t)
            masked_logits = logits.masked_fill(mask_t == 0, -1e8)
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()
        # Top 3 actions by prob
        top_ids = np.argsort(-probs)
        top = [(int(i), float(probs[i])) for i in top_ids if mask[i] == 1][:3]
        rows.append({
            "scenario": sc.name,
            "description": sc.description,
            "expected": sc.optimal_action_hint,
            "value_estimate": float(value.item()),
            "top_actions": [(_format_action(i), p) for i, p in top],
            "argmax": _format_action(int(np.argmax(probs))),
        })
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = ["# Policy behavioral analysis", ""]
    for r in rows:
        lines.append(f"## {r['scenario']}")
        lines.append(f"_{r['description']}_")
        lines.append(f"**Expected:** {r['expected']}")
        if "error" in r:
            lines.append(f"**ERROR:** {r['error']}")
            lines.append("")
            continue
        lines.append(f"**Argmax:** `{r['argmax']}`  |  **Value estimate:** `{r['value_estimate']:+.3f}`")
        lines.append("")
        lines.append("| Action | Probability |")
        lines.append("|---|---|")
        for action, prob in r["top_actions"]:
            lines.append(f"| `{action}` | {prob:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Path to .pt checkpoint")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="Output markdown file (default: <ckpt_dir>/analysis.md)")
    a = p.parse_args()

    policy = load_policy(a.checkpoint, hidden=a.hidden, device=a.device)
    rows = probe_policy(policy, SCENARIOS, device=a.device)
    md = render_markdown(rows)
    out = a.out or os.path.join(os.path.dirname(a.checkpoint) or ".", "analysis.md")
    with open(out, "w") as f:
        f.write(md)
    # JSONL of raw rows for programmatic comparison
    json_out = out.replace(".md", ".jsonl")
    with open(json_out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(md)
    print(f"\n[wrote] {out}")
    print(f"[wrote] {json_out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
