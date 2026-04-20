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
from rl.policy import ActorCritic, load_policy as _load_policy


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


# --- Item-strategy deep dives ---

def _scenario_saw_when_known_live(e: BuckshotEngine) -> None:
    """Glass already revealed next shell is LIVE, opponent at 2 HP, agent
    has Handsaw. Optimal: USE_HANDSAW (then shoot opponent for 2 dmg = kill)."""
    e.state.shells = [True, False, True]
    e.state.known_shells[e.state.current_player][0] = True
    e.state.players[1 - e.state.current_player].hp = 2
    inv = _empty_inventory()
    inv[int(Item.HANDSAW)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_saw_when_known_blank(e: BuckshotEngine) -> None:
    """Agent KNOWS next shell is blank. Saw on a blank is WASTED (damage
    resets after a blank-shot anyway). Optimal: NOT USE_HANDSAW — just
    shoot self for the free turn, or shoot opponent to burn the blank."""
    e.state.shells = [False, True]
    e.state.known_shells[e.state.current_player][0] = False
    inv = _empty_inventory()
    inv[int(Item.HANDSAW)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_smoke_useless_at_full_hp(e: BuckshotEngine) -> None:
    """Agent at full HP with a Smoke. Smoke can't heal beyond max. Should
    be masked as illegal — but if agent somehow considers it, should never
    pick. Verify legal mask excludes SMOKE."""
    me = e.state.players[e.state.current_player]
    me.hp = me.max_hp
    inv = _empty_inventory()
    inv[int(Item.SMOKE)] = 1
    me.inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()
    e.state.shells = [True, False]


def _scenario_beer_when_certain_death_next_shot(e: BuckshotEngine) -> None:
    """Agent at 1 HP, opp at FULL HP (so SHOOT_OPP cannot win this turn),
    next shell known LIVE. USE_BEER ejects the lethal shell; SHOOT_OPP
    deals 1 dmg then opp kills me next turn. Unambiguously beer is
    better than shooting. Optimal: USE_BEER."""
    me = e.state.players[e.state.current_player]
    opp = e.state.players[1 - e.state.current_player]
    me.hp = 1
    opp.hp = opp.max_hp  # opp can't be killed this turn
    e.state.shells = [True, False, True]
    e.state.known_shells[e.state.current_player][0] = True
    inv = _empty_inventory()
    inv[int(Item.BEER)] = 1
    me.inventory = inv
    opp.inventory = _empty_inventory()


def _scenario_phone_when_many_shells(e: BuckshotEngine) -> None:
    """Fresh round, 5-shell chamber, no items used yet, agent has Phone.
    Phone reveals one random shell's identity. High information value.
    Optimal: USE_PHONE before committing to a shot."""
    e.state.shells = [True, False, True, False, True]
    inv = _empty_inventory()
    inv[int(Item.PHONE)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_adrenaline_steal_saw_for_kill(e: BuckshotEngine) -> None:
    """Opponent at 2 HP, opp has Handsaw, I have Adrenaline + will shoot
    opp with live next. Optimal: USE_ADRENALINE to steal their saw, then
    USE the saw, then shoot for 2 dmg = kill."""
    e.state.players[1 - e.state.current_player].hp = 2
    e.state.shells = [True, False]
    e.state.known_shells[e.state.current_player][0] = True
    me_inv = _empty_inventory()
    me_inv[int(Item.ADRENALINE)] = 1
    opp_inv = _empty_inventory()
    opp_inv[int(Item.HANDSAW)] = 1
    e.state.players[e.state.current_player].inventory = me_inv
    e.state.players[1 - e.state.current_player].inventory = opp_inv


def _scenario_adrenaline_steal_smoke_when_low(e: BuckshotEngine) -> None:
    """Agent at 1/4 HP, has Adrenaline but no Smoke; opp has Smoke.
    Optimal: USE_ADRENALINE → PICK_SMOKE to heal yourself using their item."""
    me = e.state.players[e.state.current_player]
    me.hp = 1
    me.max_hp = 4
    e.state.shells = [True, False, True]
    me_inv = _empty_inventory()
    me_inv[int(Item.ADRENALINE)] = 1
    opp_inv = _empty_inventory()
    opp_inv[int(Item.SMOKE)] = 1
    me.inventory = me_inv
    e.state.players[1 - e.state.current_player].inventory = opp_inv


def _scenario_pills_as_last_resort(e: BuckshotEngine) -> None:
    """Agent at 1 HP, opp at 1 HP, next shell certain live (agent's turn).
    Shooting opp wins. Shooting self kills. Pills is a 40% +2 HP gamble
    (lives) vs 60% -1 HP (dies — but you're already facing a near-certain
    kill anyway). Pills is NOT obviously optimal here since opp is 1 HP
    and shooting them wins. Optimal: SHOOT_OPPONENT (live next → kill)."""
    me = e.state.players[e.state.current_player]
    me.hp = 1
    e.state.players[1 - e.state.current_player].hp = 1
    e.state.shells = [True, False]
    e.state.known_shells[e.state.current_player][0] = True
    inv = _empty_inventory()
    inv[int(Item.PILLS)] = 1
    me.inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_pills_desperate(e: BuckshotEngine) -> None:
    """Agent at 1 HP, next shell known live, NO items except Pills, NO way
    to eject. Shooting opp is impossible (shells not usable?) — actually
    shooting opp still works, damages them for 1. But if opp is at HP=3
    and you can't kill them, you might need the pills gamble for HP
    buffer. Optimal here: SHOOT_OPPONENT (damages them, ends your turn)
    is still likely better than pills (60% kill you immediately)."""
    me = e.state.players[e.state.current_player]
    me.hp = 1
    e.state.players[1 - e.state.current_player].hp = 3
    e.state.shells = [True, False]
    e.state.known_shells[e.state.current_player][0] = True
    inv = _empty_inventory()
    inv[int(Item.PILLS)] = 1
    me.inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_cuff_before_saw_combo(e: BuckshotEngine) -> None:
    """Opp at 2 HP, next shell known LIVE, agent has Handcuff + Handsaw.
    Optimal chain: USE_HANDCUFF (opp skips), then USE_HANDSAW, then
    SHOOT_OPPONENT (2 dmg = kill). Cuff first so even if agent misses next
    turn (though we know it's live), opp can't retaliate."""
    e.state.players[1 - e.state.current_player].hp = 2
    e.state.shells = [True, False]
    e.state.known_shells[e.state.current_player][0] = True
    inv = _empty_inventory()
    inv[int(Item.HANDCUFF)] = 1
    inv[int(Item.HANDSAW)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_inverter_flip_known_live_to_save_life(e: BuckshotEngine) -> None:
    """Agent at 1 HP, opp at FULL HP, next shell KNOWN LIVE. SHOOT_OPP
    deals 1 dmg (opp still alive) → opp kills me. USE_INVERTER flips
    live→blank: I shoot_self for free turn, next shell is live, kill.
    Optimal: USE_INVERTER (survival > tiny damage)."""
    me = e.state.players[e.state.current_player]
    opp = e.state.players[1 - e.state.current_player]
    me.hp = 1
    opp.hp = opp.max_hp
    e.state.shells = [True, False]
    e.state.known_shells[e.state.current_player][0] = True
    inv = _empty_inventory()
    inv[int(Item.INVERTER)] = 1
    me.inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_glass_then_act_all_live(e: BuckshotEngine) -> None:
    """Fresh 3-shell chamber, all live (unseen by agent). Agent has Glass.
    Optimal: USE_GLASS first — will reveal live, then agent knows any shot
    is a live shot → SHOOT_OPPONENT is clearly correct."""
    e.state.shells = [True, True, True]
    inv = _empty_inventory()
    inv[int(Item.GLASS)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_opp_cuffed_go_aggressive(e: BuckshotEngine) -> None:
    """Opponent is already cuffed (skip_next_turn=True) → whatever I do,
    opp doesn't act. Free offensive turn. Optimal: USE_HANDSAW (if available)
    + SHOOT_OPPONENT. Known live. Opp at 2 HP."""
    e.state.players[1 - e.state.current_player].skip_next_turn = True
    e.state.players[1 - e.state.current_player].hp = 2
    e.state.shells = [True, False]
    e.state.known_shells[e.state.current_player][0] = True
    inv = _empty_inventory()
    inv[int(Item.HANDSAW)] = 1
    e.state.players[e.state.current_player].inventory = inv
    e.state.players[1 - e.state.current_player].inventory = _empty_inventory()


def _scenario_useless_phone_after_glass(e: BuckshotEngine) -> None:
    """Agent has Glass AND Phone. Agent also already knows position 0 via
    prior Glass use (recorded in known_shells). Optimal: Phone to learn
    MORE positions. Glass would be redundant on position 0."""
    e.state.shells = [True, False, True, False]
    e.state.known_shells[e.state.current_player][0] = True
    inv = _empty_inventory()
    inv[int(Item.GLASS)] = 1
    inv[int(Item.PHONE)] = 1
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
    Scenario(
        "saw_when_known_live",
        "Known LIVE next shell; opp at 2HP; agent has Saw.",
        _scenario_saw_when_known_live,
        "USE_HANDSAW → SHOOT_OPPONENT (guaranteed 2-dmg kill).",
    ),
    Scenario(
        "saw_when_known_blank",
        "Known BLANK next shell; agent has Saw.",
        _scenario_saw_when_known_blank,
        "NOT USE_HANDSAW (wasted on blank). Prefer SHOOT_SELF (free turn) or SHOOT_OPPONENT.",
    ),
    Scenario(
        "smoke_useless_at_full_hp",
        "Agent at full HP with Smoke (should be illegal).",
        _scenario_smoke_useless_at_full_hp,
        "SMOKE masked illegal; verify engine gating.",
    ),
    Scenario(
        "beer_when_certain_death_next_shot",
        "Agent 1HP, next shell known LIVE, has Beer.",
        _scenario_beer_when_certain_death_next_shot,
        "USE_BEER to eject the lethal shell and survive.",
    ),
    Scenario(
        "phone_when_many_shells",
        "Fresh 5-shell chamber, has Phone.",
        _scenario_phone_when_many_shells,
        "USE_PHONE for free information about a shell position.",
    ),
    Scenario(
        "adrenaline_steal_saw_for_kill",
        "Opp at 2HP with Saw; I have Adrenaline; next shell known live.",
        _scenario_adrenaline_steal_saw_for_kill,
        "USE_ADRENALINE → PICK_HANDSAW (→ SHOOT_OPPONENT for 2-dmg kill).",
    ),
    Scenario(
        "adrenaline_steal_smoke_when_low",
        "I'm 1/4 HP with Adrenaline; opp has Smoke.",
        _scenario_adrenaline_steal_smoke_when_low,
        "USE_ADRENALINE → PICK_SMOKE (heal using their item).",
    ),
    Scenario(
        "pills_vs_sure_kill",
        "Both at 1HP, next shell known live, my turn; have Pills.",
        _scenario_pills_as_last_resort,
        "SHOOT_OPPONENT (guaranteed kill) — pills are a 60% suicide here.",
    ),
    Scenario(
        "pills_when_desperate",
        "1HP vs 3HP, next shell known live, have only Pills.",
        _scenario_pills_desperate,
        "SHOOT_OPPONENT (still damages them) — pills 60% suicide.",
    ),
    Scenario(
        "cuff_saw_combo",
        "Opp 2HP, next shell known LIVE, have Cuff + Saw.",
        _scenario_cuff_before_saw_combo,
        "Chain: USE_HANDCUFF → USE_HANDSAW → SHOOT_OPPONENT (clean lethal).",
    ),
    Scenario(
        "inverter_save_from_known_live",
        "Agent 1HP, next shell known LIVE, have Inverter.",
        _scenario_inverter_flip_known_live_to_save_life,
        "USE_INVERTER (flip live→blank) to survive; creative defensive play.",
    ),
    Scenario(
        "glass_peek_unseen_all_live",
        "Fresh 3L chamber, have Glass.",
        _scenario_glass_then_act_all_live,
        "USE_GLASS first (free info), then act.",
    ),
    Scenario(
        "opp_cuffed_go_aggressive",
        "Opp cuffed + 2HP + known live; have Saw.",
        _scenario_opp_cuffed_go_aggressive,
        "USE_HANDSAW → SHOOT_OPPONENT (safe kill, opp can't retaliate).",
    ),
    Scenario(
        "phone_redundant_with_glass_info",
        "Known pos 0, have Glass + Phone.",
        _scenario_useless_phone_after_glass,
        "USE_PHONE for NEW info (not Glass, which would repeat pos 0).",
    ),
]


def load_policy(checkpoint: str, device: str = "cpu") -> ActorCritic:
    return _load_policy(checkpoint, n_actions=NUM_ACTIONS, device=device)


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
        # Top 3 actions by prob. Masked logits are -1e8 so illegal probs
        # are ~0, but zero out illegal entries explicitly before argmax so
        # the "best action" can never report an illegal action even on tie.
        legal_probs = np.where(mask == 1, probs, -1.0)
        top_ids = np.argsort(-probs)
        top = [(int(i), float(probs[i])) for i in top_ids if mask[i] == 1][:3]
        rows.append({
            "scenario": sc.name,
            "description": sc.description,
            "expected": sc.optimal_action_hint,
            "value_estimate": float(value.item()),
            "top_actions": [(_format_action(i), p) for i, p in top],
            "argmax": _format_action(int(np.argmax(legal_probs))),
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
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="Output markdown file (default: <ckpt_dir>/analysis.md)")
    a = p.parse_args()

    policy = load_policy(a.checkpoint, device=a.device)
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
