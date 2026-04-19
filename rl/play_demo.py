"""Interactive CLI demo: play Buckshot Roulette against a trained bot.

Usage:
    python -m rl.play_demo                                       # default: league champion
    python -m rl.play_demo path/to/policy.pt                     # custom checkpoint
    python -m rl.play_demo --seat 0                              # force human plays first
    python -m rl.play_demo --scenario beer_when_certain_death    # start from crafted probe scenario

The bot's top-3 action probabilities are printed each turn so you can
see what it was "thinking" — useful for verifying whether probe-measured
failures reproduce in real play.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from rl.engine import Action, BuckshotEngine, Item, NUM_ACTIONS, NUM_ITEMS
from rl.policy import ActorCritic


DEFAULT_CKPT = "rl_runs/league_v1/A_ent005_gen2/policy_final.pt"


def load_policy(path: str, device: str = "cpu") -> ActorCritic:
    state = torch.load(path, map_location=device, weights_only=True)
    obs_dim = state["body.0.weight"].shape[1]
    hidden = state["body.0.weight"].shape[0]
    p = ActorCritic(obs_dim, NUM_ACTIONS, hidden=hidden).to(device)
    p.load_state_dict(state)
    p.eval()
    return p


def inv_str(inv: np.ndarray) -> str:
    parts = []
    for i in Item:
        n = int(inv[int(i)])
        if n > 0:
            parts.append(f"{i.name.lower()}×{n}" if n > 1 else i.name.lower())
    return ", ".join(parts) if parts else "(empty)"


def render_state(e: BuckshotEngine, human_pid: int) -> None:
    s = e.state
    me = s.players[human_pid]
    opp = s.players[1 - human_pid]
    n = len(s.shells)
    n_live = sum(1 for x in s.shells if x)
    n_blank = n - n_live

    known = s.known_shells[human_pid]
    # Render chamber: ? for unknown positions, L/B for revealed, in slot order
    if n > 0:
        slots = []
        for pos in range(n):
            if pos in known:
                slots.append("L" if known[pos] else "B")
            else:
                slots.append("?")
        chamber = " ".join(slots)
    else:
        chamber = "(reload pending)"

    turn = "YOU" if s.current_player == human_pid else "BOT"
    adrenaline = " [ADRENALINE: pick opp's item]" if s.adrenaline_active else ""
    dmg = f" damage_mult={s.damage_mult}" if s.damage_mult > 1 else ""
    cuffs = []
    if me.skip_next_turn:
        cuffs.append("YOU are cuffed (skip next turn)")
    if opp.skip_next_turn:
        cuffs.append("BOT is cuffed (skip next turn)")
    cuff_line = "  ".join(cuffs)

    print()
    print("─" * 64)
    print(f"Turn: {turn}{adrenaline}{dmg}")
    print(f"YOU: HP {me.hp}/{me.max_hp}   items: {inv_str(me.inventory)}")
    print(f"BOT: HP {opp.hp}/{opp.max_hp}   items: {inv_str(opp.inventory)}")
    print(f"Chamber ({n_live}L + {n_blank}B): {chamber}   [slot 0 = next to fire]")
    if cuff_line:
        print(cuff_line)
    print("─" * 64)


def legal_action_list(mask: np.ndarray) -> list[int]:
    return [i for i in range(NUM_ACTIONS) if mask[i]]


def pretty_action(a: int) -> str:
    return Action(a).name


def human_pick(mask: np.ndarray) -> int:
    legal = legal_action_list(mask)
    print("Your legal actions:")
    for idx, a in enumerate(legal):
        print(f"  [{idx}] {pretty_action(a)}")
    while True:
        try:
            s = input("Choose [number or action name]: ").strip()
        except EOFError:
            print("\n(EOF; exiting)")
            sys.exit(0)
        if not s:
            continue
        if s.isdigit():
            k = int(s)
            if 0 <= k < len(legal):
                return legal[k]
        else:
            s_up = s.upper().replace(" ", "_")
            for a in legal:
                if Action(a).name == s_up or Action(a).name.endswith("_" + s_up):
                    return a
        print(f"  (invalid: '{s}' — enter 0..{len(legal)-1} or an action name)")


def bot_pick(policy: ActorCritic, obs: np.ndarray, mask: np.ndarray, sample: bool) -> int:
    with torch.no_grad():
        o = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        m = torch.from_numpy(mask.astype(np.int8)).unsqueeze(0)
        logits, value = policy.forward(o)
        masked = logits.masked_fill(m == 0, -1e8)
        probs = torch.softmax(masked, dim=-1).squeeze(0).numpy()

    # Top 3 legal actions
    legal = [(i, float(probs[i])) for i in range(NUM_ACTIONS) if mask[i]]
    legal.sort(key=lambda x: -x[1])
    print(f"BOT is thinking...  (value estimate: {value.item():+.2f})")
    for i, (a, p) in enumerate(legal[:3]):
        tag = "→" if i == 0 else " "
        print(f"  {tag} {pretty_action(a):<25} p={p:.3f}")

    if sample:
        legal_p = np.array([probs[a] for a, _ in legal])
        legal_p = legal_p / legal_p.sum()
        idx = int(np.random.choice(len(legal), p=legal_p))
        return legal[idx][0]
    return legal[0][0]


def describe_step(action: int, actor: str, info: dict) -> None:
    base = f"  {actor} → {pretty_action(action)}"
    extras = []
    if "shot" in info:
        kind, target_pid, dmg = info["shot"]
        target = "self" if target_pid == (0 if actor == "BOT" else 1) else "opponent"
        # Shot target from engine is pid-based. Simpler: just state live/blank.
        extras.append(f"shot={kind}" + (f" dmg={dmg}" if dmg else ""))
    if "beer_ejected" in info:
        extras.append(f"BEER ejected a {info['beer_ejected']} shell")
    if "glass" in info:
        extras.append(f"GLASS revealed next = {info['glass']}")
    if "phone" in info:
        pos, kind = info["phone"]
        extras.append(f"PHONE revealed pos {pos} = {kind}")
    if "pills" in info:
        extras.append(f"PILLS: {info['pills']}")
    if extras:
        base += "   [" + " | ".join(extras) + "]"
    print(base)


SCENARIOS = {
    "beer_when_certain_death": {
        "description": "You at 1HP, next shell known LIVE, you have BEER. Use BEER to survive.",
        "shells": [True, False, True, True],
        "known_by_current": {0: True},
        "me_hp": 1,
        "me_items": {Item.BEER: 1},
        "opp_items": {},
    },
    "inverter_save": {
        "description": "You at 1HP, next shell known LIVE, you have INVERTER. Flip the shell to survive.",
        "shells": [True, False, True],
        "known_by_current": {0: True},
        "me_hp": 1,
        "me_items": {Item.INVERTER: 1},
        "opp_items": {},
    },
    "saw_lethal": {
        "description": "Opp at 2HP, next known LIVE, you have HANDSAW. Saw → shoot for clean kill.",
        "shells": [True, False, True],
        "known_by_current": {0: True},
        "me_hp": 3,
        "opp_hp": 2,
        "me_items": {Item.HANDSAW: 1},
        "opp_items": {},
    },
    "cuff_saw_combo": {
        "description": "Opp 2HP, next known LIVE, you have CUFF + SAW. Full combo: cuff → saw → shoot.",
        "shells": [True, False, True],
        "known_by_current": {0: True},
        "me_hp": 3,
        "opp_hp": 2,
        "me_items": {Item.HANDCUFF: 1, Item.HANDSAW: 1},
        "opp_items": {},
    },
}


def apply_scenario(e: BuckshotEngine, name: str, human_pid: int) -> None:
    sc = SCENARIOS[name]
    print(f"\n*** Scenario: {name} ***")
    print(f"*** {sc['description']} ***\n")
    e.state.current_player = human_pid  # human is the one facing the dilemma
    me = e.state.players[human_pid]
    opp = e.state.players[1 - human_pid]
    me.hp = sc.get("me_hp", me.max_hp)
    opp.hp = sc.get("opp_hp", opp.max_hp)
    e.state.shells = list(sc["shells"])
    e.state.known_shells = [{}, {}]
    e.state.known_shells[human_pid] = dict(sc.get("known_by_current", {}))
    me.inventory = np.zeros(NUM_ITEMS, dtype=np.int32)
    opp.inventory = np.zeros(NUM_ITEMS, dtype=np.int32)
    for item, n in sc.get("me_items", {}).items():
        me.inventory[int(item)] = n
    for item, n in sc.get("opp_items", {}).items():
        opp.inventory[int(item)] = n
    e.state.damage_mult = 1
    e.state.adrenaline_active = False
    me.skip_next_turn = False
    opp.skip_next_turn = False
    e.state.done = False
    e.state.winner = None


def play(policy: ActorCritic, human_pid: int = 0, seed: int | None = None,
         scenario: str | None = None, sample: bool = False) -> None:
    e = BuckshotEngine()
    e.reset(seed=seed)
    if scenario:
        apply_scenario(e, scenario, human_pid)

    while not e.state.done:
        render_state(e, human_pid)
        if e.state.current_player == human_pid:
            mask = e.legal_actions()
            if not mask.any():
                print("(no legal actions — game engine error?)")
                break
            a = human_pick(mask)
            _, _, done, info = e.step(int(a))
            describe_step(a, "YOU", info)
        else:
            obs = e.observation(1 - human_pid)
            mask = e.legal_actions()
            if not mask.any():
                print("(bot has no legal actions — game engine error?)")
                break
            a = bot_pick(policy, obs, mask, sample=sample)
            _, _, done, info = e.step(int(a))
            describe_step(a, "BOT", info)

    print()
    print("═" * 64)
    if e.state.winner == human_pid:
        print("  >>> YOU WIN <<<")
    else:
        print("  >>> BOT WINS <<<")
    print("═" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", nargs="?", default=DEFAULT_CKPT)
    p.add_argument("--seat", type=int, choices=[0, 1], default=None,
                   help="Force human to play as player 0 or 1 (default: coin flip).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), default=None,
                   help="Start from a crafted probe scenario instead of a random game.")
    p.add_argument("--sample", action="store_true",
                   help="Bot samples from policy (default: argmax).")
    a = p.parse_args()

    if not os.path.exists(a.checkpoint):
        print(f"ERROR: checkpoint not found: {a.checkpoint}")
        print(f"Default is {DEFAULT_CKPT} — run from the repo root, or pass a path.")
        return 1

    policy = load_policy(a.checkpoint)
    human_pid = a.seat if a.seat is not None else int(np.random.default_rng(a.seed).integers(0, 2))
    print(f"\nLoaded: {a.checkpoint}")
    print(f"You are Player {human_pid}.  Bot is Player {1 - human_pid}.")
    play(policy, human_pid=human_pid, seed=a.seed, scenario=a.scenario, sample=a.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
