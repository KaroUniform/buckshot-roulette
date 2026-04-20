"""Interactive CLI demo: play Buckshot Roulette against a trained bot.

Usage:
    python -m rl.play_demo                                       # default: league champion
    python -m rl.play_demo path/to/policy.pt                     # custom checkpoint
    python -m rl.play_demo --seat 0                              # force human plays first
    python -m rl.play_demo --scenario beer_when_certain_death    # start from crafted probe
    python -m rl.play_demo --speed fast                          # faster bot pacing

The bot's top-3 action probabilities are displayed each turn so you can
see what it was "thinking" — useful for verifying whether probe-measured
failures reproduce in real play.

Redesigned UI (April 2026):
  * Stable board header redrawn in place (not appended) — no wall of text.
  * Colored event log with emoji per event type.
  * Bot actions paced with configurable delay + "thinking" reveal.
  * Russian-flavored visible strings; action enum names kept for research.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import deque

import numpy as np
import torch

from rl.engine import Action, BuckshotEngine, Item, NUM_ACTIONS, NUM_ITEMS
from rl.policy import ActorCritic, load_policy as _load_policy


DEFAULT_CKPT = "rl_runs/league_v1/A_ent005_gen2/policy_final.pt"

# ─── ANSI helpers ────────────────────────────────────────────────────────────

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"


def _ansi(code: str) -> str:
    return f"\x1b[{code}m"


# 16-color palette shortcuts
FG = {
    "black": 30, "red": 31, "green": 32, "yellow": 33,
    "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
    "bright_red": 91, "bright_green": 92, "bright_yellow": 93,
    "bright_blue": 94, "bright_magenta": 95, "bright_cyan": 96,
    "bright_white": 97, "gray": 90,
}


def color(text: str, fg: str | None = None, bold: bool = False, dim: bool = False) -> str:
    """Wrap text with ANSI color codes. Safe no-op if fg is None and no flags."""
    codes = []
    if bold:
        codes.append("1")
    if dim:
        codes.append("2")
    if fg is not None:
        codes.append(str(FG[fg]))
    if not codes:
        return text
    return f"\x1b[{';'.join(codes)}m{text}{RESET}"


def clear_screen() -> None:
    # \x1b[2J clears the whole screen; \x1b[H homes the cursor.
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


# ─── Emoji / glyph maps ──────────────────────────────────────────────────────

ITEM_EMOJI = {
    Item.HANDSAW: "🔪",
    Item.BEER: "🍺",
    Item.SMOKE: "🚬",
    Item.HANDCUFF: "🔗",
    Item.GLASS: "🔍",
    Item.PHONE: "📞",
    Item.PILLS: "💊",
    Item.ADRENALINE: "💉",
    Item.INVERTER: "🔄",
}

ITEM_RU = {
    Item.HANDSAW: "ножовка",
    Item.BEER: "пиво",
    Item.SMOKE: "сигарета",
    Item.HANDCUFF: "наручники",
    Item.GLASS: "лупа",
    Item.PHONE: "телефон",
    Item.PILLS: "таблетки",
    Item.ADRENALINE: "адреналин",
    Item.INVERTER: "инвертор",
}

ACTION_RU = {
    Action.SHOOT_OPPONENT: "Выстрел в оппонента",
    Action.SHOOT_SELF: "Выстрел в себя",
    Action.USE_HANDSAW: "Ножовка (x2 урон)",
    Action.USE_BEER: "Пиво (выплюнуть патрон)",
    Action.USE_SMOKE: "Сигарета (+1 HP)",
    Action.USE_HANDCUFF: "Наручники (пропуск хода)",
    Action.USE_GLASS: "Лупа (посмотреть патрон)",
    Action.USE_PHONE: "Телефон (случайный патрон)",
    Action.USE_PILLS: "Таблетки (риск)",
    Action.USE_ADRENALINE: "Адреналин (украсть предмет)",
    Action.USE_INVERTER: "Инвертор (перевернуть патрон)",
    Action.PICK_HANDSAW: "Украсть: ножовку",
    Action.PICK_BEER: "Украсть: пиво",
    Action.PICK_SMOKE: "Украсть: сигарету",
    Action.PICK_HANDCUFF: "Украсть: наручники",
    Action.PICK_GLASS: "Украсть: лупу",
    Action.PICK_PHONE: "Украсть: телефон",
    Action.PICK_PILLS: "Украсть: таблетки",
    Action.PICK_INVERTER: "Украсть: инвертор",
}

SPEED_DELAYS = {"slow": 1.6, "normal": 0.8, "fast": 0.15}


# ─── Load policy (kept for backwards compatibility) ──────────────────────────

def load_policy(path: str, device: str = "cpu") -> ActorCritic:
    return _load_policy(path, n_actions=NUM_ACTIONS, device=device)


# ─── Inventory rendering ─────────────────────────────────────────────────────

def inv_str(inv: np.ndarray) -> str:
    """Plain-text inventory (no color, no emoji) — kept for legacy/tests."""
    parts = []
    for i in Item:
        n = int(inv[int(i)])
        if n > 0:
            parts.append(f"{i.name.lower()}×{n}" if n > 1 else i.name.lower())
    return ", ".join(parts) if parts else "(empty)"


def inv_pretty(inv: np.ndarray) -> str:
    parts = []
    for i in Item:
        n = int(inv[int(i)])
        if n > 0:
            chunk = ITEM_EMOJI[i]
            if n > 1:
                chunk += color(f"×{n}", fg="bright_yellow", bold=True)
            parts.append(chunk)
    return " ".join(parts) if parts else color("(пусто)", fg="gray", dim=True)


# ─── HP bar ──────────────────────────────────────────────────────────────────

def hp_bar(hp: int, max_hp: int, is_human: bool) -> str:
    hearts_full = "♥" * max(0, hp)
    hearts_empty = "♡" * max(0, max_hp - hp)
    fg = "bright_green" if is_human else "bright_red"
    full = color(hearts_full, fg=fg, bold=True)
    empty = color(hearts_empty, fg="gray", dim=True)
    return f"{full}{empty} {color(f'{hp}/{max_hp}', fg='white', bold=True)}"


# ─── Chamber rendering ───────────────────────────────────────────────────────

def chamber_str(shells: list, known: dict) -> str:
    if not shells:
        return color("(перезарядка)", fg="gray", dim=True)
    slots = []
    for pos in range(len(shells)):
        if pos in known:
            if known[pos]:
                slots.append(color("🔴", fg="bright_red"))
            else:
                slots.append(color("⚪", fg="bright_white"))
        else:
            slots.append(color("?", fg="gray", dim=True))
    return " ".join(slots)


# ─── Board header (clears + redraws in place) ────────────────────────────────

def render_header(e: BuckshotEngine, human_pid: int, turn_idx: int) -> None:
    s = e.state
    me = s.players[human_pid]
    opp = s.players[1 - human_pid]
    n = len(s.shells)
    n_live = sum(1 for x in s.shells if x)
    n_blank = n - n_live
    known = s.known_shells[human_pid]
    width = min(term_width(), 80)

    clear_screen()

    # Top banner
    turn_who = "ТВОЙ ХОД" if s.current_player == human_pid else "ХОД БОТА"
    turn_fg = "bright_cyan" if s.current_player == human_pid else "bright_magenta"
    banner = f"Ход #{turn_idx}  —  {turn_who}"
    pad_left = max(0, (width - len(banner) - 2) // 2)
    pad_right = max(0, width - len(banner) - pad_left - 2)
    print(color("╔" + "═" * (width - 2) + "╗", fg=turn_fg, bold=True))
    print(
        color("║", fg=turn_fg, bold=True)
        + " " * pad_left
        + color(banner, fg=turn_fg, bold=True)
        + " " * pad_right
        + color("║", fg=turn_fg, bold=True)
    )
    print(color("╠" + "═" * (width - 2) + "╣", fg=turn_fg, bold=True))

    # Player lines
    you_label = color("🧑 ТЫ   ", fg="bright_green", bold=True)
    bot_label = color("🤖 БОТ  ", fg="bright_red", bold=True)
    print(f"  {you_label}  HP: {hp_bar(me.hp, me.max_hp, is_human=True)}   {inv_pretty(me.inventory)}")
    print(f"  {bot_label}  HP: {hp_bar(opp.hp, opp.max_hp, is_human=False)}   {inv_pretty(opp.inventory)}")

    # Chamber line
    chamber_line = (
        f"  {color('Дробовик', fg='yellow', bold=True)}  "
        f"[{color(str(n_live), fg='bright_red', bold=True)} боевых + "
        f"{color(str(n_blank), fg='bright_white', bold=True)} холостых]   "
        f"{chamber_str(s.shells, known)}"
        f"  {color('← след.', fg='gray', dim=True)}"
    )
    print(chamber_line)

    # Status line (damage mult, adrenaline, cuffs)
    status_parts: list[str] = []
    if s.damage_mult > 1:
        status_parts.append(color(f"⚠ УРОН x{s.damage_mult}", fg="bright_red", bold=True))
    if s.adrenaline_active:
        status_parts.append(color("💉 АДРЕНАЛИН: выбери предмет", fg="bright_yellow", bold=True))
    if me.skip_next_turn:
        status_parts.append(color("🔗 Ты пропускаешь ход", fg="bright_yellow"))
    if opp.skip_next_turn:
        status_parts.append(color("🔗 Бот пропускает ход", fg="bright_yellow"))
    if status_parts:
        print("  " + "  ".join(status_parts))

    print(color("╚" + "═" * (width - 2) + "╝", fg=turn_fg, bold=True))


# ─── Event log ───────────────────────────────────────────────────────────────

class EventLog:
    """Ring buffer of formatted log lines, rendered under the header."""

    def __init__(self, max_lines: int = 8) -> None:
        self.lines: deque[str] = deque(maxlen=max_lines)

    def add(self, line: str) -> None:
        self.lines.append(line)

    def render(self) -> None:
        if not self.lines:
            return
        print(color("  Журнал:", fg="gray", dim=True))
        for line in self.lines:
            print(f"    {line}")


# ─── Action descriptions ─────────────────────────────────────────────────────

def pretty_action(a: int) -> str:
    """Enum name for research-oriented top-3 block (don't break this)."""
    return Action(a).name


def action_ru(a: int) -> str:
    return ACTION_RU.get(Action(a), Action(a).name)


def _actor_label(actor: str) -> str:
    if actor == "YOU":
        return color("🧑 ТЫ  ", fg="bright_green", bold=True)
    return color("🤖 БОТ ", fg="bright_red", bold=True)


def describe_event(action: int, actor: str, info: dict) -> str:
    """Return a pretty colored one-line description of the step outcome."""
    label = _actor_label(actor)
    act_name = action_ru(action)
    head = f"{label} → {color(act_name, fg='white', bold=True)}"
    extras: list[str] = []

    if "shot" in info:
        kind, target_pid, dmg = info["shot"]
        if kind == "live":
            icon = color("🔴 БОЕВОЙ", fg="bright_red", bold=True)
            dmg_part = color(f"  −{dmg} HP", fg="bright_red", bold=True) if dmg else ""
            extras.append(f"{icon}{dmg_part}")
        else:
            extras.append(color("⚪ Холостой", fg="bright_white", bold=True))

    if "beer_ejected" in info:
        ej = info["beer_ejected"]
        icon = "🔴" if ej == "live" else "⚪"
        col = "bright_red" if ej == "live" else "bright_white"
        extras.append(f"🍺 {color('выплюнул', fg='yellow')} {color(icon + ' ' + ej, fg=col, bold=True)}")

    if "glass" in info:
        g = info["glass"]
        icon = "🔴" if g == "live" else "⚪"
        col = "bright_red" if g == "live" else "bright_white"
        extras.append(f"🔍 {color('след. патрон:', fg='cyan')} {color(icon + ' ' + g, fg=col, bold=True)}")

    if "phone" in info:
        pos, kind = info["phone"]
        icon = "🔴" if kind == "live" else "⚪"
        col = "bright_red" if kind == "live" else "bright_white"
        extras.append(
            f"📞 {color('поз.', fg='cyan')} {color(str(pos), fg='bright_cyan', bold=True)}: "
            f"{color(icon + ' ' + kind, fg=col, bold=True)}"
        )

    if "pills" in info:
        kind = info["pills"]
        if kind == "good":
            extras.append(color("💊 +2 HP", fg="bright_green", bold=True))
        else:
            extras.append(color("💊 −1 HP (неудача)", fg="bright_red", bold=True))

    if extras:
        head += "   " + color("│", fg="gray", dim=True) + "  " + "   ".join(extras)
    return head


# ─── Top-3 bot probs block ───────────────────────────────────────────────────

def top_k_action_info(
    policy: ActorCritic, obs: np.ndarray, mask: np.ndarray, k: int = 3
) -> tuple[list[tuple[int, float]], float]:
    """Return (top_k_legal_actions, value_estimate). Pure — no printing."""
    with torch.no_grad():
        o = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        m = torch.from_numpy(mask.astype(np.int8)).unsqueeze(0)
        logits, value = policy.forward(o)
        masked = logits.masked_fill(m == 0, -1e8)
        probs = torch.softmax(masked, dim=-1).squeeze(0).numpy()
    legal = [(i, float(probs[i])) for i in range(NUM_ACTIONS) if mask[i]]
    legal.sort(key=lambda x: -x[1])
    return legal[:k], float(value.item())


def render_thinking(top_k: list[tuple[int, float]], value: float) -> None:
    print()
    print(color("  🤖 Бот думает...", fg="bright_magenta", bold=True)
          + color(f"  (оценка позиции: {value:+.2f})", fg="gray", dim=True))
    for i, (a, p) in enumerate(top_k):
        arrow = color("→", fg="bright_green", bold=True) if i == 0 else " "
        bar_len = int(round(p * 20))
        bar = color("█" * bar_len, fg="bright_magenta") + color("░" * (20 - bar_len), fg="gray", dim=True)
        name = pretty_action(a)  # enum name — research value
        ru_name = action_ru(a)
        line = (
            f"    {arrow} {bar}  "
            f"{color(f'{p*100:5.1f}%', fg='bright_yellow', bold=True)}  "
            f"{color(name, fg='cyan'):<30}  "
            f"{color(ru_name, fg='gray', dim=True)}"
        )
        print(line)
    sys.stdout.flush()


# ─── Human input ─────────────────────────────────────────────────────────────

def legal_action_list(mask: np.ndarray) -> list[int]:
    return [i for i in range(NUM_ACTIONS) if mask[i]]


def human_pick(mask: np.ndarray) -> int:
    legal = legal_action_list(mask)
    print()
    print(color("  Твои возможные действия:", fg="bright_cyan", bold=True))
    for idx, a in enumerate(legal):
        num = color(f"[{idx}]", fg="bright_yellow", bold=True)
        ru = color(action_ru(a), fg="white")
        enum = color(pretty_action(a), fg="gray", dim=True)
        print(f"    {num} {ru}  {enum}")
    while True:
        try:
            prompt = color("  Выбор [номер или имя действия]: ", fg="bright_cyan", bold=True)
            s = input(prompt).strip()
        except EOFError:
            print("\n(EOF; выход)")
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
        print(color(f"    (неверно: '{s}' — введи 0..{len(legal)-1} или имя действия)",
                    fg="bright_red"))


# ─── Bot pick ────────────────────────────────────────────────────────────────

def choose_bot_action(
    policy: ActorCritic, obs: np.ndarray, mask: np.ndarray, sample: bool,
    top_k: list[tuple[int, float]] | None = None,
) -> int:
    """Return the bot's chosen action. Optionally uses a pre-computed top-k
    to avoid recomputing probs (so rendering shows the exact distribution used)."""
    if top_k is None:
        top_k, _ = top_k_action_info(policy, obs, mask, k=NUM_ACTIONS)

    if sample:
        legal = [(a, p) for a, p in top_k if p > 0]
        if not legal:
            # Defensive: mask nonzero somewhere, fallback
            legal = [(a, 1.0) for a in range(NUM_ACTIONS) if mask[a]]
        actions = np.array([a for a, _ in legal])
        probs = np.array([p for _, p in legal], dtype=np.float64)
        probs = probs / probs.sum()
        return int(actions[np.random.choice(len(actions), p=probs)])
    return top_k[0][0]


# ─── Full-screen compose: header + log ───────────────────────────────────────

def refresh(
    e: BuckshotEngine, human_pid: int, turn_idx: int, log: EventLog,
    thinking: list[tuple[int, float]] | None = None, thinking_value: float = 0.0,
) -> None:
    render_header(e, human_pid, turn_idx)
    print()
    log.render()
    if thinking is not None:
        render_thinking(thinking, thinking_value)
    sys.stdout.flush()


# ─── Scenarios (preserved verbatim) ──────────────────────────────────────────

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


def apply_scenario(e: BuckshotEngine, name: str, human_pid: int, log: EventLog) -> None:
    sc = SCENARIOS[name]
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

    log.add(color(f"🎬 Сценарий: {name}", fg="bright_cyan", bold=True))
    log.add(color(f"   {sc['description']}", fg="cyan", dim=True))


# ─── Game-over banner ────────────────────────────────────────────────────────

def render_gameover(e: BuckshotEngine, human_pid: int, turn_idx: int, log: EventLog) -> None:
    refresh(e, human_pid, turn_idx, log)
    print()
    width = min(term_width(), 80)
    if e.state.winner == human_pid:
        emoji = "🏆"
        text = "ПОБЕДА!"
        fg = "bright_green"
    else:
        emoji = "💀"
        text = "ПОРАЖЕНИЕ"
        fg = "bright_red"

    banner = f"{emoji}  {text}  {emoji}"
    pad_left = max(0, (width - len(banner) - 2) // 2)
    pad_right = max(0, width - len(banner) - pad_left - 2)
    print(color("╔" + "═" * (width - 2) + "╗", fg=fg, bold=True))
    print(color("║" + " " * (width - 2) + "║", fg=fg, bold=True))
    print(
        color("║", fg=fg, bold=True)
        + " " * pad_left
        + color(banner, fg=fg, bold=True)
        + " " * pad_right
        + color("║", fg=fg, bold=True)
    )
    print(color("║" + " " * (width - 2) + "║", fg=fg, bold=True))
    print(color("╚" + "═" * (width - 2) + "╝", fg=fg, bold=True))
    print()


# ─── describe_step: legacy shim kept for API parity ──────────────────────────

def describe_step(action: int, actor: str, info: dict) -> None:
    """Legacy helper — prints a single line. The new UI uses EventLog directly,
    but this is kept in case anything imports it."""
    print(describe_event(action, actor, info))


# ─── Main game loop ──────────────────────────────────────────────────────────

def play(
    policy: ActorCritic,
    human_pid: int = 0,
    seed: int | None = None,
    scenario: str | None = None,
    sample: bool = False,
    speed: str = "normal",
) -> None:
    e = BuckshotEngine()
    e.reset(seed=seed)
    log = EventLog(max_lines=8)
    if scenario:
        apply_scenario(e, scenario, human_pid, log)

    delay = SPEED_DELAYS.get(speed, SPEED_DELAYS["normal"])
    # "Thinking" reveal is ~half of the main delay, but at least 0.1s
    think_delay = max(0.1, delay * 0.5) if speed != "fast" else 0.1

    turn_idx = 0
    last_actor = None

    while not e.state.done:
        # Bump turn counter on actor change
        if e.state.current_player != last_actor:
            turn_idx += 1
            last_actor = e.state.current_player

        if e.state.current_player == human_pid:
            refresh(e, human_pid, turn_idx, log)
            mask = e.legal_actions()
            if not mask.any():
                log.add(color("⚠ нет законных действий (ошибка движка?)", fg="bright_red"))
                refresh(e, human_pid, turn_idx, log)
                break
            a = human_pick(mask)
            _, _, _, info = e.step(int(a))
            log.add(describe_event(a, "YOU", info))
        else:
            obs = e.observation(1 - human_pid)
            mask = e.legal_actions()
            if not mask.any():
                log.add(color("⚠ у бота нет ходов (ошибка движка?)", fg="bright_red"))
                refresh(e, human_pid, turn_idx, log)
                break
            top_k, value = top_k_action_info(policy, obs, mask, k=3)
            # Show "thinking" state first, then the chosen action.
            refresh(e, human_pid, turn_idx, log, thinking=top_k, thinking_value=value)
            time.sleep(think_delay)
            a = choose_bot_action(policy, obs, mask, sample=sample, top_k=top_k)
            _, _, _, info = e.step(int(a))
            log.add(describe_event(a, "BOT", info))
            refresh(e, human_pid, turn_idx, log)
            time.sleep(delay)

    render_gameover(e, human_pid, turn_idx, log)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint", nargs="?", default=DEFAULT_CKPT)
    p.add_argument("--seat", type=int, choices=[0, 1], default=None,
                   help="Force human to play as player 0 or 1 (default: coin flip).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), default=None,
                   help="Start from a crafted probe scenario instead of a random game.")
    p.add_argument("--sample", action="store_true",
                   help="Bot samples from policy (default: argmax).")
    p.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal",
                   help="Bot pacing speed (slow ≈ 1.6s, normal ≈ 0.8s, fast ≈ 0.15s between actions).")
    a = p.parse_args()

    if not os.path.exists(a.checkpoint):
        print(color(f"ERROR: checkpoint not found: {a.checkpoint}", fg="bright_red", bold=True))
        print(f"Default is {DEFAULT_CKPT} — run from the repo root, or pass a path.")
        return 1

    policy = load_policy(a.checkpoint)
    human_pid = a.seat if a.seat is not None else int(np.random.default_rng(a.seed).integers(0, 2))
    clear_screen()
    print(color(f"✓ Загружено: {a.checkpoint}", fg="bright_green", bold=True))
    print(f"  Ты — Игрок {human_pid}.  Бот — Игрок {1 - human_pid}.  "
          f"Скорость: {color(a.speed, fg='bright_yellow', bold=True)}")
    time.sleep(0.6)
    play(policy, human_pid=human_pid, seed=a.seed, scenario=a.scenario,
         sample=a.sample, speed=a.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
