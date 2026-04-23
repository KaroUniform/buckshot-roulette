"""Shared Telegram rendering for canonical rl.engine-backed games."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from rl.engine import Action, GameState, Item

from . import actions as _actions


ITEM_ORDER = [
    Item.HANDSAW,
    Item.BEER,
    Item.SMOKE,
    Item.HANDCUFF,
    Item.GLASS,
    Item.PHONE,
    Item.PILLS,
    Item.ADRENALINE,
    Item.INVERTER,
]

_PICK_ACTION_TO_ITEM = {
    Action.PICK_HANDSAW: Item.HANDSAW,
    Action.PICK_BEER: Item.BEER,
    Action.PICK_SMOKE: Item.SMOKE,
    Action.PICK_HANDCUFF: Item.HANDCUFF,
    Action.PICK_GLASS: Item.GLASS,
    Action.PICK_PHONE: Item.PHONE,
    Action.PICK_PILLS: Item.PILLS,
    Action.PICK_INVERTER: Item.INVERTER,
}

_SEPARATOR = "━━━━━━━━━━━━━━━"


def _label(names: Sequence[str], viewer_id: int, player_id: int) -> str:
    return "You" if viewer_id == player_id else names[player_id]


def _hp_bar(hp: int) -> str:
    return "☠️" if hp <= 0 else "⚡️" * hp


def inventory_emoji(state: GameState, player_id: int) -> List[str]:
    out: List[str] = []
    inv = state.players[player_id].inventory
    for item in ITEM_ORDER:
        count = int(inv[int(item)])
        if count > 0:
            out.append(f"{_actions.item_emoji(item)}x{count}")
    return out


def _items_line(state: GameState, player_id: int) -> str:
    items = inventory_emoji(state, player_id)
    return "Items: " + (", ".join(items) if items else "—")


def game_summary(state: GameState, names: Sequence[str], viewer_id: int) -> str:
    ordered = [viewer_id, 1 - viewer_id]
    hp_lines: List[str] = []
    item_lines: List[str] = []
    status_lines: List[str] = []
    for player_id in ordered:
        marker = "▶️ " if state.current_player == player_id and not state.done else ""
        hp_lines.append(
            f"{marker}{_label(names, viewer_id, player_id)}: {_hp_bar(state.players[player_id].hp)}"
        )
        item_lines.append(f"{_label(names, viewer_id, player_id)}: {', '.join(inventory_emoji(state, player_id)) or '—'}")
    if state.damage_mult > 1:
        status_lines.append("🪚 The next live shot deals 2× damage")
    if state.adrenaline_active and state.current_player == viewer_id:
        status_lines.append("💉 Adrenaline is active: steal one item now")
    if state.players[viewer_id].skip_next_turn:
        status_lines.append("🔗 You are cuffed and will skip the next turn")
    elif state.players[1 - viewer_id].skip_next_turn:
        status_lines.append(f"🔗 {_label(names, viewer_id, 1 - viewer_id)} is cuffed")

    lines = ["HP:"]
    lines.extend(hp_lines)
    lines.append("")
    lines.append("Items:")
    lines.extend(item_lines)
    if status_lines:
        lines.append("")
        lines.append("Status:")
        lines.extend(status_lines)
    return "\n".join(lines)


def opening_text(state: GameState, names: Sequence[str], viewer_id: int, *, ai_mode: bool) -> str:
    if ai_mode:
        header = "▶️ You go first."
    else:
        starter = _label(names, viewer_id, state.current_player)
        verb = "go" if starter == "You" else "goes"
        header = f"🪙 Coin toss — {starter} {verb} first."
    return f"{header}\n\n{game_summary(state, names, viewer_id)}"


def reload_banner(state: GameState, names: Sequence[str], viewer_id: int) -> str:
    live = int(state.round_initial_live)
    blank = int(state.round_initial_blank)
    ordered = [viewer_id, 1 - viewer_id]
    lines = [
        _SEPARATOR,
        "🔄 NEW ROUND",
        f"Shells: 💥×{live}  🫧×{blank}",
        "",
    ]
    for player_id in ordered:
        lines.append(f"{_label(names, viewer_id, player_id)}: {' '.join(inventory_emoji(state, player_id)) or '—'}")
    lines.append(_SEPARATOR)
    return "\n".join(lines)


def wait_keyboard(label: str) -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton(text=label))
    return builder.as_markup(resize_keyboard=True)


def _chunk_buttons(builder: ReplyKeyboardBuilder, buttons: Iterable[KeyboardButton], width: int = 3):
    row: List[KeyboardButton] = []
    for button in buttons:
        row.append(button)
        if len(row) == width:
            builder.row(*row)
            row = []
    if row:
        builder.row(*row)


def turn_keyboard(engine, player_id: int):
    if getattr(engine.state, "done", False):
        return ReplyKeyboardRemove()
    if engine.state.current_player != player_id:
        return ReplyKeyboardRemove()

    state = engine.state
    legal = engine.legal_actions()
    builder = ReplyKeyboardBuilder()

    if state.adrenaline_active:
        picks = []
        for item in ITEM_ORDER:
            if item == Item.ADRENALINE:
                continue
            pick_action_int = {
                Item.HANDSAW: int(Action.PICK_HANDSAW),
                Item.BEER: int(Action.PICK_BEER),
                Item.SMOKE: int(Action.PICK_SMOKE),
                Item.HANDCUFF: int(Action.PICK_HANDCUFF),
                Item.GLASS: int(Action.PICK_GLASS),
                Item.PHONE: int(Action.PICK_PHONE),
                Item.PILLS: int(Action.PICK_PILLS),
                Item.INVERTER: int(Action.PICK_INVERTER),
            }[item]
            if legal[pick_action_int]:
                picks.append(KeyboardButton(text=_actions.pick_emoji(item)))
        if picks:
            _chunk_buttons(builder, picks)
        else:
            builder.row(KeyboardButton(text=_actions.SHOOT_OPP_GLYPH))
            builder.row(KeyboardButton(text=_actions.SHOOT_SELF_GLYPH))
        return builder.as_markup(resize_keyboard=True)

    if legal[int(Action.SHOOT_OPPONENT)]:
        builder.row(KeyboardButton(text=_actions.SHOOT_OPP_GLYPH))

    item_buttons = []
    inv = state.players[player_id].inventory
    for item in ITEM_ORDER:
        if int(inv[int(item)]) > 0:
            item_buttons.append(
                KeyboardButton(text=f"{_actions.item_emoji(item)}x{int(inv[int(item)])}")
            )
    if item_buttons:
        _chunk_buttons(builder, item_buttons)

    if legal[int(Action.SHOOT_SELF)]:
        builder.row(KeyboardButton(text=_actions.SHOOT_SELF_GLYPH))

    kb = builder.as_markup(resize_keyboard=True)
    if not kb.keyboard:
        return ReplyKeyboardRemove()
    return kb


def action_caption(
    action: int,
    info: dict,
    *,
    actor_id: int,
    viewer_id: int,
    names: Sequence[str],
) -> str:
    a = Action(int(action))
    actor = _label(names, viewer_id, actor_id)
    target_id = 1 - actor_id
    target = _label(names, viewer_id, target_id)
    actor_is_viewer = actor_id == viewer_id

    if a == Action.SHOOT_OPPONENT:
        shot = info.get("shot")
        glyph = "💥" if shot and shot[0] == "live" else "🫧"
        suffix = f" ({shot[2]}× damage)" if shot and shot[0] == "live" else ""
        return f"{actor} shot {target} with {glyph}{suffix}"

    if a == Action.SHOOT_SELF:
        shot = info.get("shot")
        if shot and shot[0] == "blank":
            return f"{actor} shot {'yourself' if actor_is_viewer else 'themselves'} with 🫧 and kept the turn"
        return f"{actor} shot {'yourself' if actor_is_viewer else 'themselves'} with 💥"

    if a == Action.USE_HANDSAW:
        return f"{actor} used 🪚 — next live shot deals 2× damage"
    if a == Action.USE_BEER:
        glyph = "💥" if info.get("beer_ejected") == "live" else "🫧"
        return f"{actor} used 🍺 — {glyph} flew out of the shotgun"
    if a == Action.USE_SMOKE:
        return f"{actor} used 🚬 — healed 1⚡️"
    if a == Action.USE_HANDCUFF:
        return f"🔗 {actor} cuffed {target} — next turn is skipped"
    if a == Action.USE_GLASS:
        if actor_is_viewer:
            shell = info.get("glass")
            if shell == "live":
                return "🔍 You inspected the next shell — 💥 live"
            if shell == "blank":
                return "🔍 You inspected the next shell — 🫧 blank"
        return f"🔍 {actor} inspected the next shell"
    if a == Action.USE_PHONE:
        if actor_is_viewer:
            phone = info.get("phone")
            if phone is not None:
                pos, kind = phone
                glyph = "💥" if kind == "live" else "🫧"
                return f"📞 You got a tip — shell #{int(pos) + 1} is {glyph} {kind}"
        return f"📞 {actor} called for a shell hint"
    if a == Action.USE_PILLS:
        if info.get("pills") == "good":
            return f"💊 {actor} took pills — healed 2⚡️"
        return f"💊 {actor} took pills — lost 1⚡️"
    if a == Action.USE_ADRENALINE:
        return f"💉 {actor} used adrenaline — stealing one item…"
    if a == Action.USE_INVERTER:
        return f"🔀 {actor} used inverter — current shell polarity flipped"

    if a in _PICK_ACTION_TO_ITEM:
        item = _PICK_ACTION_TO_ITEM[a]
        prefix = f"💉 {actor} stole {_actions.item_emoji(item)} and used it"
        if item == Item.BEER:
            glyph = "💥" if info.get("beer_ejected") == "live" else "🫧"
            return f"{prefix} — {glyph} flew out of the shotgun"
        if item == Item.HANDSAW:
            return f"{prefix} — next live shot deals 2× damage"
        if item == Item.SMOKE:
            return f"{prefix} — healed 1⚡️"
        if item == Item.HANDCUFF:
            return f"{prefix} — {target} skips the next turn"
        if item == Item.PILLS:
            return f"{prefix} — {'healed 2⚡️' if info.get('pills') == 'good' else 'lost 1⚡️'}"
        if item == Item.INVERTER:
            return f"{prefix} — current shell polarity flipped"
        if item == Item.GLASS:
            if actor_is_viewer:
                shell = info.get("glass")
                if shell == "live":
                    return f"{prefix} — 💥 live"
                if shell == "blank":
                    return f"{prefix} — 🫧 blank"
            return f"{prefix} — inspected the next shell"
        if item == Item.PHONE:
            if actor_is_viewer:
                phone = info.get("phone")
                if phone is not None:
                    pos, kind = phone
                    glyph = "💥" if kind == "live" else "🫧"
                    return f"{prefix} — shell #{int(pos) + 1} is {glyph} {kind}"
            return f"{prefix} — got a shell hint"
        return prefix

    return f"{actor} took action {a.name}"


def game_over_message(
    state: GameState,
    names: Sequence[str],
    viewer_id: int,
    *,
    ai_mode: bool,
    ai_seat: int | None,
) -> str:
    if ai_mode and ai_seat is not None:
        if state.winner == ai_seat:
            return "⚰️ The AI won. Try /ai to play again."
        return "💼 You defeated the AI — gg!"

    if state.winner == viewer_id:
        return "💼 Congratulations, you've won!"
    return f"⚰️ {_label(names, viewer_id, state.winner)} won the round."
