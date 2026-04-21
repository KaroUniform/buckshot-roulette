"""Render BuckshotEngine state into Telegram-friendly messages + keyboards.

Style deliberately mirrors `handlers/rooms_manager.py` (the 2-player
flow) so switching between AI and multiplayer doesn't feel jarring:
same HP/⚡️ glyphs, same item emoji layout, same shoot-up / shoot-down
bottom/top button placement.
"""

from __future__ import annotations

from typing import List

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from rl.engine import Action, GameState, Item

from . import actions as _actions


_ITEM_ORDER = [
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


def hp_line(name: str, state: GameState, player_id: int) -> str:
    p = state.players[player_id]
    return f"{name} : {'⚡️' * p.hp}"


def inventory_emoji(state: GameState, player_id: int) -> List[str]:
    out: List[str] = []
    inv = state.players[player_id].inventory
    for item in _ITEM_ORDER:
        count = int(inv[int(item)])
        if count > 0:
            out.append(f"{_actions.item_emoji(item)}x{count}")
    return out


def loadout_line(state: GameState) -> str:
    """Human-facing initial-round declaration — matches the 2-player flow.

    Deliberately reveals both live and blank counts even under the
    "hack" obs layout. This is the information a real Buckshot Roulette
    round announcer gives ("N live, N blank") before the shells are
    shuffled, so showing it here doesn't leak more than a tabletop host
    would.
    """
    live = int(state.round_initial_live)
    blank = int(state.round_initial_blank)
    return f"Loadout: 💥×{live}  🫧×{blank}"


def _keyboard_for(state: GameState, human_id: int) -> ReplyKeyboardMarkup:
    """Build the reply keyboard the human sees based on engine legality."""
    from rl.engine import BuckshotEngine

    # Recompute legality from state (not from an engine instance) because
    # AIRoom holds the engine and we only want state here.
    # Cheap: reconstructing a transient engine with the shared state
    # lets us reuse BuckshotEngine.legal_actions without duplicating its
    # ~40 lines of gating logic.
    stub = BuckshotEngine.__new__(BuckshotEngine)
    stub.state = state
    legal = stub.legal_actions()

    builder = ReplyKeyboardBuilder()

    if state.adrenaline_active and state.current_player == human_id:
        # Pick-mode: only 💉<item> buttons for items the opponent has
        # AND the policy deemed usable.
        picks = []
        for item in _ITEM_ORDER:
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
            builder.row(*picks)
        else:
            # No legal picks — engine-side fallback makes shoot legal.
            builder.row(KeyboardButton(text=_actions.SHOOT_OPP_GLYPH))
            builder.row(KeyboardButton(text=_actions.SHOOT_SELF_GLYPH))
        return builder.as_markup(resize_keyboard=True)

    # Normal turn: shoot ↑, then items in the middle, shoot ↓.
    if legal[int(Action.SHOOT_OPPONENT)]:
        builder.row(KeyboardButton(text=_actions.SHOOT_OPP_GLYPH))

    item_buttons = []
    human_inv = state.players[human_id].inventory
    for item in _ITEM_ORDER:
        use_action = {
            Item.HANDSAW: int(Action.USE_HANDSAW),
            Item.BEER: int(Action.USE_BEER),
            Item.SMOKE: int(Action.USE_SMOKE),
            Item.HANDCUFF: int(Action.USE_HANDCUFF),
            Item.GLASS: int(Action.USE_GLASS),
            Item.PHONE: int(Action.USE_PHONE),
            Item.PILLS: int(Action.USE_PILLS),
            Item.ADRENALINE: int(Action.USE_ADRENALINE),
            Item.INVERTER: int(Action.USE_INVERTER),
        }[item]
        # Show only items the player owns; the engine decides usability.
        # An unusable-but-owned item stays on the keyboard so the player
        # can see it exists — pressing it triggers the illegal-action
        # path, which the handler turns into a polite "can't use that
        # right now" reply.
        if int(human_inv[int(item)]) > 0:
            item_buttons.append(
                KeyboardButton(text=f"{_actions.item_emoji(item)}x{int(human_inv[int(item)])}"),
            )
    if item_buttons:
        builder.row(*item_buttons)

    if legal[int(Action.SHOOT_SELF)]:
        builder.row(KeyboardButton(text=_actions.SHOOT_SELF_GLYPH))

    return builder.as_markup(resize_keyboard=True)


def wait_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton(text="🕓AI is thinking🕓"))
    return builder.as_markup(resize_keyboard=True)


def human_turn_keyboard(state: GameState, human_id: int) -> ReplyKeyboardMarkup:
    return _keyboard_for(state, human_id)


def game_summary(state: GameState, human_name: str, human_id: int) -> str:
    """Compact status block: both HP bars + both inventories."""
    ai_id = 1 - human_id
    lines = [
        hp_line(human_name, state, human_id),
        "Items: " + ", ".join(inventory_emoji(state, human_id)),
        hp_line("🤖 AI", state, ai_id),
        "Items: " + ", ".join(inventory_emoji(state, ai_id)),
    ]
    dmg = state.damage_mult
    if dmg > 1:
        lines.append(f"🪚 Shotgun damage is 2× until next shot")
    return "\n".join(lines)


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


def action_caption(action: int, info: dict, *, actor: str) -> str:
    """One-liner describing what just happened, rendered from the human's POV.

    `actor` is either "ai" (policy just moved) or "human" (the player
    did). We render each side with matching phrasing so the two event
    streams can share this helper.
    """
    a = Action(int(action))
    assert actor in ("ai", "human")
    is_ai = actor == "ai"
    subj = "🤖" if is_ai else "You"
    obj_opp = "you" if is_ai else "🤖"
    self_refl = "itself" if is_ai else "yourself"
    poss_opp = "your" if is_ai else "🤖's"
    aux = "are" if actor == "human" else "is"

    if a == Action.SHOOT_OPPONENT:
        shot = info.get("shot")
        if shot and shot[0] == "live":
            return f"{subj} shot {obj_opp} with a 💥 ({shot[2]}× damage)"
        return f"{subj} shot {obj_opp} with a 🫧"
    if a == Action.SHOOT_SELF:
        shot = info.get("shot")
        if shot and shot[0] == "blank":
            return f"{subj} shot {self_refl} with a 🫧 — keeps the turn"
        return f"{subj} shot {self_refl} with a 💥"
    if a == Action.USE_HANDSAW:
        return f"{subj} used 🪚 — damage is now 2× for the next shot"
    if a == Action.USE_BEER:
        ej = info.get("beer_ejected")
        glyph = "💥" if ej == "live" else "🫧"
        return f"{subj} used 🍺 — {glyph} flew out of the shotgun"
    if a == Action.USE_SMOKE:
        return f"{subj} used 🚬 — healed 1⚡️"
    if a == Action.USE_HANDCUFF:
        return f"🔗 {subj} cuffed {obj_opp} — skip the next turn"
    if a == Action.USE_GLASS:
        return f"🔍 {subj} inspected the next shell"
    if a == Action.USE_PHONE:
        return f"📞 {subj} phoned in a shell hint"
    if a == Action.USE_PILLS:
        kind = info.get("pills")
        if kind == "good":
            return f"💊 {subj} took pills — healed 2⚡️"
        return f"💊 {subj} took pills — lost 1⚡️"
    if a == Action.USE_ADRENALINE:
        return f"💉 {subj} used adrenaline — stealing one of {poss_opp} items…"
    if a == Action.USE_INVERTER:
        return f"🔀 {subj} used inverter — current shell polarity flipped"
    if a in _PICK_ACTION_TO_ITEM:
        item = _PICK_ACTION_TO_ITEM[a]
        return (
            f"💉 {subj} stole {poss_opp} {_actions.item_emoji(item)} "
            f"and {aux} using it"
        )
    return f"{subj} took action {a.name}"


def reload_banner() -> str:
    return "🔄 Shotgun reloaded — new round"


def game_over_message(state: GameState, human_name: str, human_id: int) -> str:
    if state.winner == human_id:
        return f"💼 {human_name} defeated the AI — gg!"
    return "⚰️ The AI won. Try /ai to play again."
