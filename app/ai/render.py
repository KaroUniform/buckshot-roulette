"""Render BuckshotEngine state into Telegram-friendly messages + keyboards.

Style deliberately mirrors `handlers/rooms_manager.py` (the 2-player
flow) so switching between AI and multiplayer doesn't feel jarring:
same HP/⚡️ glyphs, same item emoji layout, same shoot-up / shoot-down
bottom/top button placement.
"""

from __future__ import annotations

from typing import List, Optional

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


def _keyboard_for(engine, human_id: int) -> ReplyKeyboardMarkup:
    """Build the reply keyboard the human sees based on engine legality.

    Takes the live engine (not a free-standing state) because
    `legal_actions` is defined as a method and may grow dependencies on
    other engine attributes (rng, honest_obs, etc.); a previous
    `__new__`-stub approach worked only by coincidence of the current
    method body and would silently break if legality gained new
    instance-attribute reads.
    """
    state = engine.state
    legal = engine.legal_actions()

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


def human_turn_keyboard(engine, human_id: int):
    """Return the human's keyboard, or a safe fallback on terminal state.

    Takes the live engine so legality can be queried on an authentic
    engine instance (not a `__new__` stub — see `_keyboard_for`).

    Defensive: if the state is terminal (or the engine somehow reports no
    legal actions and no inventory), `_keyboard_for` produces an empty
    ReplyKeyboardMarkup which Telegram rejects. Callers should normally
    use `keyboard_hint="game_over"` when the engine reports done, but we
    guard here so a stray human_turn hint on a terminal state doesn't
    crash the handler.
    """
    if getattr(engine.state, "done", False):
        return ReplyKeyboardRemove()
    kb = _keyboard_for(engine, human_id)
    if not kb.keyboard:
        return ReplyKeyboardRemove()
    return kb


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
        # Human sees the reveal; AI's caption stays generic so we don't
        # leak the AI's private knowledge to the human watching.
        if actor == "human":
            shell = info.get("glass")
            if shell == "live":
                return f"🔍 You inspected the next shell — 💥 live"
            if shell == "blank":
                return f"🔍 You inspected the next shell — 🫧 blank"
        return f"🔍 {subj} inspected the next shell"
    if a == Action.USE_PHONE:
        if actor == "human":
            phone = info.get("phone")
            if phone is not None:
                pos, kind = phone
                glyph = "💥" if kind == "live" else "🫧"
                # 1-index for display: "#1" = the next shell in the chamber.
                return f"📞 You phoned — shell #{int(pos) + 1} is {glyph} {kind}"
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
        # The pick caption must include the item's outcome, same as the
        # USE_<item> captions would have. Otherwise a picked BEER silently
        # ejects a shell with no message — leaving the player confused
        # about where a live round went ("куда пропал один выстрел?").
        glyph = _actions.item_emoji(item)
        prefix = f"💉 {subj} stole {poss_opp} {glyph} and used it"
        if item == Item.BEER:
            ej = info.get("beer_ejected")
            out_glyph = "💥" if ej == "live" else "🫧"
            return f"{prefix} — {out_glyph} flew out of the shotgun"
        if item == Item.HANDSAW:
            return f"{prefix} — damage is now 2× for the next shot"
        if item == Item.SMOKE:
            return f"{prefix} — healed 1⚡️"
        if item == Item.HANDCUFF:
            return f"{prefix} — {obj_opp} skip the next turn"
        if item == Item.PILLS:
            kind = info.get("pills")
            if kind == "good":
                return f"{prefix} — healed 2⚡️"
            return f"{prefix} — lost 1⚡️"
        if item == Item.INVERTER:
            return f"{prefix} — current shell polarity flipped"
        # Glass and phone reveal private info to the picker only. Human
        # picker sees the reveal; AI picker stays opaque so we don't
        # leak its knowledge to the watching human.
        if item == Item.GLASS:
            if actor == "human":
                shell = info.get("glass")
                if shell == "live":
                    return f"{prefix} — 💥 live"
                if shell == "blank":
                    return f"{prefix} — 🫧 blank"
            return f"{prefix} — inspected the next shell"
        if item == Item.PHONE:
            if actor == "human":
                phone = info.get("phone")
                if phone is not None:
                    pos, kind = phone
                    gl = "💥" if kind == "live" else "🫧"
                    return f"{prefix} — shell #{int(pos) + 1} is {gl} {kind}"
            return f"{prefix} — got a shell hint"
        return prefix
    return f"{subj} took action {a.name}"


def _inventory_block(state: GameState, player_id: int) -> str:
    items = inventory_emoji(state, player_id)
    return " ".join(items) if items else "—"


_SEPARATOR = "━━━━━━━━━━━━━━━"


def reload_banner(
    state: Optional[GameState] = None,
    human_name: Optional[str] = None,
    human_id: Optional[int] = None,
) -> str:
    """Round-change announcement.

    With no args — the legacy compact form, still used as a fallback.
    With (state, human_name, human_id) — a richer multi-line message
    that includes the new loadout and both players' current inventory
    (items are distributed inside `_load_round` so the round-start
    inventory already reflects whatever was handed out). Pinned at
    event-construction time in `room.py` so `await` boundaries can't
    race the next reload into the rendered string.
    """
    if state is None or human_name is None or human_id is None:
        return "🔄 Shotgun reloaded — new round"
    ai_id = 1 - human_id
    live = int(state.round_initial_live)
    blank = int(state.round_initial_blank)
    return (
        f"{_SEPARATOR}\n"
        f"🔄  NEW ROUND\n"
        f"Shells: 💥×{live}  🫧×{blank}\n"
        f"\n"
        f"{human_name}: {_inventory_block(state, human_id)}\n"
        f"🤖 AI: {_inventory_block(state, ai_id)}\n"
        f"{_SEPARATOR}"
    )


def game_over_message(state: GameState, human_name: str, human_id: int) -> str:
    if state.winner == human_id:
        return f"💼 {human_name} defeated the AI — gg!"
    return "⚰️ The AI won. Try /ai to play again."
