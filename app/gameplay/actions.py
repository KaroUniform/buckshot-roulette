"""Shared emoji ↔ rl.engine.Action translation for Telegram game flows."""

from __future__ import annotations

from rl.engine import Action, Item


_ITEM_EMOJI: dict[Item, str] = {
    Item.HANDSAW: "🪚",
    Item.BEER: "🍺",
    Item.SMOKE: "🚬",
    Item.HANDCUFF: "🔗",
    Item.GLASS: "🔍",
    Item.ADRENALINE: "💉",
    Item.INVERTER: "🔀",
    Item.PHONE: "📞",
    Item.PILLS: "💊",
}

SHOOT_OPP_GLYPH = "🔼"
SHOOT_SELF_GLYPH = "🔽"


def item_emoji(item: Item) -> str:
    return _ITEM_EMOJI[item]


_USE_ACTION_BY_EMOJI: dict[str, Action] = {
    SHOOT_OPP_GLYPH: Action.SHOOT_OPPONENT,
    SHOOT_SELF_GLYPH: Action.SHOOT_SELF,
    _ITEM_EMOJI[Item.HANDSAW]: Action.USE_HANDSAW,
    _ITEM_EMOJI[Item.BEER]: Action.USE_BEER,
    _ITEM_EMOJI[Item.SMOKE]: Action.USE_SMOKE,
    _ITEM_EMOJI[Item.HANDCUFF]: Action.USE_HANDCUFF,
    _ITEM_EMOJI[Item.GLASS]: Action.USE_GLASS,
    _ITEM_EMOJI[Item.PHONE]: Action.USE_PHONE,
    _ITEM_EMOJI[Item.PILLS]: Action.USE_PILLS,
    _ITEM_EMOJI[Item.ADRENALINE]: Action.USE_ADRENALINE,
    _ITEM_EMOJI[Item.INVERTER]: Action.USE_INVERTER,
}

_PICK_PREFIX = _ITEM_EMOJI[Item.ADRENALINE]

_PICK_ACTION_BY_ITEM: dict[Item, Action] = {
    Item.HANDSAW: Action.PICK_HANDSAW,
    Item.BEER: Action.PICK_BEER,
    Item.SMOKE: Action.PICK_SMOKE,
    Item.HANDCUFF: Action.PICK_HANDCUFF,
    Item.GLASS: Action.PICK_GLASS,
    Item.PHONE: Action.PICK_PHONE,
    Item.PILLS: Action.PICK_PILLS,
    Item.INVERTER: Action.PICK_INVERTER,
}


def pick_emoji(item: Item) -> str:
    return f"{_PICK_PREFIX}{_ITEM_EMOJI[item]}"


def emoji_to_action(text: str, adrenaline_active: bool) -> Action | None:
    if not text:
        return None

    if adrenaline_active:
        if text.startswith(SHOOT_OPP_GLYPH):
            return Action.SHOOT_OPPONENT
        if text.startswith(SHOOT_SELF_GLYPH):
            return Action.SHOOT_SELF
        if not text.startswith(_PICK_PREFIX):
            return None
        tail = text[len(_PICK_PREFIX):]
        for item, glyph in _ITEM_EMOJI.items():
            if item == Item.ADRENALINE:
                continue
            if tail.startswith(glyph):
                return _PICK_ACTION_BY_ITEM[item]
        return None

    for token in sorted(_USE_ACTION_BY_EMOJI.keys(), key=len, reverse=True):
        if text.startswith(token):
            return _USE_ACTION_BY_EMOJI[token]
    return None

