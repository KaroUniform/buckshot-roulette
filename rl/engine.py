"""Pure-Python Buckshot Roulette engine for RL training.

Heads-up (1v1) variant covering all 9 official Story / Double-or-Nothing items.
Deterministic given a seed; no I/O; cheap to clone. Designed to be wrapped
in PettingZoo / OpenSpiel envs without further refactoring.

Action space (discrete, 19 actions):
    0  SHOOT_OPPONENT
    1  SHOOT_SELF
    2..10  USE_<ITEM>          (one per item, in Item enum order)
    11..18 ADRENALINE_PICK_<X> (8 actions; cannot pick ADRENALINE itself)

Adrenaline is modeled as a two-step action: USE_ADRENALINE puts the engine
into a pick-mode where the only legal actions are ADRENALINE_PICK_<X>, which
consume the opponent's item X and apply its effect on behalf of the active
player. This matches the in-game mechanic without any wall-clock dependency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Item(IntEnum):
    HANDSAW = 0
    BEER = 1
    SMOKE = 2
    HANDCUFF = 3
    GLASS = 4
    PHONE = 5
    PILLS = 6
    ADRENALINE = 7
    INVERTER = 8


NUM_ITEMS = len(Item)
MAX_INVENTORY = 8

_NON_ADRENALINE_ITEMS = tuple(i for i in Item if i != Item.ADRENALINE)


class Action(IntEnum):
    SHOOT_OPPONENT = 0
    SHOOT_SELF = 1
    USE_HANDSAW = 2
    USE_BEER = 3
    USE_SMOKE = 4
    USE_HANDCUFF = 5
    USE_GLASS = 6
    USE_PHONE = 7
    USE_PILLS = 8
    USE_ADRENALINE = 9
    USE_INVERTER = 10
    PICK_HANDSAW = 11
    PICK_BEER = 12
    PICK_SMOKE = 13
    PICK_HANDCUFF = 14
    PICK_GLASS = 15
    PICK_PHONE = 16
    PICK_PILLS = 17
    PICK_INVERTER = 18


NUM_ACTIONS = len(Action)

_USE_ACTION_FOR_ITEM = {
    Item.HANDSAW: Action.USE_HANDSAW,
    Item.BEER: Action.USE_BEER,
    Item.SMOKE: Action.USE_SMOKE,
    Item.HANDCUFF: Action.USE_HANDCUFF,
    Item.GLASS: Action.USE_GLASS,
    Item.PHONE: Action.USE_PHONE,
    Item.PILLS: Action.USE_PILLS,
    Item.ADRENALINE: Action.USE_ADRENALINE,
    Item.INVERTER: Action.USE_INVERTER,
}

_PICK_ACTION_FOR_ITEM = {
    Item.HANDSAW: Action.PICK_HANDSAW,
    Item.BEER: Action.PICK_BEER,
    Item.SMOKE: Action.PICK_SMOKE,
    Item.HANDCUFF: Action.PICK_HANDCUFF,
    Item.GLASS: Action.PICK_GLASS,
    Item.PHONE: Action.PICK_PHONE,
    Item.PILLS: Action.PICK_PILLS,
    Item.INVERTER: Action.PICK_INVERTER,
}

_ITEM_FOR_PICK_ACTION = {a: i for i, a in _PICK_ACTION_FOR_ITEM.items()}


@dataclass
class PlayerState:
    hp: int
    max_hp: int
    inventory: np.ndarray  # int counts, shape (NUM_ITEMS,)
    skip_next_turn: bool = False  # set by opponent's handcuffs

    def n_items(self) -> int:
        return int(self.inventory.sum())

    def has(self, item: Item) -> bool:
        return self.inventory[int(item)] > 0


@dataclass
class GameState:
    players: list  # [PlayerState, PlayerState]
    shells: list  # ordered; shells[0] is the NEXT to be fired
    damage_mult: int  # applied to next shot, reset to 1 after firing
    current_player: int  # 0 or 1
    non_adrenaline_used_this_turn: bool
    adrenaline_active: bool  # current_player has used adrenaline and must pick
    # Per-player knowledge: maps shell-index (0 = next) to True/False (live/blank)
    known_shells: list  # [dict, dict]
    done: bool = False
    winner: Optional[int] = None


class BuckshotEngine:
    """Stateful, seedable engine. One instance == one ongoing match."""

    def __init__(
        self,
        seed: Optional[int] = None,
        hp_range: tuple = (2, 4),
        shells_range: tuple = (2, 8),
        items_per_round_choices: tuple = (1, 1, 1, 1, 2, 2, 2, 3, 3, 4),
    ) -> None:
        self.hp_range = hp_range
        self.shells_range = shells_range
        self.items_per_round_choices = items_per_round_choices
        self.rng = np.random.default_rng(seed)
        self.state: Optional[GameState] = None

    # ---- lifecycle ----

    def reset(self, seed: Optional[int] = None) -> GameState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        hp = int(self.rng.integers(self.hp_range[0], self.hp_range[1] + 1))
        players = [
            PlayerState(hp=hp, max_hp=hp, inventory=np.zeros(NUM_ITEMS, dtype=np.int32)),
            PlayerState(hp=hp, max_hp=hp, inventory=np.zeros(NUM_ITEMS, dtype=np.int32)),
        ]
        self.state = GameState(
            players=players,
            shells=[],
            damage_mult=1,
            current_player=int(self.rng.integers(0, 2)),
            non_adrenaline_used_this_turn=False,
            adrenaline_active=False,
            known_shells=[{}, {}],
        )
        self._load_round()
        return self.state

    def clone(self) -> "BuckshotEngine":
        new = BuckshotEngine.__new__(BuckshotEngine)
        new.hp_range = self.hp_range
        new.shells_range = self.shells_range
        new.items_per_round_choices = self.items_per_round_choices
        # Deep-copy the entire Generator (including PCG64's `inc` field).
        # Re-seeding from `bit_generator.state["state"]["state"]` would route
        # the value through SeedSequence and produce a different stream.
        new.rng = copy.deepcopy(self.rng)
        new.state = copy.deepcopy(self.state)
        return new

    # ---- core step ----

    def step(self, action: int) -> tuple[GameState, float, bool, dict]:
        if self.state is None or self.state.done:
            raise RuntimeError("Call reset() before step(), and don't step a finished game.")

        if not self.legal_actions()[action]:
            raise ValueError(f"Illegal action {Action(action).name} in current state.")

        info: dict = {}
        action = Action(action)

        if action == Action.SHOOT_OPPONENT:
            self._shoot(target=1 - self.state.current_player, info=info)
            self._end_turn_after_shot(self_target=False, info=info)
        elif action == Action.SHOOT_SELF:
            was_blank = self._shoot(target=self.state.current_player, info=info)
            self._end_turn_after_shot(self_target=True, was_blank=was_blank, info=info)
        elif Action.USE_HANDSAW <= action <= Action.USE_INVERTER:
            item = _action_to_item(action)
            self._apply_item(item, from_opponent_inventory=False, info=info)
        elif Action.PICK_HANDSAW <= action <= Action.PICK_INVERTER:
            item = _ITEM_FOR_PICK_ACTION[action]
            self._apply_item(item, from_opponent_inventory=True, info=info)

        reward = 0.0
        if self.state.done:
            reward = 1.0 if self.state.winner == self.state.current_player else -1.0
            # Note: reward sign is from current player's perspective at the moment
            # of termination. PettingZoo wrapper will translate per-agent.

        return self.state, reward, self.state.done, info

    # ---- legality ----

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self.state is None or self.state.done:
            return mask
        s = self.state
        me = s.players[s.current_player]
        opp = s.players[1 - s.current_player]

        if s.adrenaline_active:
            # Only pick actions, restricted to items the opponent actually has
            # AND items that respect the "1 non-adrenaline per turn" rule.
            # The trigger USE_ADRENALINE itself sets non_adrenaline_used to False,
            # so picking is always allowed by that rule (we check inventory only).
            for item in _NON_ADRENALINE_ITEMS:
                if opp.inventory[int(item)] > 0 and self._item_usable(item, target=opp):
                    mask[int(_PICK_ACTION_FOR_ITEM[item])] = True
            # Also allowed: do nothing? Original game forces a pick or wastes
            # adrenaline by timeout. We'll require a pick whenever any is legal.
            # If NO item is usable we silently drop adrenaline by allowing both
            # shoot actions as a fallback.
            if not mask.any():
                mask[int(Action.SHOOT_OPPONENT)] = s.shells != []
                mask[int(Action.SHOOT_SELF)] = s.shells != []
            return mask

        # Normal turn: shoot is always legal if shotgun has shells
        if s.shells:
            mask[int(Action.SHOOT_OPPONENT)] = True
            mask[int(Action.SHOOT_SELF)] = True

        # Item uses
        if not s.non_adrenaline_used_this_turn:
            for item in _NON_ADRENALINE_ITEMS:
                if me.inventory[int(item)] > 0 and self._item_usable(item, target=opp):
                    mask[int(_USE_ACTION_FOR_ITEM[item])] = True

        # Adrenaline can only be triggered if not already in pick mode and opponent
        # has at least one stealable+usable item.
        if (
            me.inventory[int(Item.ADRENALINE)] > 0
            and not s.non_adrenaline_used_this_turn
        ):
            opp_has_stealable = any(
                opp.inventory[int(it)] > 0 and self._item_usable(it, target=opp)
                for it in _NON_ADRENALINE_ITEMS
            )
            if opp_has_stealable:
                mask[int(Action.USE_ADRENALINE)] = True

        return mask

    def _item_usable(self, item: Item, target: PlayerState) -> bool:
        """Per-item legality predicate (beyond inventory presence)."""
        s = self.state
        me = s.players[s.current_player]
        if item in (Item.HANDSAW, Item.BEER, Item.GLASS, Item.PHONE, Item.INVERTER):
            return bool(s.shells)
        if item == Item.SMOKE:
            return me.hp < me.max_hp
        if item == Item.HANDCUFF:
            return not target.skip_next_turn
        if item == Item.PILLS:
            return True  # always — risk is part of the item
        return True

    # ---- observation ----

    def observation(self, player_id: int) -> np.ndarray:
        """Per-player observation tensor.

        Hides:
          - The exact shell order (only n_live, n_blank, total are visible)
          - The opponent's per-shell knowledge

        Reveals:
          - Both HPs and inventories (perfect-information style for these)
          - Own per-shell knowledge (positions where this player learned via items)
          - Whose turn it is, current damage multiplier, adrenaline state
        """
        s = self.state
        me = s.players[player_id]
        opp = s.players[1 - player_id]
        n = len(s.shells)
        n_live = sum(1 for live in s.shells if live)
        n_blank = n - n_live

        max_shells = self.shells_range[1]
        # Per-shell knowledge: (known_live, known_blank) pair per shell slot.
        # Shells beyond current count are zeroed.
        known_live = np.zeros(max_shells, dtype=np.float32)
        known_blank = np.zeros(max_shells, dtype=np.float32)
        for pos, is_live in s.known_shells[player_id].items():
            if 0 <= pos < max_shells:
                if is_live:
                    known_live[pos] = 1.0
                else:
                    known_blank[pos] = 1.0

        parts = [
            np.array(
                [
                    me.hp,
                    me.max_hp,
                    opp.hp,
                    opp.max_hp,
                    n,
                    n_live,
                    n_blank,
                    s.damage_mult,
                    int(s.current_player == player_id),
                    int(opp.skip_next_turn),
                    int(me.skip_next_turn),
                    int(s.adrenaline_active),
                    int(s.non_adrenaline_used_this_turn),
                ],
                dtype=np.float32,
            ),
            me.inventory.astype(np.float32),
            opp.inventory.astype(np.float32),
            known_live,
            known_blank,
        ]
        return np.concatenate(parts)

    # ---- internal mechanics ----

    def _load_round(self) -> None:
        s = self.state
        n = int(self.rng.integers(self.shells_range[0], self.shells_range[1] + 1))
        # ~50/50 with mild jitter, at least 1 of each
        n_live = max(1, min(n - 1, int(round(n / 2 + self.rng.uniform(-0.5, 0.5) * n / 4))))
        n_blank = n - n_live
        shells = [True] * n_live + [False] * n_blank
        self.rng.shuffle(shells)
        s.shells = shells
        s.damage_mult = 1
        s.known_shells = [{}, {}]
        # Reset cuffs at recharge (matches original game)
        s.players[0].skip_next_turn = False
        s.players[1].skip_next_turn = False
        # Distribute items
        n_items = int(self.rng.choice(self.items_per_round_choices))
        for p in s.players:
            self._distribute_items(p, n_items)

    def _distribute_items(self, player: PlayerState, count: int) -> None:
        for _ in range(count):
            if player.n_items() >= MAX_INVENTORY:
                return
            item_idx = int(self.rng.integers(0, NUM_ITEMS))
            player.inventory[item_idx] += 1

    def _shoot(self, target: int, info: dict) -> bool:
        """Fire next shell at target. Returns True if blank (target undamaged)."""
        s = self.state
        live = s.shells.pop(0)
        # Shift known-shells indices: position 0 is gone, all others shift down.
        for k in range(2):
            shifted = {}
            for pos, val in s.known_shells[k].items():
                if pos == 0:
                    continue
                shifted[pos - 1] = val
            s.known_shells[k] = shifted

        if live:
            s.players[target].hp -= s.damage_mult
            info["shot"] = ("live", target, s.damage_mult)
        else:
            info["shot"] = ("blank", target, 0)
        s.damage_mult = 1
        return not live

    def _end_turn_after_shot(
        self,
        self_target: bool,
        was_blank: bool = False,
        info: Optional[dict] = None,
    ) -> None:
        s = self.state
        # Check death
        for pid, p in enumerate(s.players):
            if p.hp <= 0:
                s.done = True
                s.winner = 1 - pid
                return
        # The blank-self-shot rule ALWAYS preserves the turn — including when
        # the blank was the last shell. Decide `keep` first, reload if needed,
        # then advance with the right flag.
        keep = self_target and was_blank
        if not s.shells:
            self._load_round()
        self._advance_turn(keep=keep)

    def _advance_turn(self, keep: bool) -> None:
        s = self.state
        s.non_adrenaline_used_this_turn = False
        s.adrenaline_active = False
        if keep:
            return
        next_player = 1 - s.current_player
        if s.players[next_player].skip_next_turn:
            s.players[next_player].skip_next_turn = False
            # Skipped — current player gets to go again
            return
        s.current_player = next_player

    def _apply_item(self, item: Item, from_opponent_inventory: bool, info: dict) -> None:
        s = self.state
        me = s.players[s.current_player]
        opp = s.players[1 - s.current_player]
        source_inv = opp.inventory if from_opponent_inventory else me.inventory
        source_inv[int(item)] -= 1

        if item == Item.HANDSAW:
            s.damage_mult *= 2
        elif item == Item.BEER:
            ejected_live = s.shells.pop(0)
            for k in range(2):
                shifted = {}
                for pos, val in s.known_shells[k].items():
                    if pos == 0:
                        continue
                    shifted[pos - 1] = val
                s.known_shells[k] = shifted
            info["beer_ejected"] = "live" if ejected_live else "blank"
            if not s.shells:
                self._load_round()
                # Beer that empties the chamber doesn't end the turn in the
                # original; current player keeps acting in the new round.
        elif item == Item.SMOKE:
            me.hp = min(me.hp + 1, me.max_hp)
        elif item == Item.HANDCUFF:
            opp.skip_next_turn = True
        elif item == Item.GLASS:
            # Reveal the next shell to current player only
            s.known_shells[s.current_player][0] = s.shells[0]
            info["glass"] = "live" if s.shells[0] else "blank"
        elif item == Item.PHONE:
            # Reveal a uniformly-random shell position (excluding pos 0 if multiple
            # remain — matches Burner Phone wiki: "tells you about one of the
            # shells", typically not the next one to maintain value). Simpler:
            # uniform over all positions including the next.
            pos = int(self.rng.integers(0, len(s.shells)))
            s.known_shells[s.current_player][pos] = s.shells[pos]
            info["phone"] = (pos, "live" if s.shells[pos] else "blank")
        elif item == Item.PILLS:
            # 40% +2 HP (capped), 60% -1 HP (could kill)
            if self.rng.random() < 0.4:
                me.hp = min(me.hp + 2, me.max_hp)
                info["pills"] = "good"
            else:
                me.hp -= 1
                info["pills"] = "bad"
                if me.hp <= 0:
                    s.done = True
                    s.winner = 1 - s.current_player
                    return
        elif item == Item.ADRENALINE:
            s.adrenaline_active = True
            return  # do not flip non_adrenaline_used; do not end turn
        elif item == Item.INVERTER:
            # Flip the next shell only (matches wiki: "swaps the polarity of the
            # current shell in the chamber").
            s.shells[0] = not s.shells[0]
            # Any prior knowledge of position 0 is now stale — flip it too
            for k in range(2):
                if 0 in s.known_shells[k]:
                    s.known_shells[k][0] = not s.known_shells[k][0]

        # Mark non-adrenaline use; clear adrenaline_active (a pick consumes it)
        s.non_adrenaline_used_this_turn = True
        s.adrenaline_active = False


def _action_to_item(action: Action) -> Item:
    """Map a USE_<ITEM> action back to its Item."""
    return {
        Action.USE_HANDSAW: Item.HANDSAW,
        Action.USE_BEER: Item.BEER,
        Action.USE_SMOKE: Item.SMOKE,
        Action.USE_HANDCUFF: Item.HANDCUFF,
        Action.USE_GLASS: Item.GLASS,
        Action.USE_PHONE: Item.PHONE,
        Action.USE_PILLS: Item.PILLS,
        Action.USE_ADRENALINE: Item.ADRENALINE,
        Action.USE_INVERTER: Item.INVERTER,
    }[action]
