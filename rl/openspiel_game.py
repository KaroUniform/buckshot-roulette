"""OpenSpiel game wrapper for *simplified* Buckshot Roulette.

The full game (HP up to 4, 9 items, variable shell counts, multiple random
choices per turn) has too large an info-state space for tabular CFR. We
therefore expose a stripped-down version that captures the core mechanic
(known shell counts, hidden order, shoot-self-or-opponent dynamics) where
CFR converges in seconds and gives us a true Nash baseline to validate
PPO against.

Simplifications:
- Fixed HP (default 2) per player.
- Fixed shell composition per round (default 2 live + 2 blank).
- No items.
- Round reload happens automatically when shells run out (rare with HP=2).
- Chance node at the start of each round samples a uniformly-random shell
  permutation. Each permutation has equal probability 1/C(L+B, L).

This is enough to test whether our learning loop reaches Nash on a small
imperfect-info game; an agent that's near-optimal here gives us
*calibration* for trusting the larger-scale runs.

Run:
    python -m rl.openspiel_game            # quick self-check via Random play
    python -m rl.cfr_baseline              # see rl/cfr_baseline.py
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np
import pyspiel


_SHOOT_OPPONENT = 0
_SHOOT_SELF = 1
_NUM_PLAYER_ACTIONS = 2

_DEFAULT_PARAMS = {"hp": 2, "n_live": 2, "n_blank": 2}


def _shell_perms(n_live: int, n_blank: int) -> list[tuple[bool, ...]]:
    """All distinct multiset permutations of `n_live` live + `n_blank` blank
    shells, listed in a deterministic order suitable for chance-action ids.
    Index 0 in each tuple is the next shot."""
    n = n_live + n_blank
    perms = []
    for live_positions in combinations(range(n), n_live):
        shells = [False] * n
        for pos in live_positions:
            shells[pos] = True
        perms.append(tuple(shells))
    return perms


_GAME_TYPE = pyspiel.GameType(
    short_name="simple_buckshot",
    long_name="Simplified Buckshot Roulette (heads-up, no items)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "hp": 2,
        "n_live": 2,
        "n_blank": 2,
    },
)


class SimpleBuckshotState(pyspiel.State):
    """State for simplified Buckshot Roulette."""

    def __init__(self, game) -> None:
        super().__init__(game)
        # Read params via game.get_parameters() so we work whether pyspiel
        # passes us our own SimpleBuckshotGame instance (which has the
        # attributes set in __init__) or a freshly-reconstructed one from
        # its registry (which only has the params dict).
        params = dict(game.get_parameters() or {})
        self._hp_max = int(params.get("hp", _DEFAULT_PARAMS["hp"]))
        self._n_live_total = int(params.get("n_live", _DEFAULT_PARAMS["n_live"]))
        self._n_blank_total = int(params.get("n_blank", _DEFAULT_PARAMS["n_blank"]))
        self._hps = [self._hp_max, self._hp_max]
        self._shells: list[bool] = []  # index 0 = next shot
        self._cur_player = pyspiel.PlayerId.CHANCE  # round 1 begins with shell deal
        self._winner: Optional[int] = None
        self._n_round_start_chance_actions = math.comb(
            self._n_live_total + self._n_blank_total, self._n_live_total
        )
        self._perms = _shell_perms(self._n_live_total, self._n_blank_total)

    # ---- pyspiel.State required methods ----

    def current_player(self) -> int:
        if self._winner is not None:
            return pyspiel.PlayerId.TERMINAL
        return self._cur_player

    def is_terminal(self) -> bool:
        return self._winner is not None

    def is_chance_node(self) -> bool:
        return (not self.is_terminal()) and self._cur_player == pyspiel.PlayerId.CHANCE

    def chance_outcomes(self) -> list[tuple[int, float]]:
        n = self._n_round_start_chance_actions
        p = 1.0 / n
        return [(i, p) for i in range(n)]

    def _legal_actions(self, player: int) -> list[int]:
        if self.is_terminal():
            return []
        if self.is_chance_node():
            return list(range(self._n_round_start_chance_actions))
        return [_SHOOT_OPPONENT, _SHOOT_SELF]

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            self._shells = list(self._perms[action])
            # Player 0 always opens the round in this simplified version.
            # (We could randomize via a second chance node, but it doesn't
            # affect Nash strategy — both seats see the same info structure.)
            self._cur_player = 0
            return

        actor = self._cur_player
        target = (1 - actor) if action == _SHOOT_OPPONENT else actor
        live = self._shells.pop(0)
        if live:
            self._hps[target] -= 1
            if self._hps[target] <= 0:
                self._winner = 1 - target
                return

        # Turn flow: blank self-shot keeps the turn; otherwise pass.
        if action == _SHOOT_SELF and not live:
            pass  # keep turn
        else:
            self._cur_player = 1 - actor

        # Reload if shells exhausted
        if not self._shells:
            self._cur_player = pyspiel.PlayerId.CHANCE

    def returns(self) -> list[float]:
        if self._winner is None:
            return [0.0, 0.0]
        return [1.0 if pid == self._winner else -1.0 for pid in range(2)]

    # ---- Information state / observation ----
    # An info state captures everything a player has seen that's relevant.
    # In this simplified game (no items, no info-revealing actions), both
    # players see the same public information: (hps, shells_remaining,
    # whose-turn). The hidden info is the shell ORDER. So info_state ==
    # observation here. CFR will treat siblings under chance as the same
    # info state, which is what we want.

    def information_state_string(self, player: int = 0) -> str:
        # In this simplified, no-items game ALL info is public except the
        # exact shell order (which neither player sees). The info state is
        # therefore the same for both players. We still take a `player`
        # argument to satisfy the API but ignore it — encoding it would
        # double the table size and break TabularPolicy lookups since
        # OpenSpiel keys policy entries by the acting player only.
        del player
        if self.is_chance_node():
            return f"chance:hps={self._hps},to_deal={self._n_live_total + self._n_blank_total}"
        n = len(self._shells)
        n_live = sum(1 for x in self._shells if x)
        n_blank = n - n_live
        return (
            f"hps={self._hps[0]},{self._hps[1]}|"
            f"shells={n}({n_live}L,{n_blank}B)|turn={self._cur_player}"
        )

    def observation_string(self, player: int = 0) -> str:
        return self.information_state_string(player)

    # Tensor encodings — flat vectors. Shape is identical for all states.
    def _state_tensor(self, player: int) -> np.ndarray:
        # [hp_me, hp_opp, hp_max, n_shells, n_live_remain, n_blank_remain, my_turn]
        n = len(self._shells)
        n_live = sum(1 for x in self._shells if x)
        n_blank = n - n_live
        my_turn = 1.0 if self._cur_player == player else 0.0
        return np.array(
            [
                self._hps[player],
                self._hps[1 - player],
                self._hp_max,
                n,
                n_live,
                n_blank,
                my_turn,
            ],
            dtype=np.float32,
        )

    def information_state_tensor(self, player: int = 0) -> list[float]:
        return self._state_tensor(player).tolist()

    def observation_tensor(self, player: int = 0) -> list[float]:
        return self._state_tensor(player).tolist()

    def __str__(self) -> str:
        live = sum(1 for x in self._shells if x)
        blank = len(self._shells) - live
        return (
            f"SimpleBuckshot[hp={self._hps},shells={len(self._shells)}"
            f"({live}L,{blank}B),turn={self._cur_player},winner={self._winner}]"
        )

    def clone(self) -> "SimpleBuckshotState":
        new = SimpleBuckshotState.__new__(SimpleBuckshotState)
        pyspiel.State.__init__(new, self.get_game())
        new._hp_max = self._hp_max
        new._n_live_total = self._n_live_total
        new._n_blank_total = self._n_blank_total
        new._hps = list(self._hps)
        new._shells = list(self._shells)
        new._cur_player = self._cur_player
        new._winner = self._winner
        new._n_round_start_chance_actions = self._n_round_start_chance_actions
        new._perms = self._perms
        return new


class SimpleBuckshotGame(pyspiel.Game):
    """Simplified Buckshot Roulette as an OpenSpiel game."""

    def __init__(self, params: Optional[dict] = None) -> None:
        params = dict(_DEFAULT_PARAMS) | (params or {})
        self.hp = int(params["hp"])
        self.n_live = int(params["n_live"])
        self.n_blank = int(params["n_blank"])
        n_chance = math.comb(self.n_live + self.n_blank, self.n_live)
        # num_distinct_actions covers BOTH chance and player actions; take max
        n_actions = max(_NUM_PLAYER_ACTIONS, n_chance)
        info = pyspiel.GameInfo(
            num_distinct_actions=n_actions,
            max_chance_outcomes=n_chance,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            # Generous upper bound: HP rounds * shells/round * 4 (worst case)
            max_game_length=200,
        )
        super().__init__(_GAME_TYPE, info, params)

    def new_initial_state(self) -> SimpleBuckshotState:
        return SimpleBuckshotState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        # Return None — we expose tensors directly via the State class.
        return None


# Register so pyspiel.load_game("simple_buckshot") works
pyspiel.register_game(_GAME_TYPE, SimpleBuckshotGame)


def _self_check() -> None:
    """Quick self-check: load the game, play a few random episodes."""
    game = pyspiel.load_game("simple_buckshot")
    rng = np.random.default_rng(0)
    wins = [0, 0]
    for _ in range(200):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                a = int(rng.choice(actions, p=probs))
            else:
                a = int(rng.choice(state.legal_actions()))
            state.apply_action(a)
        ret = state.returns()
        if ret[0] > 0:
            wins[0] += 1
        else:
            wins[1] += 1
    print(f"random vs random over 200 games: {wins[0]} / {wins[1]}")
    assert 70 < wins[0] < 130, f"Suspicious imbalance: {wins}"
    print("ok  simple_buckshot self-check")


if __name__ == "__main__":
    _self_check()
