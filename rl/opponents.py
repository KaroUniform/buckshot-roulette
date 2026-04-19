"""Opponent policies for self-play training and evaluation.

An Opponent is anything implementing `act(obs, mask, rng) -> int`:
  - obs: float32 observation tensor (numpy)
  - mask: int8 action mask (numpy), 1 where legal
  - rng: numpy Generator for stochasticity

These are intentionally thin so they're cheap to call inside the env loop and
trivial to swap (random / rule-based / frozen neural net all conform).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from rl.engine import Action, NUM_ITEMS

OpponentFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


# ---- Layout helpers (tied to engine.observation()) ----
# Scalar slots in the obs vector — keep in sync with BuckshotEngine.observation()
_O_MY_HP = 0
_O_MY_MAX_HP = 1
_O_OPP_HP = 2
_O_OPP_MAX_HP = 3
_O_N_SHELLS = 4
_O_N_LIVE = 5
_O_N_BLANK = 6
_O_DAMAGE_MULT = 7
_O_IS_MY_TURN = 8
_O_OPP_CUFFED = 9
_O_MY_CUFFED = 10
_O_ADRENALINE_ACTIVE = 11
_O_USED_NON_ADRENALINE = 12

_INV_OFFSET = 13
_OPP_INV_OFFSET = _INV_OFFSET + NUM_ITEMS  # 22


def random_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def aggressive_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Always shoot opponent; prefer handsaw before shooting; otherwise shoot.
    No info-gathering, no self-shots. A "rush" baseline."""
    if mask[int(Action.USE_HANDSAW)] and obs[_O_DAMAGE_MULT] < 2:
        return int(Action.USE_HANDSAW)
    if mask[int(Action.SHOOT_OPPONENT)]:
        return int(Action.SHOOT_OPPONENT)
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def conservative_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Naive EV-style heuristic. Uses info items when available; shoots self
    if blanks > lives, opponent otherwise. Heals when low. No bluff modeling.

    This mirrors the popular wisdom from the Buckshot Roulette community,
    so the trained agent should be able to *exploit* it if it learns nuance.
    """
    n_live = obs[_O_N_LIVE]
    n_blank = obs[_O_N_BLANK]
    my_hp = obs[_O_MY_HP]
    my_max = obs[_O_MY_MAX_HP]

    # Heal first if hurt and have a smoke
    if my_hp < my_max and mask[int(Action.USE_SMOKE)]:
        return int(Action.USE_SMOKE)

    # Gather info if uncertain
    if n_live > 0 and n_blank > 0:
        if mask[int(Action.USE_GLASS)]:
            return int(Action.USE_GLASS)
        if mask[int(Action.USE_PHONE)]:
            return int(Action.USE_PHONE)

    # Cuff opponent before lethal shot
    if mask[int(Action.USE_HANDCUFF)]:
        return int(Action.USE_HANDCUFF)

    # Saw before opponent shot
    if mask[int(Action.USE_HANDSAW)] and obs[_O_DAMAGE_MULT] < 2:
        return int(Action.USE_HANDSAW)

    # Shoot decision
    if n_blank > n_live and mask[int(Action.SHOOT_SELF)]:
        return int(Action.SHOOT_SELF)
    if mask[int(Action.SHOOT_OPPONENT)]:
        return int(Action.SHOOT_OPPONENT)

    # Fallback
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


# Registry for easy lookup
NAMED_OPPONENTS: dict[str, OpponentFn] = {
    "random": random_opponent,
    "aggressive": aggressive_opponent,
    "conservative": conservative_opponent,
}


def make_frozen_policy_opponent(policy, device: str = "cpu") -> OpponentFn:
    """Wraps a torch policy network into an opponent function.

    The policy must expose `.act(obs_tensor, mask_tensor) -> action: int`.
    A fresh forward pass is run per call — fine for training collectors but
    avoid calling on huge batches; vectorize at the env level instead.
    """
    import torch

    def fn(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)
            action = policy.act(obs_t, mask_t)
        return int(action.item())

    return fn


class OpponentPool:
    """Sampling pool for self-play league.

    Holds a set of named opponents (functions). `sample()` returns one
    according to weights — used by the env at reset time to pick who the
    agent plays this episode. Snapshots of the training policy can be added
    via `add_snapshot(name, opponent_fn)`.
    """

    def __init__(self, base_opponents: Optional[dict[str, OpponentFn]] = None) -> None:
        self.opponents: dict[str, OpponentFn] = dict(base_opponents or {})
        self.weights: dict[str, float] = {name: 1.0 for name in self.opponents}

    def add(self, name: str, fn: OpponentFn, weight: float = 1.0) -> None:
        self.opponents[name] = fn
        self.weights[name] = weight

    def sample(self, rng: np.random.Generator) -> tuple[str, OpponentFn]:
        names = list(self.opponents.keys())
        if not names:
            raise ValueError("OpponentPool is empty")
        ws = np.array([self.weights[n] for n in names], dtype=np.float64)
        ws /= ws.sum()
        idx = int(rng.choice(len(names), p=ws))
        name = names[idx]
        return name, self.opponents[name]

    def __len__(self) -> int:
        return len(self.opponents)
