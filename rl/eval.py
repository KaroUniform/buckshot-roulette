"""Evaluate a policy against a set of named opponents.

Returns per-opponent win rate over `n_episodes` games. Uses fresh seeds and
a fresh env per episode for clean independence (cheap given our small game).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from rl.opponents import NAMED_OPPONENTS, OpponentFn, OpponentPool
from rl.single_agent_env import (
    OPPONENT_ID_OTHER,
    SingleAgentBuckshotEnv,
)


def evaluate_policy(
    policy,
    opponents: Optional[dict[str, OpponentFn]] = None,
    n_episodes: int = 200,
    seed: int = 12345,
    device: str = "cpu",
    honest_obs: bool = False,
) -> dict[str, float]:
    """For each named opponent, return the policy's win rate.

    `policy` must implement `.act(obs_tensor, mask_tensor) -> action_int_tensor`.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    opponents = opponents or NAMED_OPPONENTS
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}

    for name, fn in opponents.items():
        pool = OpponentPool({name: fn})
        env = SingleAgentBuckshotEnv(opponent_pool=pool, honest_obs=honest_obs)
        wins = 0
        for ep in range(n_episodes):
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
            terminal_reward = info.get("_terminal_reward", None)
            if info.get("_terminated_in_reset"):
                if terminal_reward is not None and terminal_reward > 0:
                    wins += 1
                continue
            done = False
            while not done:
                obs_t = torch.from_numpy(obs["observation"]).to(device).unsqueeze(0)
                mask_t = torch.from_numpy(obs["action_mask"]).to(device).unsqueeze(0)
                action = policy.act(obs_t, mask_t).item()
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                if done and reward > 0:
                    wins += 1
        results[name] = wins / n_episodes

    return results


def evaluate_recurrent_policy(
    policy,
    opponents: Optional[dict[str, OpponentFn]] = None,
    n_episodes: int = 200,
    seed: int = 12345,
    device: str = "cpu",
    honest_obs: bool = False,
) -> dict[str, float]:
    """Like evaluate_policy but for RecurrentActorCritic.

    Maintains per-episode hidden state (reset on each env reset). For
    policies trained with the E19 opponent embedding (n_opponents > 0),
    threads the env's `opponent_id` info into `act_stateful(opp_ids=...)`
    so the embedding contributes during eval the same way it did during
    training. Policies without the embedding ignore opp_ids transparently.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    opponents = opponents or NAMED_OPPONENTS
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}

    has_embed = getattr(policy, "n_opponents", 0) > 0

    for name, fn in opponents.items():
        pool = OpponentPool({name: fn})
        env = SingleAgentBuckshotEnv(opponent_pool=pool, honest_obs=honest_obs)
        wins = 0
        for ep in range(n_episodes):
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
            terminal_reward = info.get("_terminal_reward", None)
            if info.get("_terminated_in_reset"):
                if terminal_reward is not None and terminal_reward > 0:
                    wins += 1
                continue
            h = policy.initial_hidden(1, device=device)
            done = False
            # opponent_id stays constant across an episode (one opponent
            # per episode), so we can capture once from reset's info.
            opp_id = int(info.get("opponent_id", OPPONENT_ID_OTHER))
            opp_ids_t = (
                torch.tensor([opp_id], dtype=torch.long, device=device)
                if has_embed else None
            )
            while not done:
                obs_t = torch.from_numpy(obs["observation"]).to(device).unsqueeze(0)
                mask_t = torch.from_numpy(obs["action_mask"]).to(device).unsqueeze(0)
                done_prev = torch.zeros(1, device=device)
                action, h = policy.act_stateful(
                    obs_t, mask_t, h, done_prev, opp_ids=opp_ids_t,
                )
                obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated
                if done and reward > 0:
                    wins += 1
        results[name] = wins / n_episodes

    return results
