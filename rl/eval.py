"""Evaluate a policy against a set of named opponents.

Returns per-opponent win rate over `n_episodes` games. Uses fresh seeds and
a fresh env per episode for clean independence (cheap given our small game).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from rl.opponents import NAMED_OPPONENTS, OpponentFn, OpponentPool
from rl.single_agent_env import SingleAgentBuckshotEnv


def evaluate_policy(
    policy,
    opponents: Optional[dict[str, OpponentFn]] = None,
    n_episodes: int = 200,
    seed: int = 12345,
    device: str = "cpu",
) -> dict[str, float]:
    """For each named opponent, return the policy's win rate.

    `policy` must implement `.act(obs_tensor, mask_tensor) -> action_int_tensor`.
    """
    opponents = opponents or NAMED_OPPONENTS
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}

    for name, fn in opponents.items():
        pool = OpponentPool({name: fn})
        env = SingleAgentBuckshotEnv(opponent_pool=pool)
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
) -> dict[str, float]:
    """Like evaluate_policy but for RecurrentActorCritic.

    Maintains per-episode hidden state (reset on each env reset).
    """
    opponents = opponents or NAMED_OPPONENTS
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}

    for name, fn in opponents.items():
        pool = OpponentPool({name: fn})
        env = SingleAgentBuckshotEnv(opponent_pool=pool)
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
            while not done:
                obs_t = torch.from_numpy(obs["observation"]).to(device).unsqueeze(0)
                mask_t = torch.from_numpy(obs["action_mask"]).to(device).unsqueeze(0)
                done_prev = torch.zeros(1, device=device)
                action, h = policy.act_stateful(obs_t, mask_t, h, done_prev)
                obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated
                if done and reward > 0:
                    wins += 1
        results[name] = wins / n_episodes

    return results
