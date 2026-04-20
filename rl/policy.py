"""Actor-critic MLP with action masking, kept intentionally small.

Tied to BuckshotEngine: 47-dim observation, 19 discrete actions. Easy to
swap for a larger / transformer / recurrent network later — interface is
stable: `get_action_and_value(obs, mask, action=None)` for PPO, `act(obs, mask)`
for evaluation/opponents.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _orthogonal_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        # Smaller std on the actor head — standard CleanRL trick for stable starts
        self.actor = _orthogonal_init(nn.Linear(hidden, n_actions), std=0.01)
        # Critic head with std=1 per CleanRL
        self.critic = _orthogonal_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(obs)
        return self.actor(z), self.critic(z).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        # Set illegal-action logits to a large negative number so softmax assigns ~0 prob
        masked_logits = logits.masked_fill(mask == 0, -1e8)
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(obs)
        masked_logits = logits.masked_fill(mask == 0, -1e8)
        return Categorical(logits=masked_logits).sample()


def load_policy(path: str, n_actions: int, device: str = "cpu") -> ActorCritic:
    """Load an ActorCritic checkpoint, auto-detecting obs_dim/hidden from weights.

    Works across old runs that may have used different hidden sizes.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    w0 = state["body.0.weight"]
    hidden, obs_dim = int(w0.shape[0]), int(w0.shape[1])
    pol = ActorCritic(obs_dim, n_actions, hidden=hidden).to(device)
    pol.load_state_dict(state)
    pol.eval()
    return pol
