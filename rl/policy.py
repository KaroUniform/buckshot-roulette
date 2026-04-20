"""Actor-critic MLP with action masking, kept intentionally small.

Tied to BuckshotEngine: 47-dim observation, 19 discrete actions. Easy to
swap for a larger / transformer / recurrent network later — interface is
stable: `get_action_and_value(obs, mask, action=None)` for PPO, `act(obs, mask)`
for evaluation/opponents.
"""

from __future__ import annotations

from typing import Optional

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


class RecurrentActorCritic(nn.Module):
    """GRU-based actor-critic with action masking.

    Architecture:
        obs -> MLP embed -> GRU(hidden) -> (actor_head, critic_head)

    Interface designed for PPO rollouts:
      - Single-step (rollout collection):
          get_action_and_value(obs, mask, h, done_prev, action=None)
        Returns (action, log_prob, entropy, value, new_h).
        `done_prev` is 1 where the previous step ended the env's episode;
        hidden is zeroed on those envs BEFORE the GRU step.
      - Batched sequence (training minibatch):
          forward_sequence(obs_seq, h_init, dones_seq) -> (logits_seq, values_seq)
        Processes T steps respecting mid-sequence resets. Python-for loop
        over T — negligible for T=128; trades a microbench for correctness.
      - Stateful single-agent eval (frozen opponent inside env):
          act_stateful(obs, mask, h, done_prev) -> (action, new_h)
        Deterministic sampling; no gradient; used by the opponent pool.

    We intentionally do NOT implement the stateless `act(obs, mask)` that
    feedforward ActorCritic has — a recurrent policy needs its hidden
    state to act consistently. Callers that want a memoryless evaluation
    should reset h to zero per call.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        embed: int | None = None,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.aux_dim = aux_dim
        # Embed obs into a hidden-sized vector before the GRU. Keeps GRU input
        # dimension uniform so hidden size is the only capacity knob.
        embed_dim = embed if embed is not None else hidden
        self.obs_embed = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, embed_dim)),
            nn.Tanh(),
        )
        # Single-layer GRU. Multi-layer doesn't help here — episodes are short
        # and the task is information integration, not hierarchical.
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=False,  # we feed (T=1, B, E) directly
        )
        # Reset GRU weights with the CleanRL-recommended scheme.
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, 1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        self.actor = _orthogonal_init(nn.Linear(hidden, n_actions), std=0.01)
        self.critic = _orthogonal_init(nn.Linear(hidden, 1), std=1.0)
        # Optional auxiliary regression head: predicts continuous-valued
        # ground-truth quantities from the GRU's hidden state. Used for
        # representation shaping — e.g. forcing the trunk to track the
        # remaining chamber composition (n_live, n_blank). Only constructed
        # when aux_dim > 0 so older checkpoints without an aux head still
        # load cleanly with strict=True.
        # (Initialized with std=1.0 since regression targets are O(1).)
        self.aux_head: Optional[nn.Linear] = (
            _orthogonal_init(nn.Linear(hidden, aux_dim), std=1.0)
            if aux_dim > 0 else None
        )
        # PyTorch emits a cudnn warning if weights aren't contiguous in memory
        # (easy to trigger after .to(device) / deepcopy). Calling
        # flatten_parameters here makes the first forward pass warning-free
        # and slightly faster.
        self.gru.flatten_parameters()

    def _apply(self, fn):  # noqa: D401
        # torch's Module._apply runs on .to(), .cuda(), .cpu() — re-flatten
        # so moved/cloned modules don't re-trigger the warning.
        out = super()._apply(fn)
        if isinstance(self.gru, nn.GRU):
            self.gru.flatten_parameters()
        return out

    def initial_hidden(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden, device=device)

    def _step(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (B, obs_dim) — returns body (B, hidden), new h (1, B, hidden)
        x = self.obs_embed(obs).unsqueeze(0)  # (1, B, E)
        y, new_h = self.gru(x, h)
        return y.squeeze(0), new_h

    def forward_step(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        done_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # done_prev: (B,), 1 if previous step terminated an episode
        reset = done_prev.view(1, -1, 1).to(h.dtype)
        h = h * (1.0 - reset)
        body, new_h = self._step(obs, h)
        return self.actor(body), self.critic(body).squeeze(-1), new_h

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        h_init: torch.Tensor,
        dones_seq: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Process a (T, B) sequence with correct mid-sequence resets.

        dones_seq[t] = 1 if step t-1 ended the episode for that env, i.e.
        the hidden state just BEFORE step t must be zeroed. Matches the
        semantics of the rollout buffer's `dones_buf`.

        With return_aux=True and aux_head present, also returns the
        per-timestep aux prediction tensor (T, B, aux_dim). Returned as a
        third element of the tuple. With return_aux=True but no aux_head,
        the third element is None.
        """
        T, B = obs_seq.shape[:2]
        h = h_init
        logits_list = []
        values_list = []
        aux_list = [] if return_aux and self.aux_head is not None else None
        for t in range(T):
            reset = dones_seq[t].view(1, B, 1).to(h.dtype)
            h = h * (1.0 - reset)
            body, h = self._step(obs_seq[t], h)
            logits_list.append(self.actor(body))
            values_list.append(self.critic(body).squeeze(-1))
            if aux_list is not None:
                aux_list.append(self.aux_head(body))
        logits = torch.stack(logits_list, dim=0)
        values = torch.stack(values_list, dim=0)
        if return_aux:
            aux = torch.stack(aux_list, dim=0) if aux_list is not None else None
            return logits, values, aux
        return logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        h: torch.Tensor,
        done_prev: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, new_h = self.forward_step(obs, h, done_prev)
        masked_logits = logits.masked_fill(mask == 0, -1e8)
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, new_h

    @torch.no_grad()
    def act_stateful(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        h: torch.Tensor,
        done_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _, new_h = self.forward_step(obs, h, done_prev)
        masked_logits = logits.masked_fill(mask == 0, -1e8)
        return Categorical(logits=masked_logits).sample(), new_h


def load_recurrent_policy(
    path: str, n_actions: int, device: str = "cpu"
) -> RecurrentActorCritic:
    """Load a RecurrentActorCritic from disk, auto-detecting obs_dim/hidden/aux_dim.

    Detects layout from stored weight shapes:
      - obs_embed.0.weight: (embed, obs_dim)
      - gru.weight_ih_l0:   (3*hidden, embed)  -- GRU has 3 gates
      - aux_head.weight:    (aux_dim, hidden)  -- absent for pre-E12b checkpoints
    """
    state = torch.load(path, map_location=device, weights_only=True)
    w_embed = state["obs_embed.0.weight"]
    embed_dim, obs_dim = int(w_embed.shape[0]), int(w_embed.shape[1])
    w_ih = state["gru.weight_ih_l0"]
    hidden = int(w_ih.shape[0] // 3)
    aux_dim = int(state["aux_head.weight"].shape[0]) if "aux_head.weight" in state else 0
    pol = RecurrentActorCritic(
        obs_dim, n_actions, hidden=hidden, embed=embed_dim, aux_dim=aux_dim
    ).to(device)
    pol.load_state_dict(state)
    pol.eval()
    return pol
