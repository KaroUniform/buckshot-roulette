"""PettingZoo AEC wrapper around BuckshotEngine.

Two agents: "player_0", "player_1". Action space: Discrete(NUM_ACTIONS).
Observation space: Dict with "observation" (Box) + "action_mask" (Box of 0/1).

Reward convention: zero reward at every intermediate step. Terminal step gives
+1 to the winner and -1 to the loser. Per AEC convention, the reward for an
agent is delivered when it next observes via env.last(); we keep the engine's
"reward from current player perspective" semantics and translate per-agent here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from rl.engine import NUM_ACTIONS, BuckshotEngine


AGENTS = ("player_0", "player_1")


def _agent_to_pid(agent: str) -> int:
    return int(agent.split("_")[1])


class BuckshotAECEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "buckshot_v0", "is_parallelizable": False}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.engine = BuckshotEngine()
        # Compute observation length from a probe reset
        self.engine.reset(seed=0)
        obs_len = self.engine.observation(0).shape[0]
        self._obs_low = np.full(obs_len, -np.inf, dtype=np.float32)
        self._obs_high = np.full(obs_len, np.inf, dtype=np.float32)

        self.possible_agents = list(AGENTS)
        self.action_spaces = {a: spaces.Discrete(NUM_ACTIONS) for a in self.possible_agents}
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(low=self._obs_low, high=self._obs_high, dtype=np.float32),
                    "action_mask": spaces.Box(low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.int8),
                }
            )
            for a in self.possible_agents
        }

        self.agents: list = []
        self.rewards: dict = {}
        self._cumulative_rewards: dict = {}
        self.terminations: dict = {}
        self.truncations: dict = {}
        self.infos: dict = {}
        self._agent_selector: Optional[AgentSelector] = None
        self.agent_selection: Optional[str] = None

    # ---- AEC required API ----

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def observe(self, agent: str) -> dict:
        pid = _agent_to_pid(agent)
        obs = self.engine.observation(pid)
        if pid == self.engine.state.current_player:
            mask = self.engine.legal_actions().astype(np.int8)
        else:
            # Off-turn agents have no legal actions; keep an all-zero mask.
            mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        return {"observation": obs, "action_mask": mask}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        self.engine.reset(seed=seed)
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self._agent_selector.reinit(self.agents)
        # Advance the selector to land on whichever player the engine picked first
        first = AGENTS[self.engine.state.current_player]
        while self._agent_selector.selected_agent != first:
            self._agent_selector.next()
        self.agent_selection = first

    def step(self, action) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        pid = _agent_to_pid(agent)
        if pid != self.engine.state.current_player:
            raise RuntimeError(
                f"Agent {agent} acted out of turn; engine expects player "
                f"{self.engine.state.current_player}"
            )

        # Reset rewards before the step so AEC accumulators are clean
        self.rewards = {a: 0.0 for a in self.agents}

        _state, eng_reward, done, info = self.engine.step(int(action))
        self.infos[agent] = info

        if done:
            winner = self.engine.state.winner
            for a in self.agents:
                self.rewards[a] = 1.0 if _agent_to_pid(a) == winner else -1.0
                self.terminations[a] = True
        else:
            # Sync selector to whichever player the engine is now waiting on
            next_agent = AGENTS[self.engine.state.current_player]
            while self._agent_selector.selected_agent != next_agent:
                self._agent_selector.next()
            self.agent_selection = self._agent_selector.selected_agent

        self._accumulate_rewards()

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        s = self.engine.state
        if s is None:
            return "<not reset>"
        live = sum(1 for x in s.shells if x)
        blank = len(s.shells) - live
        lines = [
            f"turn: player_{s.current_player}  shells: {len(s.shells)} ({live} live, {blank} blank)  dmg_x: {s.damage_mult}",
            f"  player_0: hp={s.players[0].hp}/{s.players[0].max_hp} cuffed={s.players[0].skip_next_turn} inv={s.players[0].inventory.tolist()}",
            f"  player_1: hp={s.players[1].hp}/{s.players[1].max_hp} cuffed={s.players[1].skip_next_turn} inv={s.players[1].inventory.tolist()}",
        ]
        if s.adrenaline_active:
            lines.append("  >> adrenaline pick mode <<")
        if s.done:
            lines.append(f"  GAME OVER. winner=player_{s.winner}")
        return "\n".join(lines)

    def close(self) -> None:
        pass
