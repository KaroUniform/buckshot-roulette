"""Single-agent gymnasium env with a frozen opponent inside.

Sidesteps multi-agent training complexity: the agent sees only its own turns,
and the wrapper internally consults `opponent_fn` whenever it's the
opponent's turn. The "environment" is therefore engine + opponent.

This is the standard self-play pattern used in AlphaStar, OpenAI Five, and
most RL-on-2-player-games papers — train a single agent against a pool of
frozen past selves (and rule-based baselines for warmup).
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.engine import NUM_ACTIONS, BuckshotEngine
from rl.opponents import OpponentFn, OpponentPool, random_opponent


class SingleAgentBuckshotEnv(gym.Env):
    """Gymnasium env. Agent plays against `opponent_fn`.

    Args:
        opponent_pool: sampled at reset() to pick this episode's opponent.
            If None, defaults to a pool containing only `random`.
        agent_pid: 0 or 1 to fix which seat the agent plays, or None to
            randomize per-episode (recommended for training).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_pool: Optional[OpponentPool] = None,
        agent_pid: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.engine = BuckshotEngine()
        # Probe observation length on a throwaway engine so the real engine's
        # RNG isn't pinned to seed=0 (which would make seedless reset()s
        # repeat the same game forever across env instances).
        _probe = BuckshotEngine()
        _probe.reset(seed=0)
        obs_len = _probe.observation(0).shape[0]

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.int8
                ),
            }
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.pool = opponent_pool or OpponentPool({"random": random_opponent})
        self.agent_pid = agent_pid
        self._opp_fn: OpponentFn = random_opponent
        self._opp_name = "random"
        self._agent_pid_this_ep = 0
        self._opp_rng = np.random.default_rng()

    # ---- gym API ----

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Rare edge case: the opponent's first moves can end the game
        # (e.g., pills backfire into a kill). gymnasium's SyncVectorEnv
        # doesn't learn the game ended during reset, so it would feed us
        # a step() on a terminal state, creating phantom training
        # transitions. Loop with perturbed seeds until we reset into a
        # non-terminal state. In practice this almost always resolves in
        # one attempt.
        base_seed = seed
        for attempt in range(50):
            if base_seed is not None:
                eff_seed = base_seed + attempt
                self._opp_rng = np.random.default_rng(eff_seed + 1)
            else:
                eff_seed = None
            self.engine.reset(seed=eff_seed)

            if self.agent_pid is None:
                self._agent_pid_this_ep = int(self._opp_rng.integers(0, 2))
            else:
                self._agent_pid_this_ep = self.agent_pid

            self._opp_name, self._opp_fn = self.pool.sample(self._opp_rng)

            terminated, terminal_reward = self._play_opponent_until_agent_turn()
            if not terminated:
                return self._obs(), {
                    "opponent": self._opp_name,
                    "agent_pid": self._agent_pid_this_ep,
                }
            # else: very rare — retry with a perturbed seed (or a fresh
            # random seed if base_seed was None; np.random.default_rng()
            # with no seed picks entropy from OS)

        # Degenerate fallback: surface the terminal state with flags so
        # callers (eval.py) can short-circuit; the PPO training loop
        # also handles this safely via _was_dead_step-like semantics
        # because terminations==True would fire on the next step.
        return self._obs(), {
            "opponent": self._opp_name,
            "agent_pid": self._agent_pid_this_ep,
            "_terminated_in_reset": True,
            "_terminal_reward": terminal_reward,
        }

    def step(self, action):
        # Caller may have ignored the _terminated_in_reset flag and stepped anyway —
        # in that case mirror the terminal step.
        if self.engine.state.done:
            obs = self._obs()
            winner = self.engine.state.winner
            reward = 1.0 if winner == self._agent_pid_this_ep else -1.0
            return obs, reward, True, False, {"opponent": self._opp_name}

        # Agent acts
        self.engine.step(int(action))
        if self.engine.state.done:
            return self._terminal_return()

        # Opponent acts until either game ends or it's our turn again
        terminated, _ = self._play_opponent_until_agent_turn()
        if terminated:
            return self._terminal_return()

        return self._obs(), 0.0, False, False, {"opponent": self._opp_name}

    def render(self) -> Optional[str]:
        s = self.engine.state
        if s is None:
            return "<not reset>"
        return (
            f"agent=player_{self._agent_pid_this_ep} vs {self._opp_name}  "
            f"turn=player_{s.current_player}  hps={s.players[0].hp}/{s.players[1].hp}  "
            f"shells={len(s.shells)}"
        )

    # ---- internals ----

    def _obs(self) -> dict:
        obs = self.engine.observation(self._agent_pid_this_ep).astype(np.float32)
        if self.engine.state.current_player == self._agent_pid_this_ep and not self.engine.state.done:
            mask = self.engine.legal_actions().astype(np.int8)
        else:
            mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        return {"observation": obs, "action_mask": mask}

    def _play_opponent_until_agent_turn(self) -> tuple[bool, float]:
        """Run opponent moves while it's their turn. Returns (terminated, reward)."""
        while (
            not self.engine.state.done
            and self.engine.state.current_player != self._agent_pid_this_ep
        ):
            opp_pid = 1 - self._agent_pid_this_ep
            obs = self.engine.observation(opp_pid).astype(np.float32)
            mask = self.engine.legal_actions().astype(np.int8)
            if not mask.any():
                # Shouldn't happen with current engine; safety net
                break
            action = self._opp_fn(obs, mask, self._opp_rng)
            self.engine.step(int(action))
        if self.engine.state.done:
            winner = self.engine.state.winner
            return True, (1.0 if winner == self._agent_pid_this_ep else -1.0)
        return False, 0.0

    def _terminal_return(self):
        winner = self.engine.state.winner
        reward = 1.0 if winner == self._agent_pid_this_ep else -1.0
        return self._obs(), float(reward), True, False, {"opponent": self._opp_name}
