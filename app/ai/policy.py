"""E19 inference adapter for the bot.

The E19 policy (`RecurrentActorCritic` + opponent-class embedding) is
loaded once at import time; every AI game reuses the same weights but
maintains its own GRU hidden state. Inference runs on CPU — the whole
net is ~400K params, forward-pass budget is <10 ms per call which is
comfortably under the Telegram RTT.

We feed `opp_ids = OPPONENT_ID_OTHER` (slot 4 in the training ID table)
for human players. That slot was reserved during E19 training for
self-play snapshots — i.e. "unknown but competent opponent" — which is
the closest analogue to a Telegram user we have. Feeding a rule-based
slot (0-3) would bias the policy toward beating one fixed rule-based
opponent, not a human.

If this module fails to import because torch or the checkpoint is
missing, `app/handlers/rooms_manager.py` catches the `AIUnavailable`
error and falls back to the multiplayer-only experience, so the rest
of the bot still works in environments without the ML stack installed.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AIUnavailable(RuntimeError):
    """Raised when the AI policy cannot be loaded (missing torch / ckpt)."""


_DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__), "models", "E19_snapshot_u2160.pt"
)

# Hidden-slot for "human opponent". Kept in sync with
# `rl.single_agent_env.OPPONENT_ID_OTHER` — if that table ever changes
# we must retrain, not hot-swap the constant here.
_OPPONENT_ID_HUMAN = 4


class AIPolicy:
    """Thread-safe singleton wrapper over the E19 RecurrentActorCritic.

    One instance per bot process. `act()` is called from the aiogram
    event loop but is not awaited — torch inference is synchronous and
    fast enough that we don't bother off-loading to a thread pool.
    A lock guards against two concurrent rooms stepping the policy at
    the same time (defensive; torch's inference graph is re-entrant but
    our hidden-state bookkeeping isn't).
    """

    _instance_lock = threading.Lock()
    _instance: Optional["AIPolicy"] = None

    def __init__(self, ckpt_path: str = _DEFAULT_CKPT) -> None:
        try:
            import torch  # noqa: F401  (imported for the side-effect of failing fast)
        except ImportError as exc:  # pragma: no cover — env without torch
            raise AIUnavailable(
                "torch is not installed — add it to app/requirements.txt "
                "or run `pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu`"
            ) from exc

        if not os.path.exists(ckpt_path):
            raise AIUnavailable(
                f"E19 checkpoint not found at {ckpt_path}. "
                "Did you forget to `scp` it in from the training box?"
            )

        from rl.engine import NUM_ACTIONS
        from rl.policy import load_recurrent_policy

        self._torch = __import__("torch")
        self._device = "cpu"
        self._policy = load_recurrent_policy(
            ckpt_path, n_actions=NUM_ACTIONS, device=self._device,
        )
        self._policy.eval()
        self._hidden_dim = self._policy.hidden
        self._has_opp_embed = self._policy.n_opponents > 0
        self._step_lock = threading.Lock()
        logger.info(
            "AIPolicy loaded from %s: hidden=%d n_opp=%d obs_dim=%d",
            ckpt_path,
            self._hidden_dim,
            self._policy.n_opponents,
            self._policy.obs_dim,
        )

    @classmethod
    def get(cls) -> "AIPolicy":
        """Lazy singleton accessor — builds on first call, reused after."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def obs_dim(self) -> int:
        return int(self._policy.obs_dim)

    @property
    def uses_honest_obs(self) -> bool:
        """Whether the loaded checkpoint was trained with the 52-dim
        'honest' observation (no revealed chamber counts). E19 is 47-dim
        'hack'-layout, so this returns False in production, but the flag
        keeps the rendering code honest if we later promote a honest-obs
        checkpoint without re-reading this file."""
        # 47 = hack (reveals n_live/n_blank), 52 = honest.
        return self.obs_dim == 52

    def initial_hidden(self):
        return self._policy.initial_hidden(batch_size=1, device=self._device)

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        hidden,
        done_prev: bool = False,
    ) -> tuple[int, object]:
        """Select an action for the AI given engine-emitted obs + mask.

        Returns `(action_int, new_hidden)`. The caller is responsible
        for persisting `new_hidden` on the `AIRoom` so the next call
        sees the same GRU memory.
        """
        torch = self._torch
        with self._step_lock:
            obs_t = torch.from_numpy(obs).to(self._device).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(self._device).unsqueeze(0)
            done_t = torch.tensor([float(done_prev)], device=self._device)
            opp_t = (
                torch.tensor([_OPPONENT_ID_HUMAN], dtype=torch.long,
                             device=self._device)
                if self._has_opp_embed
                else None
            )
            action_t, new_hidden = self._policy.act_stateful(
                obs_t, mask_t, hidden, done_t, opp_ids=opp_t,
            )
        return int(action_t.item()), new_hidden
