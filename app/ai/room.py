"""AIRoom — single source of truth for a 1-human-vs-E19 game.

The room owns a `BuckshotEngine` instance (the RL-side game state), a
reference to the shared `AIPolicy` (weights), and a per-session GRU
hidden state. Human messages come in via `step_human(action)`; between
human turns the room drives the AI until it's either the human's turn
again or the game is over.

The room does NOT own the Telegram rendering layer — callers pass in a
`send()` coroutine and the room feeds it pre-rendered messages +
keyboards in order. Keeping I/O out of the room keeps it unit-testable
and the handler thin.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from rl.engine import Action, BuckshotEngine, NUM_ACTIONS

from .policy import AIPolicy


@dataclass
class AIRoomEvent:
    """One step in the game, in chronological order.

    `text` is the human-facing narration. `keyboard_hint` is one of:
        'human_turn' — the human's reply keyboard should be rebuilt
        'wait'       — render the 🕓 waiting keyboard
        'game_over'  — terminal; render a rematch/leave keyboard

    `loadout_text` is non-None when the event should be followed by an
    initial-chamber declaration. It's pre-rendered here (not at send
    time) because `room.state` is mutable — by the time the handler
    awaits `bot.send_message`, additional AI sub-actions may have
    reloaded the chamber and the live state would report the *new*
    round's counts. Pinning the string at event-construction avoids
    that drift.
    """
    text: str
    keyboard_hint: str = "human_turn"
    loadout_text: Optional[str] = None


@dataclass
class AIRoom:
    """One human vs E19. Deterministic-per-seed; 1 engine per room.

    The `turn_idx` tracking in our engine is always 0/1; we pick which
    one is the human at reset-time and keep it constant for the match
    (no side-swap). `human_id` lands on whichever slot the engine picks
    to go first roughly half the time. That's fine — E19 was trained
    in self-play with no first-player bias.
    """

    human_id: int
    human_name: str
    engine: BuckshotEngine
    policy: AIPolicy
    hidden: object  # torch.Tensor, opaque to avoid torch import here
    ai_done_prev: bool = False

    @classmethod
    def new(
        cls, human_name: str, policy: Optional[AIPolicy] = None,
        seed: Optional[int] = None,
    ) -> "AIRoom":
        if policy is None:
            policy = AIPolicy.get()
        # Respect the policy's trained obs layout. E19 is "hack" (47-dim).
        # Choose human slot by coin toss so neither side has a home-field
        # advantage; the policy saw both start positions during training.
        rng = random.Random(seed)
        human_id = rng.randint(0, 1)
        engine = BuckshotEngine(
            seed=seed,
            honest_obs=policy.uses_honest_obs,
        )
        engine.reset()
        return cls(
            human_id=human_id,
            human_name=human_name,
            engine=engine,
            policy=policy,
            hidden=policy.initial_hidden(),
            ai_done_prev=False,
        )

    # ---- helpers ----

    @property
    def ai_id(self) -> int:
        return 1 - self.human_id

    @property
    def state(self):
        return self.engine.state

    @property
    def game_over(self) -> bool:
        return bool(self.engine.state.done)

    @property
    def is_human_turn(self) -> bool:
        s = self.engine.state
        return (not s.done) and s.current_player == self.human_id

    # ---- main driver ----

    def start(self) -> List[AIRoomEvent]:
        """Return the opening event list — loadout + first keyboard.

        If the AI happens to go first, this also drains the AI's opening
        moves before handing control back to the human.
        """
        from . import render
        events: List[AIRoomEvent] = []
        events.append(
            AIRoomEvent(
                text=self._opening_text(),
                keyboard_hint="wait" if not self.is_human_turn else "human_turn",
                loadout_text=render.loadout_line(self.state),
            )
        )
        if not self.is_human_turn and not self.game_over:
            events.extend(self._drain_ai())
        return events

    def step_human(self, action: Action) -> List[AIRoomEvent]:
        """Apply a human action to the engine, then drain any AI moves.

        Illegal actions (e.g. pressing 🪚 while the shotgun is empty)
        return a single 'soft error' event and do NOT advance state.
        This matches the 2-player flow where "Make a valid turn" was
        the text for the same condition.
        """
        if self.game_over:
            return [AIRoomEvent(
                text="The game is already over. Use /ai to start a new one.",
                keyboard_hint="game_over",
            )]
        if not self.is_human_turn:
            return [AIRoomEvent(
                text="⏳ Please wait — the AI is still moving.",
                keyboard_hint="wait",
            )]

        legal = self.engine.legal_actions()
        if not legal[int(action)]:
            return [AIRoomEvent(
                text="That action isn't legal right now. Try a different move.",
                keyboard_hint="human_turn",
            )]

        prev_reloads = self.engine.state.n_reloads
        state, reward, done, info = self.engine.step(int(action))

        events = self._compose_human_events(action, info, prev_reloads, done)
        if done:
            events.append(self._game_over_event())
            return events
        # AI may need to move until control comes back. If the human's
        # action was USE_ADRENALINE the engine will be in pick-mode with
        # the *human* still the current player — stop draining and let
        # them pick.
        if self.is_human_turn:
            return events
        events.extend(self._drain_ai())
        return events

    # ---- AI drive loop ----

    def _drain_ai(self) -> List[AIRoomEvent]:
        """Run the policy until the human's turn or terminal."""
        events: List[AIRoomEvent] = []
        # Guard rail — reasonable upper bound on a single AI burst.
        # If this fires we probably have a legality-mask bug.
        for _ in range(64):
            if self.game_over or self.is_human_turn:
                break
            obs = self.engine.observation(self.ai_id)
            mask = self.engine.legal_actions()
            action, self.hidden = self.policy.act(
                obs, mask, self.hidden, done_prev=self.ai_done_prev,
            )
            self.ai_done_prev = False

            prev_reloads = self.engine.state.n_reloads
            _, _, done, info = self.engine.step(int(action))

            events.extend(self._compose_ai_events(action, info, prev_reloads, done))
            if done:
                events.append(self._game_over_event())
                return events
        else:
            events.append(AIRoomEvent(
                text="⚠️ AI exceeded its move budget — report this as a bug.",
                keyboard_hint="human_turn",
            ))
        return events

    # ---- event composition ----

    def _opening_text(self) -> str:
        from . import render
        starter = "You" if self.is_human_turn else "🤖 AI"
        header = f"🪙 Coin toss — {starter} go first."
        body = render.game_summary(self.state, self.human_name, self.human_id)
        return f"{header}\n\n{body}"

    def _compose_human_events(
        self, action: Action, info: dict, prev_reloads: int, done: bool,
    ) -> List[AIRoomEvent]:
        from . import render
        events: List[AIRoomEvent] = []
        # When the engine has marked the game terminal on this step, every
        # event we emit must render under the game-over keyboard — the
        # live state has no legal actions and `human_turn_keyboard` would
        # produce an empty ReplyKeyboardMarkup that Telegram rejects.
        default_hint = "game_over" if done else "human_turn"
        mid_hint = "game_over" if done else "wait"
        events.append(AIRoomEvent(
            text=render.action_caption(int(action), info, actor="human"),
            keyboard_hint=default_hint,
        ))
        if self.engine.state.n_reloads > prev_reloads:
            # Snapshot the new round's declaration NOW — deferring to
            # handler send-time would show stale counts once a later
            # AI reload mutates round_initial_live/_blank again.
            events.append(AIRoomEvent(
                text=render.reload_banner(),
                keyboard_hint=default_hint,
                loadout_text=render.loadout_line(self.state),
            ))
        # Final state summary + keyboard hint (handler rebuilds reply
        # markup from the live state, so we just flag which mode).
        summary = render.game_summary(self.state, self.human_name, self.human_id)
        if done:
            hint = "game_over"
        else:
            hint = "human_turn" if self.is_human_turn else mid_hint
        events.append(AIRoomEvent(text=summary, keyboard_hint=hint))
        return events

    def _compose_ai_events(
        self, action: Action, info: dict, prev_reloads: int, done: bool,
    ) -> List[AIRoomEvent]:
        from . import render
        # When the AI action terminates the game, switch every hint to
        # "game_over" so the handler doesn't paint the "🕓 thinking" key-
        # board on the kill-shot and then sleep 1.1s before announcing
        # the result. Symmetric to `_compose_human_events`.
        mid_hint = "game_over" if done else "wait"
        events: List[AIRoomEvent] = [
            AIRoomEvent(
                text=render.action_caption(int(action), info, actor="ai"),
                keyboard_hint=mid_hint,
            ),
        ]
        if self.engine.state.n_reloads > prev_reloads:
            # Snapshot the new round's declaration here — the AI may
            # chain further sub-actions that trigger another reload
            # before the handler gets to dispatch this event, and the
            # live `state.round_initial_*` would then reflect the
            # later round instead of the one we want to announce.
            events.append(AIRoomEvent(
                text=render.reload_banner(),
                keyboard_hint=mid_hint,
                loadout_text=render.loadout_line(self.state),
            ))
        if self.is_human_turn:
            # Final summary only when control returns — avoids spamming
            # the same HP block after every AI sub-action.
            events.append(AIRoomEvent(
                text=render.game_summary(self.state, self.human_name, self.human_id),
                keyboard_hint="human_turn",
            ))
        return events

    def _game_over_event(self) -> AIRoomEvent:
        from . import render
        return AIRoomEvent(
            text=render.game_over_message(self.state, self.human_name, self.human_id),
            keyboard_hint="game_over",
        )
