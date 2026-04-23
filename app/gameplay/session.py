"""Canonical rl.engine-backed session shared by PvP and /ai wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from rl.engine import Action, BuckshotEngine

from gameplay.actions import emoji_to_action
from gameplay import render


RELOAD_PAUSE_MS = 2500


@dataclass
class SessionEvent:
    text: str
    keyboard_hint: str
    event_type: str = "message"
    pause_after_ms: int = 0


@dataclass
class Participant:
    name: str
    chat_id: Optional[int]
    is_ai: bool = False


class EngineSession:
    def __init__(
        self,
        participants: Sequence[Participant],
        engine: BuckshotEngine,
        *,
        ai_seat: Optional[int] = None,
        ai_policy=None,
        ai_hidden=None,
        ai_mode: bool = False,
    ) -> None:
        self.participants = list(participants)
        self.engine = engine
        self.ai_seat = ai_seat
        self.ai_policy = ai_policy
        self.ai_hidden = ai_hidden
        self.ai_mode = ai_mode
        self._chat_to_seat = {
            participant.chat_id: seat
            for seat, participant in enumerate(self.participants)
            if participant.chat_id is not None
        }

    @classmethod
    def new_pvp(
        cls,
        *,
        first_name: str,
        first_chat_id: int,
        second_name: str,
        second_chat_id: int,
        seed: Optional[int] = None,
    ) -> "EngineSession":
        engine = BuckshotEngine(seed=seed)
        engine.reset()
        return cls(
            [
                Participant(name=first_name, chat_id=first_chat_id),
                Participant(name=second_name, chat_id=second_chat_id),
            ],
            engine,
            ai_mode=False,
        )

    @classmethod
    def new_vs_ai(
        cls,
        *,
        human_name: str,
        human_chat_id: int,
        policy,
        seed: Optional[int] = None,
    ) -> "EngineSession":
        engine = BuckshotEngine(seed=seed, honest_obs=policy.uses_honest_obs)
        engine.reset()
        engine.state.current_player = 0
        return cls(
            [
                Participant(name=human_name, chat_id=human_chat_id),
                Participant(name="🤖 AI", chat_id=None, is_ai=True),
            ],
            engine,
            ai_seat=1,
            ai_policy=policy,
            ai_hidden=policy.initial_hidden(),
            ai_mode=True,
        )

    @property
    def names(self) -> List[str]:
        return [participant.name for participant in self.participants]

    @property
    def state(self):
        return self.engine.state

    @property
    def game_over(self) -> bool:
        return bool(self.state.done)

    @property
    def winner(self) -> Optional[int]:
        return self.state.winner

    def ai_won(self) -> bool:
        return self.ai_seat is not None and self.winner == self.ai_seat

    def human_chat_ids(self) -> List[int]:
        return [
            participant.chat_id
            for participant in self.participants
            if participant.chat_id is not None
        ]

    def seat_for_chat(self, chat_id: int) -> Optional[int]:
        return self._chat_to_seat.get(chat_id)

    def start(self) -> Dict[int, List[SessionEvent]]:
        dispatch = self._broadcast(
            lambda viewer_id: SessionEvent(
                text=render.opening_text(self.state, self.names, viewer_id, ai_mode=self.ai_mode),
                keyboard_hint=self._hint_for_viewer(viewer_id, done=self.game_over),
                event_type="opening",
            )
        )
        dispatch = self._merge_dispatches(
            dispatch,
            self._broadcast(
                lambda viewer_id: SessionEvent(
                    text=render.reload_banner(self.state, self.names, viewer_id),
                    keyboard_hint=self._hint_for_viewer(viewer_id, done=self.game_over),
                    event_type="banner",
                )
            ),
        )
        if self._current_player_is_ai():
            dispatch = self._merge_dispatches(dispatch, self._drain_ai())
        return dispatch

    def handle_text(self, chat_id: int, text: str) -> Dict[int, List[SessionEvent]]:
        seat = self.seat_for_chat(chat_id)
        if seat is None:
            return {}
        if self.game_over:
            return self._single(
                chat_id,
                SessionEvent(
                    text="The game is already over. Start a rematch or leave.",
                    keyboard_hint="game_over",
                    event_type="notice",
                ),
            )
        if self.state.current_player != seat:
            return self._single(
                chat_id,
                SessionEvent(
                    text="Please, wait your turn.",
                    keyboard_hint="wait",
                    event_type="notice",
                ),
            )

        action = emoji_to_action(text, adrenaline_active=self.state.adrenaline_active)
        if action is None:
            return self._single(
                chat_id,
                SessionEvent(
                    text="Make a valid move.",
                    keyboard_hint="human_turn",
                    event_type="notice",
                ),
            )

        legal = self.engine.legal_actions()
        if not legal[int(action)]:
            return self._single(
                chat_id,
                SessionEvent(
                    text="That action isn't legal right now. Try a different move.",
                    keyboard_hint="human_turn",
                    event_type="notice",
                ),
            )

        return self._step_actor(seat, action, actor_is_ai=False)

    def _current_player_is_ai(self) -> bool:
        return (
            self.ai_seat is not None
            and not self.game_over
            and self.state.current_player == self.ai_seat
        )

    def _step_actor(self, actor_id: int, action: Action, *, actor_is_ai: bool) -> Dict[int, List[SessionEvent]]:
        prev_reloads = self.state.n_reloads
        _, _, done, info = self.engine.step(int(action))

        if actor_is_ai:
            dispatch = self._compose_ai_dispatch(actor_id, action, info, prev_reloads, done)
        else:
            dispatch = self._compose_human_dispatch(actor_id, action, info, prev_reloads, done)

        if done:
            dispatch = self._merge_dispatches(dispatch, self._game_over_dispatch())
            return dispatch

        if self._current_player_is_ai():
            dispatch = self._merge_dispatches(dispatch, self._drain_ai())
        return dispatch

    def _drain_ai(self) -> Dict[int, List[SessionEvent]]:
        dispatch: Dict[int, List[SessionEvent]] = {}
        for _ in range(64):
            if not self._current_player_is_ai():
                break
            obs = self.engine.observation(self.ai_seat)
            mask = self.engine.legal_actions()
            action, self.ai_hidden = self.ai_policy.act(
                obs, mask, self.ai_hidden, done_prev=False,
            )
            dispatch = self._merge_dispatches(
                dispatch,
                self._step_actor(self.ai_seat, Action(action), actor_is_ai=True),
            )
            if self.game_over:
                return dispatch
        else:
            dispatch = self._merge_dispatches(
                dispatch,
                self._broadcast(
                    lambda viewer_id: SessionEvent(
                        text="⚠️ AI exceeded its move budget — report this as a bug.",
                        keyboard_hint=self._hint_for_viewer(viewer_id, done=self.game_over),
                        event_type="notice",
                    )
                ),
            )
        return dispatch

    def _compose_human_dispatch(
        self,
        actor_id: int,
        action: Action,
        info: dict,
        prev_reloads: int,
        done: bool,
    ) -> Dict[int, List[SessionEvent]]:
        dispatch = self._broadcast(
            lambda viewer_id: SessionEvent(
                text=render.action_caption(
                    int(action), info, actor_id=actor_id, viewer_id=viewer_id, names=self.names,
                ),
                keyboard_hint=self._hint_for_viewer(viewer_id, done=done),
                event_type="caption",
            )
        )
        if self.state.n_reloads > prev_reloads:
            dispatch = self._merge_dispatches(
                dispatch,
                self._broadcast(
                    lambda viewer_id: SessionEvent(
                        text=render.reload_banner(self.state, self.names, viewer_id),
                        keyboard_hint=self._hint_for_viewer(viewer_id, done=done),
                        event_type="banner",
                        pause_after_ms=RELOAD_PAUSE_MS,
                    )
                ),
            )
        dispatch = self._merge_dispatches(
            dispatch,
            self._broadcast(
                lambda viewer_id: SessionEvent(
                    text=render.game_summary(self.state, self.names, viewer_id),
                    keyboard_hint=self._hint_for_viewer(viewer_id, done=done),
                    event_type="summary",
                )
            ),
        )
        return dispatch

    def _compose_ai_dispatch(
        self,
        actor_id: int,
        action: Action,
        info: dict,
        prev_reloads: int,
        done: bool,
    ) -> Dict[int, List[SessionEvent]]:
        dispatch = self._broadcast(
            lambda viewer_id: SessionEvent(
                text=render.action_caption(
                    int(action), info, actor_id=actor_id, viewer_id=viewer_id, names=self.names,
                ),
                keyboard_hint=self._hint_for_viewer(viewer_id, done=done),
                event_type="caption",
            )
        )
        if self.state.n_reloads > prev_reloads:
            dispatch = self._merge_dispatches(
                dispatch,
                self._broadcast(
                    lambda viewer_id: SessionEvent(
                        text=render.reload_banner(self.state, self.names, viewer_id),
                        keyboard_hint=self._hint_for_viewer(viewer_id, done=done),
                        event_type="banner",
                        pause_after_ms=RELOAD_PAUSE_MS,
                    )
                ),
            )
        if done or not self._current_player_is_ai():
            dispatch = self._merge_dispatches(
                dispatch,
                self._broadcast(
                    lambda viewer_id: SessionEvent(
                        text=render.game_summary(self.state, self.names, viewer_id),
                        keyboard_hint=self._hint_for_viewer(viewer_id, done=done),
                        event_type="summary",
                    )
                ),
            )
        return dispatch

    def _game_over_dispatch(self) -> Dict[int, List[SessionEvent]]:
        return self._broadcast(
            lambda viewer_id: SessionEvent(
                text=render.game_over_message(
                    self.state,
                    self.names,
                    viewer_id,
                    ai_mode=self.ai_mode,
                    ai_seat=self.ai_seat,
                ),
                keyboard_hint="game_over",
                event_type="game_over",
            )
        )

    def _broadcast(self, factory) -> Dict[int, List[SessionEvent]]:
        dispatch: Dict[int, List[SessionEvent]] = {}
        for viewer_id, participant in enumerate(self.participants):
            if participant.chat_id is None:
                continue
            dispatch.setdefault(participant.chat_id, []).append(factory(viewer_id))
        return dispatch

    def _hint_for_viewer(self, viewer_id: int, *, done: bool) -> str:
        if done:
            return "game_over"
        if self.state.current_player == viewer_id and not self.participants[viewer_id].is_ai:
            return "human_turn"
        return "wait"

    @staticmethod
    def _merge_dispatches(
        left: Dict[int, List[SessionEvent]],
        right: Dict[int, List[SessionEvent]],
    ) -> Dict[int, List[SessionEvent]]:
        merged = {chat_id: list(events) for chat_id, events in left.items()}
        for chat_id, events in right.items():
            merged.setdefault(chat_id, []).extend(events)
        return merged

    @staticmethod
    def _single(chat_id: int, event: SessionEvent) -> Dict[int, List[SessionEvent]]:
        return {chat_id: [event]}
