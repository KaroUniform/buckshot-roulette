"""/stats command — public winrate summary and top-5 human leaderboard.

Read-only handler. Reaches into the SQLite store via `ai.stats.get_stats`
(lazy singleton) and renders a plain-text message. No Markdown / HTML
parse mode — human names can contain arbitrary characters and breaking
format parsing on a name with an underscore is a bad trade for bold.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

logger = logging.getLogger(__name__)

router = Router()


def _ago(dt: datetime) -> str:
    """Short human-readable 'N <unit> ago' string, UTC-aware."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = int((now - dt).total_seconds())
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


@router.message(Command("stats"))
async def show_stats(message: Message) -> None:
    from ai.stats import get_stats

    try:
        store = await get_stats()
    except Exception as exc:  # pragma: no cover — defensive for first boot issues
        logger.exception("stats store init failed")
        await message.answer(f"Stats unavailable: {exc}")
        return

    summary = await store.summary()
    if summary.total == 0:
        await message.answer("📊 No matches recorded yet. Start one with /ai.")
        return

    top = await store.top_humans(limit=5)

    lines = [
        "📊 AI winrate",
        f"Total matches: {summary.total}",
        (
            f"🤖 AI wins: {summary.ai_wins}  "
            f"({summary.ai_winrate * 100:.1f}%)"
        ),
        f"👤 Human wins: {summary.human_wins}",
    ]
    # CI is meaningless on a couple of samples; only show once the
    # interval starts conveying something useful.
    if summary.total >= 5:
        lines.append(
            f"95% CI: {summary.ai_winrate_lo * 100:.1f}%–"
            f"{summary.ai_winrate_hi * 100:.1f}%"
        )
    if summary.last_ended_at is not None:
        who = "AI won" if summary.last_ai_won else "human won"
        turns = summary.last_n_turns or 0
        lines.append(
            f"Last: {_ago(summary.last_ended_at)} — {who} ({turns} turns)"
        )
    lines.append("")
    lines.append("🏆 Top humans")
    if top:
        for i, row in enumerate(top, 1):
            lines.append(f"{i}. {row.name} — {row.wins} wins ({row.games} games)")
    else:
        lines.append("Nobody has beaten the AI yet.")

    await message.answer("\n".join(lines))
