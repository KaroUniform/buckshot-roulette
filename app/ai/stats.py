"""Persistent match-history store for the /ai mode.

Every finished game (state.done == True) is written here with enough
context to compute the AI's long-term winrate, a Wilson 95% confidence
interval, and a small human-player leaderboard. Storage is SQLite via
aiosqlite — single file under the container's `./data` volume, WAL
journal so concurrent readers (the `/stats` command) don't block the
writer (game-over path).

AIRoom itself stays storage-agnostic; the handler is where we know the
Telegram chat / user context and calls `record()` there.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import aiosqlite

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ended_at TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    user_id INTEGER,
    human_name TEXT,
    ai_won INTEGER NOT NULL CHECK (ai_won IN (0, 1)),
    human_went_first INTEGER NOT NULL CHECK (human_went_first IN (0, 1)),
    n_turns INTEGER NOT NULL,
    n_reloads INTEGER NOT NULL,
    seed INTEGER,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS ix_matches_user ON matches(user_id);
CREATE INDEX IF NOT EXISTS ix_matches_ended_at ON matches(ended_at);
"""


@dataclass
class MatchRecord:
    ended_at: datetime
    chat_id: int
    user_id: Optional[int]
    human_name: str
    ai_won: bool
    human_went_first: bool
    n_turns: int
    n_reloads: int
    seed: Optional[int]
    duration_ms: int


@dataclass
class Summary:
    total: int
    ai_wins: int
    human_wins: int
    ai_winrate: float  # 0..1
    ai_winrate_lo: float
    ai_winrate_hi: float
    last_ended_at: Optional[datetime]
    last_ai_won: Optional[bool]
    last_n_turns: Optional[int]


@dataclass
class LeaderRow:
    user_id: Optional[int]
    name: str
    wins: int
    games: int


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Returns `(lo, hi)` on [0, 1]. Returns `(0, 0)` for trials == 0 to
    avoid a divide-by-zero; callers should special-case an empty table
    before formatting anyway.
    """
    if trials == 0:
        return (0.0, 0.0)
    n = trials
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


class StatsStore:
    """Thin wrapper around aiosqlite. One instance per process.

    Opens a fresh connection per call — the cost is dwarfed by the
    Telegram round-trip, and this keeps us safe from "connection used
    across coroutines" / threading pitfalls.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def init_schema(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_SCHEMA)
            await db.commit()

    async def record(self, m: MatchRecord) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO matches (
                    ended_at, chat_id, user_id, human_name,
                    ai_won, human_went_first, n_turns, n_reloads,
                    seed, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    m.ended_at.isoformat(),
                    m.chat_id,
                    m.user_id,
                    m.human_name,
                    int(m.ai_won),
                    int(m.human_went_first),
                    m.n_turns,
                    m.n_reloads,
                    m.seed,
                    m.duration_ms,
                ),
            )
            await db.commit()
        logger.info(
            "match recorded: user=%s ai_won=%s turns=%d reloads=%d seed=%s",
            m.user_id, m.ai_won, m.n_turns, m.n_reloads, m.seed,
        )

    async def summary(self) -> Summary:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*), COALESCE(SUM(ai_won), 0) FROM matches"
            ) as cur:
                row = await cur.fetchone()
            total = int(row[0]) if row else 0
            ai_wins = int(row[1]) if row else 0
            last_ended = last_aw = last_turns = None
            if total > 0:
                async with db.execute(
                    "SELECT ended_at, ai_won, n_turns "
                    "FROM matches ORDER BY id DESC LIMIT 1"
                ) as cur:
                    last = await cur.fetchone()
                if last is not None:
                    last_ended = datetime.fromisoformat(last[0])
                    # SQLite doesn't attach tz if the stored string had
                    # none — normalise to UTC-aware for downstream math.
                    if last_ended.tzinfo is None:
                        last_ended = last_ended.replace(tzinfo=timezone.utc)
                    last_aw = bool(last[1])
                    last_turns = int(last[2])
        human_wins = total - ai_wins
        ai_winrate = ai_wins / total if total > 0 else 0.0
        lo, hi = wilson_ci(ai_wins, total)
        return Summary(
            total=total,
            ai_wins=ai_wins,
            human_wins=human_wins,
            ai_winrate=ai_winrate,
            ai_winrate_lo=lo,
            ai_winrate_hi=hi,
            last_ended_at=last_ended,
            last_ai_won=last_aw,
            last_n_turns=last_turns,
        )

    async def top_humans(self, limit: int = 5) -> List[LeaderRow]:
        """Return humans ranked by wins-vs-AI descending, tiebreak by
        fewer total games (better winrate). Only users who've beaten
        the AI at least once make the list; Telegram users without a
        linked user_id (rare) are excluded to avoid merging them.
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT
                    user_id,
                    (SELECT human_name FROM matches m2
                     WHERE m2.user_id = m1.user_id
                     ORDER BY m2.id DESC LIMIT 1) AS name,
                    SUM(CASE WHEN ai_won = 0 THEN 1 ELSE 0 END) AS wins,
                    COUNT(*) AS games
                FROM matches m1
                WHERE user_id IS NOT NULL
                GROUP BY user_id
                HAVING wins > 0
                ORDER BY wins DESC, games ASC
                LIMIT ?
                """,
                (int(limit),),
            ) as cur:
                rows = await cur.fetchall()
        return [
            LeaderRow(
                user_id=r[0],
                name=r[1] or "?",
                wins=int(r[2]),
                games=int(r[3]),
            )
            for r in rows
        ]


_instance: Optional[StatsStore] = None
_init_lock = asyncio.Lock()


def _default_db_path() -> str:
    return os.getenv("STATS_DB_PATH", "/app/data/stats.sqlite3")


async def get_stats() -> StatsStore:
    """Lazy-init singleton. Call this whenever you need the store.

    Double-checked locking so concurrent `/stats` and `/ai` commands
    don't race the schema creation on first boot.
    """
    global _instance
    if _instance is not None:
        return _instance
    async with _init_lock:
        if _instance is None:
            inst = StatsStore(_default_db_path())
            await inst.init_schema()
            _instance = inst
    return _instance


def _reset_for_test() -> None:
    """Drop the cached singleton so tests can point at a different file."""
    global _instance
    _instance = None
