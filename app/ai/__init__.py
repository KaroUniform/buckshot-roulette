"""Single-player AI mode for the Telegram bot.

The RL engine at repo-root `rl/` is the *authoritative* game state for AI
matches. We do NOT reuse `app/core/scene.py` here — its rule-set diverges
from the trained policy's (notably the "1 non-adrenaline item per turn"
restriction and the stacking handsaw bug, both deliberately removed in the
RL engine). Dual-tracking state across both would invite subtle drift and
give the AI a different view than it was trained on.

The rl/ package lives at `<repo>/rl`, the bot lives at `<repo>/app`. The bot
is launched with `cwd=app/`, so we splice the parent of `app/` into
sys.path when this package is imported. This keeps all RL-side imports
unchanged (`from rl.engine import ...`) while still letting `app/main.py`
boot in its usual working directory.
"""

from __future__ import annotations

import os
import sys

_APP_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_REPO_ROOT = os.path.abspath(os.path.join(_APP_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    # Prepend so repo-root `rl/` wins over anything installed globally.
    sys.path.insert(0, _REPO_ROOT)
