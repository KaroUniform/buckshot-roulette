from .engine import Action, BuckshotEngine, GameState, Item, NUM_ACTIONS, NUM_ITEMS

# `BuckshotAECEnv` pulls gymnasium + pettingzoo (training-only deps).
# Keep the import lazy so inference-only consumers — notably the Telegram
# bot at `app/` — don't have to ship the full training stack. Training
# code can still do `from rl.env import BuckshotAECEnv` directly to
# trigger an ImportError if the deps are actually missing.
try:
    from .env import BuckshotAECEnv  # noqa: F401
except ImportError:  # pragma: no cover
    BuckshotAECEnv = None

__all__ = [
    "Action",
    "BuckshotAECEnv",
    "BuckshotEngine",
    "GameState",
    "Item",
    "NUM_ACTIONS",
    "NUM_ITEMS",
]
