# Buckshot Roulette Telegram bot — multiplayer + /ai single-player.
#
# The /ai mode needs BOTH `app/` (the bot) AND `rl/` (the trained policy's
# engine and actor-critic — `app/ai/__init__.py` splices the parent of
# `app/` into sys.path at import time so `from rl.engine import ...`
# resolves inside the container).
FROM python:3.11-slim

WORKDIR /app

# Two-layer dep install:
#  1. CPU-only torch wheel. E19 inference is <10 ms per call on CPU —
#     pulling the CUDA wheel would add ~2 GB to the image for nothing.
#  2. The rest of requirements.txt.
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<3" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the bot AND the RL package. Layout mirrors the repo root so
# `_REPO_ROOT = parent of app/` in `app/ai/__init__.py` resolves to `/`
# and `/rl` becomes importable.
COPY app /app
COPY rl  /rl

CMD ["python", "main.py"]
