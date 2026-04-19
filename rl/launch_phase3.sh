#!/bin/bash
# Launch Phase 3: 50M-step long training run with best config from Phase 2 sweep.
# Pass the best config's hyperparameters as env vars, or edit defaults below.
# Pin to $GPU (default 1).

set -euo pipefail

GPU="${GPU:-1}"
LR="${LR:-3e-4}"
ENT_COEF="${ENT_COEF:-0.01}"
HIDDEN="${HIDDEN:-256}"
RUN_NAME="${RUN_NAME:-phase3_final_50M}"
STEPS="${STEPS:-50000000}"

cd "$(dirname "$0")/.."
mkdir -p rl_runs

CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 nohup python -u -m rl.ppo \
  --total-timesteps "$STEPS" \
  --num-envs 32 \
  --num-steps 128 \
  --lr "$LR" \
  --ent-coef "$ENT_COEF" \
  --hidden "$HIDDEN" \
  --eval-every 50 \
  --eval-episodes 200 \
  --snapshot-every 100 \
  --device cuda \
  --run-name "$RUN_NAME" \
  > "rl_runs/${RUN_NAME}.log" 2>&1 < /dev/null &
PID=$!

echo "launched phase 3: PID=$PID, GPU=$GPU, lr=$LR ent=$ENT_COEF hidden=$HIDDEN"
echo "log: rl_runs/${RUN_NAME}.log"
echo "will save checkpoints to rl_runs/${RUN_NAME}/checkpoints/"
