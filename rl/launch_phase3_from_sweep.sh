#!/bin/bash
# Auto-select best config from sweep_phase2/ranking.json and launch Phase 3.
# Requires ranking.json to exist (sweep finished).
#
# Usage:
#   ./rl/launch_phase3_from_sweep.sh [GPU] [STEPS]
#
# Examples:
#   ./rl/launch_phase3_from_sweep.sh           # default: GPU 1, 50M steps
#   ./rl/launch_phase3_from_sweep.sh 3 10000000

set -euo pipefail

GPU="${1:-1}"
STEPS="${2:-50000000}"
SWEEP_DIR="rl_runs/sweep_phase2"
RUN_NAME="phase3_final_${STEPS}"

cd "$(dirname "$0")/.."

if [ ! -f "$SWEEP_DIR/ranking.json" ]; then
    echo "ERROR: $SWEEP_DIR/ranking.json not found. Has the sweep finished?"
    exit 1
fi

# Extract top config using python (conda env buckshot-rl must be active)
read LR ENT HIDDEN NAME < <(python -c "
import json
with open('$SWEEP_DIR/ranking.json') as f:
    ranking = json.load(f)
top = ranking[0]
cfg = top['cfg']
print(cfg['lr'], cfg['ent_coef'], cfg['hidden'], top['name'])
")

echo "Best sweep config: $NAME"
echo "  lr=$LR  ent_coef=$ENT  hidden=$HIDDEN"
echo "  score=$(python -c "import json; print(json.load(open('$SWEEP_DIR/ranking.json'))[0]['score'])")"
echo ""
echo "Launching Phase 3: $STEPS steps on GPU $GPU (run_name=$RUN_NAME)"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 nohup python -u -m rl.ppo \
  --total-timesteps "$STEPS" \
  --num-envs 32 \
  --num-steps 128 \
  --lr "$LR" \
  --ent-coef "$ENT" \
  --hidden "$HIDDEN" \
  --eval-every 50 \
  --eval-episodes 200 \
  --snapshot-every 100 \
  --device cuda \
  --run-name "$RUN_NAME" \
  > "rl_runs/${RUN_NAME}.log" 2>&1 < /dev/null &
PID=$!

echo "PID=$PID"
echo "log: rl_runs/${RUN_NAME}.log"
echo "run:  rl_runs/${RUN_NAME}/"
