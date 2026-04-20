#!/usr/bin/env bash
# E10: escape actor entropy collapse observed in E9.
# - ent_coef 0.01 → 0.05 (keep exploration floor)
# - scenario_replay_prob 0.15 → 0.30 (more s* visits per batch)
# - --no-anneal-lr (LR stays at 3e-4, gradients stay active to end)
# - hidden=256, γ=0.999, 3M steps (matches E9 compute budget)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate buckshot-rl
cd ~/buckshot-roulette
export CUDA_VISIBLE_DEVICES=1
python -m rl.ppo \
    --total-timesteps 3000000 \
    --num-envs 16 \
    --gamma 0.999 \
    --lr 3e-4 \
    --ent-coef 0.05 \
    --no-anneal-lr \
    --scenario-replay-prob 0.30 \
    --run-name E10_escape_collapse \
    --device cuda \
    2>&1 | tee /home/a_kravchenko/e10.log
