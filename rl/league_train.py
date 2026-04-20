"""Generational population-based league training.

Implements Option A from the mid-session experiment plan: train N agents
with different entropy coefficients in parallel, and between generations
seed each agent's opponent pool with the final policies from every agent
of the previous generation. Expected to collapse the combo-master vs
info-gatherer strategic dichotomy into a single stronger agent that can
beat all archetypes.

Structure:
  Gen 1: 4 agents vs rule-based only.
  Gen 2: 4 agents vs rule-based + 4 finals from Gen 1.
  Gen 3: 4 agents vs rule-based + 4 finals from Gen 2.
  ...

Each agent's ent_coef stays fixed across generations to preserve
archetype diversity; only the pool they train against evolves.

Usage:
    python -m rl.league_train --gpus 0,1,2,3 --steps-per-gen 3000000 \
        --generations 3 --sweep-dir rl_runs/league_v1
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


AGENTS = [
    {"id": "A_ent005", "ent_coef": 0.005, "lr": 1e-3, "hidden": 256, "seed_base": 1000},
    {"id": "B_ent010", "ent_coef": 0.010, "lr": 1e-3, "hidden": 256, "seed_base": 2000},
    {"id": "C_ent020", "ent_coef": 0.020, "lr": 1e-3, "hidden": 256, "seed_base": 3000},
    {"id": "D_ent030", "ent_coef": 0.030, "lr": 1e-3, "hidden": 256, "seed_base": 4000},
]


def agent_run_name(agent_id: str, gen: int) -> str:
    return f"{agent_id}_gen{gen}"


def agent_final_path(sweep_dir: str, agent_id: str, gen: int) -> str:
    return os.path.join(sweep_dir, agent_run_name(agent_id, gen), "policy_final.pt")


def run_one(agent: dict, gen: int, gpu_id: int, steps: int, sweep_dir: str,
            extra_opponent_ckpts: list[str], snapshot_every: int,
            eval_every: int, eval_episodes: int) -> dict:
    name = agent_run_name(agent["id"], gen)
    log_path = os.path.join(sweep_dir, f"{name}.log")
    extras_str = ",".join(extra_opponent_ckpts) if extra_opponent_ckpts else ""
    cmd = [
        sys.executable, "-u", "-m", "rl.ppo",
        "--total-timesteps", str(steps),
        "--num-envs", "32",
        "--num-steps", "128",
        "--lr", str(agent["lr"]),
        "--ent-coef", str(agent["ent_coef"]),
        "--hidden", str(agent["hidden"]),
        "--seed", str(agent["seed_base"] + gen),
        "--device", "cuda",
        "--save-dir", sweep_dir,
        "--snapshot-every", str(snapshot_every),
        "--eval-every", str(eval_every),
        "--eval-episodes", str(eval_episodes),
        "--run-name", name,
    ]
    if extras_str:
        cmd.extend(["--extra-opponent-ckpts", extras_str])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return {
        "agent": agent["id"],
        "gen": gen,
        "gpu": gpu_id,
        "run_name": name,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "log_path": log_path,
        "final_ckpt": agent_final_path(sweep_dir, agent["id"], gen),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU ids")
    p.add_argument("--steps-per-gen", type=int, default=3_000_000)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--sweep-dir", default=None)
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=100)
    a = p.parse_args()

    gpus = [int(g) for g in a.gpus.split(",")]
    if len(gpus) < len(AGENTS):
        print(f"WARNING: {len(AGENTS)} agents but only {len(gpus)} GPUs — will queue.")

    if a.sweep_dir is None:
        a.sweep_dir = os.path.join("rl_runs", f"league_{int(time.time())}")
    os.makedirs(a.sweep_dir, exist_ok=True)

    manifest = {
        "sweep_dir": a.sweep_dir,
        "gpus": gpus,
        "steps_per_gen": a.steps_per_gen,
        "generations": a.generations,
        "agents": AGENTS,
    }
    with open(os.path.join(a.sweep_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[league] {len(AGENTS)} agents × {a.generations} gens × {a.steps_per_gen} steps "
          f"on GPUs {gpus}, dir={a.sweep_dir}")

    all_results: list[dict] = []
    prev_gen_finals: list[str] = []

    for gen in range(1, a.generations + 1):
        print(f"\n=== Generation {gen} / {a.generations} ===")
        if prev_gen_finals:
            print(f"  opponent seeds from gen {gen-1}: {len(prev_gen_finals)} ckpts")

        # Dynamically lease a GPU per task from a queue to avoid
        # oversubscription when len(AGENTS) > len(gpus). Pre-assigning
        # gpu = gpus[i % len(gpus)] would let two tasks for the same
        # GPU run concurrently if an earlier GPU happens to free up first.
        gpu_pool: queue.Queue[int] = queue.Queue()
        for g in gpus:
            gpu_pool.put(g)

        def run_with_gpu_lease(agent: dict) -> dict:
            gpu = gpu_pool.get()
            print(f"  [gen{gen}/{agent['id']} gpu{gpu}] launching ...")
            try:
                return run_one(
                    agent, gen, gpu, a.steps_per_gen, a.sweep_dir,
                    prev_gen_finals, a.snapshot_every, a.eval_every, a.eval_episodes,
                )
            finally:
                gpu_pool.put(gpu)

        with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            futures = [ex.submit(run_with_gpu_lease, agent) for agent in AGENTS]
            gen_results = [f.result() for f in futures]
        for r in gen_results:
            status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
            print(f"  [gen{r['gen']}/{r['agent']}] {r['elapsed_s']:.1f}s [{status}]")
        all_results.extend(gen_results)

        # Collect this generation's finals for the next generation's pool
        prev_gen_finals = [r["final_ckpt"] for r in gen_results if r["returncode"] == 0]
        if not prev_gen_finals and gen > 1:
            print(f"[league] WARNING: all agents in gen {gen} failed; aborting.")
            break

    # Save runs index
    with open(os.path.join(a.sweep_dir, "runs.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[league] done. runs index -> {a.sweep_dir}/runs.json")
    print("Next: cross-generation round-robin via `python -m rl.round_robin --sweep-dir ...`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
