"""Hyperparameter sweep launcher.

Generates a grid of PPO configs, distributes them across N GPUs (one
worker per GPU; configs run sequentially within a worker), and ranks
results by final eval win-rate.

Usage:
    python -m rl.sweep --gpus 0,1,2,3 --steps-per-config 2000000
    python -m rl.sweep --gpus 0,2,3 --grid quick    # tiny smoke grid

Output:
    sweep_dir/
      manifest.json          # full config -> run_dir mapping
      ranking.json           # final rankings (sorted by score)
      <run_name>/metrics.jsonl   # per-config training metrics
      <run_name>.log         # stdout/stderr capture

After completion, generates a summary table to stdout.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


DEFAULT_GRID = {
    "lr": [1e-4, 3e-4, 1e-3],
    "ent_coef": [0.005, 0.01, 0.02],
    "hidden": [128, 256],
}

QUICK_GRID = {
    "lr": [3e-4],
    "ent_coef": [0.01, 0.02],
    "hidden": [128],
}


def cartesian(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        out.append(dict(zip(keys, vals)))
    return out


def cfg_name(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        if isinstance(v, float):
            parts.append(f"{k}{v:.0e}".replace("+0", "+"))
        else:
            parts.append(f"{k}{v}")
    return "_".join(parts)


def run_config(
    cfg: dict,
    gpu_id: int,
    steps: int,
    sweep_dir: str,
    seed: int,
    snapshot_every: int,
    eval_every: int,
    eval_episodes: int,
) -> dict:
    name = f"sweep_{cfg_name(cfg)}_gpu{gpu_id}"
    log_path = os.path.join(sweep_dir, f"{name}.log")
    cmd = [
        sys.executable, "-u", "-m", "rl.ppo",
        "--total-timesteps", str(steps),
        "--num-envs", "32",
        "--num-steps", "128",
        "--lr", str(cfg["lr"]),
        "--ent-coef", str(cfg["ent_coef"]),
        "--hidden", str(cfg["hidden"]),
        "--seed", str(seed),
        "--device", "cuda",
        "--save-dir", sweep_dir,
        "--snapshot-every", str(snapshot_every),
        "--eval-every", str(eval_every),
        "--eval-episodes", str(eval_episodes),
        "--run-name", name,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return {
        "cfg": cfg,
        "name": name,
        "gpu": gpu_id,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "log_path": log_path,
        "metrics_path": os.path.join(sweep_dir, name, "metrics.jsonl"),
    }


def worker(gpu_id: int, configs: list[dict], steps: int, sweep_dir: str,
           seed: int, snapshot_every: int, eval_every: int, eval_episodes: int) -> list[dict]:
    print(f"[gpu{gpu_id}] starting worker with {len(configs)} configs", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        print(f"[gpu{gpu_id}] {i+1}/{len(configs)} {cfg_name(cfg)} ...", flush=True)
        r = run_config(cfg, gpu_id, steps, sweep_dir, seed, snapshot_every, eval_every, eval_episodes)
        status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
        print(f"[gpu{gpu_id}] done {cfg_name(cfg)} in {r['elapsed_s']:.1f}s [{status}]", flush=True)
        results.append(r)
    return results


def parse_metrics(metrics_path: str) -> dict:
    """Pull the last evaluated win-rates and final loss values."""
    if not os.path.exists(metrics_path):
        return {"error": "metrics file missing"}
    last_eval = None
    last_row = None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            last_row = entry
            if any(k.startswith("eval/winrate_vs_") for k in entry):
                last_eval = entry
    out = {}
    if last_row is None:
        return {"error": "metrics empty"}
    out["last_global_step"] = last_row.get("global_step", 0)
    out["last_return50"] = last_row.get("rollout/mean_return_50", 0.0)
    if last_eval:
        for k, v in last_eval.items():
            if k.startswith("eval/winrate_vs_"):
                out[k] = v
    return out


def rank_results(results: list[dict]) -> list[dict]:
    """Score = mean of eval winrates against the three baselines."""
    scored = []
    for r in results:
        metrics = parse_metrics(r["metrics_path"])
        wrs = [v for k, v in metrics.items() if k.startswith("eval/winrate_vs_")]
        score = float(sum(wrs) / len(wrs)) if wrs else float("-inf")
        scored.append({**r, "metrics": metrics, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def print_table(ranking: list[dict]) -> None:
    print("\n=== sweep ranking (mean win-rate vs baselines) ===")
    print(
        f"{'rank':>4}  {'score':>7}  {'name':<55}  "
        f"{'vs_random':>10}  {'vs_aggr':>9}  {'vs_cons':>9}  {'elapsed':>9}"
    )
    for i, r in enumerate(ranking):
        m = r["metrics"]
        wr_r = m.get("eval/winrate_vs_random", float("nan"))
        wr_a = m.get("eval/winrate_vs_aggressive", float("nan"))
        wr_c = m.get("eval/winrate_vs_conservative", float("nan"))
        print(
            f"{i+1:>4}  {r['score']:>7.3f}  {r['name']:<55}  "
            f"{wr_r:>10.3f}  {wr_a:>9.3f}  {wr_c:>9.3f}  {r['elapsed_s']:>8.1f}s"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    p.add_argument("--steps-per-config", type=int, default=2_000_000)
    p.add_argument("--grid", choices=["default", "quick"], default="default")
    p.add_argument("--sweep-dir", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=100)
    a = p.parse_args()

    gpus = [int(g) for g in a.gpus.split(",")]
    grid = DEFAULT_GRID if a.grid == "default" else QUICK_GRID
    configs = cartesian(grid)
    if a.sweep_dir is None:
        a.sweep_dir = os.path.join("rl_runs", f"sweep_{int(time.time())}")
    os.makedirs(a.sweep_dir, exist_ok=True)

    # Round-robin distribute configs across GPUs
    per_gpu: dict[int, list[dict]] = {g: [] for g in gpus}
    for i, cfg in enumerate(configs):
        per_gpu[gpus[i % len(gpus)]].append(cfg)

    manifest = {
        "sweep_dir": a.sweep_dir,
        "gpus": gpus,
        "steps_per_config": a.steps_per_config,
        "grid": grid,
        "configs": configs,
        "per_gpu": {str(g): [cfg_name(c) for c in cfgs] for g, cfgs in per_gpu.items()},
    }
    with open(os.path.join(a.sweep_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sweep] {len(configs)} configs across {len(gpus)} GPUs ({a.gpus}), "
          f"{a.steps_per_config} steps each, dir={a.sweep_dir}")

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [
            ex.submit(
                worker, g, per_gpu[g], a.steps_per_config, a.sweep_dir,
                a.seed, a.snapshot_every, a.eval_every, a.eval_episodes,
            )
            for g in gpus
        ]
        for f in futures:
            results.extend(f.result())
    total = time.time() - t0
    print(f"[sweep] all {len(results)} configs done in {total:.1f}s")

    ranking = rank_results(results)
    with open(os.path.join(a.sweep_dir, "ranking.json"), "w") as f:
        json.dump([{**r, "elapsed_s": float(r["elapsed_s"])} for r in ranking], f, indent=2, default=str)
    print_table(ranking)
    return 0


if __name__ == "__main__":
    sys.exit(main())
