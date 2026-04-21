# PPO Throughput: Async Vector Envs Plan

**Status:** designed 2026-04-21 (during E19 training), not yet implemented.
**Target:** land as a single coherent PR after E19 finishes, before E20.
**Expected speedup:** 4–6× wall-clock end-to-end; H100 utilization from <10% to 30–50%.

## 1. Recommendation

Use `gymnasium.vector.AsyncVectorEnv` with `context="spawn"` and
`shared_memory=True`. Drop-in replacement for `SyncVectorEnv`: same
`reset(seed=...)` / `step(actions)` / `close()` surface, same info-dict
aggregation, same observation-space dict support. Fork-based workers
break CUDA (training policy is on H100) — **spawn is mandatory**.
Shared memory halves the obs-dict round-trip vs pickling every step.

Rejected alternatives:
- `stable-baselines3 SubprocVecEnv` — duplicates gym's abstraction, extra dep.
- Just scaling `num_envs` on `SyncVectorEnv` — still one CPU; per-step
  ~50–150 µs × 32 envs ≈ 3–5 ms dominates H100 forward (~0.5 ms).
- Full hybrid policy-eval-in-main — too-large rewrite; fallback only.

## 2. Sharp Edges

### 2a. Opponent-pool coherence (biggest risk)

Subprocess envs pickle a pool at `make_env_fn` time and will never see
subsequent `pool.add`/`remove` calls in the parent.

**Chosen approach: snapshot-file + reload-on-signal.** Parent already
writes `run_dir/checkpoints/snapshot_uN.pt`. After `pool.add`+`pool.remove`,
parent calls `envs.call("sync_pool", pool_manifest)` where
`pool_manifest = list[(name, "rule:<fn_name>") | (name, "ckpt:<path>")]`.
Worker's env exposes a method that rebuilds its local pool from the
manifest (rule-based from `NAMED_OPPONENTS`, checkpoints via
`load_recurrent_policy` + `make_frozen_recurrent_opponent_factory`).
`AsyncVectorEnv.call(method_name, *args)` is the supported broadcast API.

Rejected alternatives:
- Periodic re-spawn (throw-away + rebuild). Costs ~1–2 s spawn per refresh.
- Push `state_dict` via queue every 10 updates. Checkpoint file already
  exists, so strict subset of the chosen approach.
- Hybrid eval-in-main. Too big — punt to fallback.

### 2b. Recurrent opponent state & pickling

`make_frozen_recurrent_opponent_factory` returns a closure over a
`RecurrentActorCritic`. Python closures pickle fine iff captured
objects pickle fine; `nn.Module` pickles via `torch.save` semantics.

**With spawn:** modules re-import rather than pickle, so closures
captured at function scope must be picklable. `policy` and `device`
both are. `_FrozenRecurrentOpponent._h` is lazily initialized
per-instance inside a worker, no sharing. `_is_factory = True`
attribute survives pickling.

**Gotcha:** worker-side snapshots MUST be CPU. In the manifest-reload
path, force `device="cpu"` for worker-side opponents. Env stepping
on CPU is the whole point.

### 2c. Seeding determinism

Gymnasium's contract: pass a list-sized seed to the top-level
`reset(seed=seed_list)`, OR use `AsyncVectorEnv(..., seed=...)` at
construction. Current `make_env_fn(pool, cfg.seed + i, ...)` seeds
the env-fn closure; `envs.reset(seed=cfg.seed)` seeds the scalar.

For Async: each sub-env gets `cfg.seed + i` at construction, and
top-level `reset(seed=S)` propagates `[S, S+1, ..., S+N-1]` to workers.
Document with a comment that we rely on gymnasium's per-worker seed
offsetting. The per-env `_opp_rng = np.random.default_rng(eff_seed + 1)`
in `SingleAgentBuckshotEnv.reset` remains correct — each worker's
RNG is process-local.

### 2d. Info-dict aggregation

`AsyncVectorEnv` uses identical dict-of-arrays aggregation to
`SyncVectorEnv`. `_aux_from_infos` and `_opp_ids_from_infos` already
use `infos.get(key, default)` with fallback. Termination order:
`step_wait` gathers in a fixed per-worker order (not completion
order), so indexing by env-index is stable. `ppo_recurrent.py`
never relies on termination ordering — indexes by env-index
throughout, `record_terminal_returns` iterates `done` by index.
**No code change needed.**

### 2e. Scenario replay & opponent RNG

`np.random.default_rng(eff_seed + 1)` is process-local. The per-env
`_opp_rng` lives inside each worker's `SingleAgentBuckshotEnv`, no
cross-worker state. Python's global `random` is not used in the env
(confirmed — everything uses numpy's `Generator`). **Fine as-is.**

### 2f. SIGINT / shutdown

`AsyncVectorEnv.close()` sends a close command and `join`s workers
with a timeout; on timeout it `terminate()`s them. Current
`finally: envs.close()` handles this. Gotcha: on SIGINT the parent
may receive the signal mid-worker-step, leaving the pipe in an odd
state. **Add `envs.close(terminate=True)`** in the `finally`. Also
set `daemon=True` on the spawn context if not default so orphan
workers die with the parent.

## 3. Suggested Config

H100 + 32 vcores:
- `num_envs = 32` (one env per worker — current gymnasium runs
  one-per-process). Verify with `gymnasium.__version__`.
- Start with `num_envs = 32, num_steps = 256` — same batch size as
  current 32×128 = 4096, doubled to 8192. Larger batch → fewer
  update passes hitting CPU round-trip.
- `num_minibatches = 4` unchanged (must divide num_envs).

Back-of-envelope:
- Current: CPU-limited, ~32 envs × ~100 µs/step × 128 steps ≈ **0.4 s/rollout**.
- Async: 32 workers parallel: ideal ~13 ms, realistic ~50 ms after
  IPC (shared-mem obs + pickle for action arrays, ~2–5 ms/step × 128).
- **Expected 5–8× rollout speedup**, PPO-update overhead unchanged.
- **End-to-end ~4–6× wall clock.** H100 utilization jumps from
  <10% to 30–50% (now GPU-bound on forward, not env-bound).

## 4. Scope Gates (do NOT do in this PR)

- Do NOT change policy architecture or hidden size.
- Do NOT touch PPO hyperparameters (lr, clip, GAE, minibatches).
- Do NOT rewrite `OpponentPool` internals — only add a `manifest()`
  method and a worker-side `sync_pool(manifest)` env method.
- Do NOT change reward-shaping, scenario-replay, or curriculum knobs.
- Do NOT move training policy to non-CUDA or re-batch league snapshots
  GPU-side.
- Do NOT port `ppo.py` (FF) — recurrent-only; FF is maintenance mode.
- Do NOT add vectorized opponent evaluation (hybrid-batch path —
  separate PR).

**Single PR:** AsyncVectorEnv switch + pool-sync plumbing + config
tweaks + one CHANGELOG line in experiment_log.md. Nothing else.

## 5. Fallback (if Async is flaky)

**Lane 2:** 4 parallel `SyncVectorEnv` processes using
`torch.multiprocessing.Process`, 8 envs each, rollouts gathered via
shared-memory tensors. Gets ~3–4× without depending on gymnasium's
Async internals, trivially debuggable. Pool sync by file-checkpoint
polling (workers re-read `run_dir` on mtime change).

If `AsyncVectorEnv` hits issues with:
- **Pickling** `_FrozenRecurrentOpponent` closures or `obs_layout`
  attributes on bound functions: fall back, use manual
  `torch.multiprocessing` pool instead — you control the payload.
- **Shared-memory obs dicts**: disable `shared_memory=True`
  (costs ~20% throughput, still 3–5× faster than Sync).
- **Deadlocks on close**: wrap with an `atexit` handler that sends
  SIGKILL to worker PIDs.

**80/20 worst-case:** two training jobs on the same GPU with
`num_envs=16` each, seeds offset. Doubles aggregate sample throughput
at the cost of sharing H100 VRAM. Zero code change.

## Implementation Order

1. Read current `AsyncVectorEnv` API in the installed `gymnasium`
   version — confirm `call()` broadcasting and `seed` semantics.
2. Add `OpponentPool.manifest()` → `list[(name, "rule:<fn_name>" | "ckpt:<path>")]`;
   add reverse-loader helper.
3. Add `SingleAgentBuckshotEnv.sync_pool(manifest, device="cpu")`
   method for worker-side reload.
4. In `ppo_recurrent.train`: swap `SyncVectorEnv` for
   `AsyncVectorEnv(..., context="spawn", shared_memory=True)`;
   keep signature otherwise.
5. After `pool.add`/eviction block, call
   `envs.call("sync_pool", pool.manifest())`.
6. Update `finally: envs.close(terminate=True)`.
7. Smoke test: 200k steps locally first (spawn works on mac/linux),
   then 1M on H100.

## Critical Files

- `rl/ppo_recurrent.py` — swap env class, add sync_pool broadcast
- `rl/single_agent_env.py` — add `sync_pool` method
- `rl/opponents.py` — add `manifest()`; `NAMED_OPPONENTS` lookup already fine
- `rl/policy.py` — no change, `load_recurrent_policy` already handles
  aux_dim + n_opponents auto-detect
- `rl/ppo.py` — minor: check `make_env_fn` is spawn-safe (no unpicklable
  closures beyond pool+seed)
