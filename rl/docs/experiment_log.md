# Experiment log — Buckshot Roulette RL

A chronological record of every training run + analysis. Each entry:
**hypothesis** → **setup** → **result** → **interpretation** → **next step**.

Keep this file concise; detailed artifacts live under `rl_runs/<run_name>/`
(on beeline-prod) and `rl/docs/` (reports).

---

## Baseline reference point

**Agent used as "the baseline":** `sweep_rr_winner` = `lr=1e-3`,
`ent_coef=0.01`, `hidden=256`, 2M steps, PPO + frozen-league self-play on
the engine WITH the old "1 non-adrenaline item per turn" rule.

**Published numbers (1000 eval episodes per opponent):**

| Opponent | Win rate | 95 % CI |
|---|---|---|
| random | 0.97 | ±0.01 |
| rule:aggressive | 0.81 | ±0.02 |
| rule:conservative | 0.68 | ±0.03 |
| **mean** | **0.82** | |

In pairwise play against other sweep configs the same checkpoint sits at
**60.7 %** mean head-to-head win rate — i.e. it's already near the empirical
ceiling for that engine + hyperparameter family.

Any future intervention is measured **against this baseline**.

---

## E1 — PPO + self-play league, default hyperparameters

| | |
|---|---|
| Date | 2026-04-19 |
| Run dir | `rl_runs/phase1_baseline_5M/` |
| Hypothesis | PPO with the default CleanRL-style config (lr=3e-4, ent=0.01, hidden=256) + a league pool (3 rule-based + up-to-8 training snapshots) will steadily improve win-rate vs. baselines over 5M steps. |
| Setup | 32 envs × 128 rollout; snapshot every 10 updates; 200 eval eps every 5 updates; CUDA, H100 ×1, ~30 min wallclock. |
| Result | Plateaued at u≈50 (~200 k steps). Final 1000-ep eval: random 0.90, aggressive 0.67, conservative 0.60, mean **0.725**. |
| Interpretation | Default lr is too small. Agent learned item use (saw+shoot, heal, cuff-then-kill, inverter-on-known-blank) but refuses `SHOOT_SELF` even in 75 %-blank situations — i.e. entered the "never-shoot-self" local optimum seen in heads-up self-play. |
| Next step | Hyperparameter sweep focusing on lr and entropy. |

---

## E2 — hyperparameter sweep (18 configs × 2M)

| | |
|---|---|
| Date | 2026-04-19 |
| Run dir | `rl_runs/sweep_phase2/` |
| Hypothesis | A broader search over (lr × ent × hidden) will find a config that breaks the E1 plateau. |
| Setup | 3×3×2 grid: lr ∈ {1e-4, 3e-4, 1e-3}, ent ∈ {0.005, 0.01, 0.02}, hidden ∈ {128, 256}. 2M steps each, 4 GPUs parallel. |
| Result | Top by baseline eval: lr=1e-3, ent=0.02, h=256 (mean 0.80). Top 6 all use lr=1e-3; bottom 6 all use lr=1e-4. |
| Interpretation | E1's lr=3e-4 was **5× too small**. The sweep's per-config ceiling is ≈0.80 mean — modest improvement over E1's 0.725, but no config broke through the "never shoot self" pattern. |
| Next step | Head-to-head round-robin to distinguish brittle from robust configs. |

---

## E2b — cross-config round-robin

| | |
|---|---|
| Date | 2026-04-19 |
| Artifact | `/tmp/sweep_phase2/round_robin.json` |
| Result | Best by baseline eval (ent=0.02, h=256) ranked **#6** in head-to-head (mean 0.566). Actual most-robust: ent=0.01, h=256 (0.607). Behavioral diff: ent=0.02 skips handsaw in lethal setups (wastes the saw+shot combo); ent=0.01 keeps it. 1 rock-paper-scissors triple among the lr=1e-4 weaklings. |
| Interpretation | Baseline eval is a biased proxy — it rewards configs tuned for the specific rule-based opponents, not robust policies. Head-to-head is what matters. |
| Next step | Long 50M run on the **baseline-eval** winner to check whether more training breaks the plateau. |

---

## E3 — long run on sweep winner (aborted at u≈5100, 21M/50M steps)

| | |
|---|---|
| Date | 2026-04-19 / 20 |
| Run dir | `rl_runs/phase3_final_50M/` (on beeline-prod) |
| Hypothesis | 10× more training on the best sweep config will finally break through 80 % mean. |
| Setup | lr=1e-3, ent=0.02, hidden=256, 32 envs × 128 rollout, snapshot every 100 updates, 200 eval eps every 50. Engine with the last-shell-blank-self-shot fix applied. |
| Result | Reached **vs_random 91-96 %, vs_aggr 74-81 %, vs_cons 72-79 %** by u=300 (~4 % of planned training) and oscillated in that band for the next 17M steps with no upward trend. Tournament vs sweep finals: `phase3_u5000` 0.681 mean, `sweep_rr_winner` 0.655 — i.e. **10× compute bought 2-3 pp**. |
| Interpretation | More data is not the bottleneck. The plateau is a **valuation** failure: never-shoot-self is self-reinforced by symmetric self-play. We aborted the run and pivoted. |
| Next step | Two parallel tracks: (a) gameplay-fidelity change (remove 1-item-per-turn, C1), and (b) a literature review to pick a real intervention. |

---

## C1 — code change: remove 1-item-per-turn rule

| | |
|---|---|
| Date | 2026-04-20 |
| Commit | e713464 |
| Motivation | The engine inherited a "one non-adrenaline item per turn" restriction from the user's original Telegram bot. The actual Steam game permits chaining arbitrary items (Glass → Saw → Cuff → Shoot). The restriction was collapsing the strategic state space. |
| Change | `legal_actions()` no longer gates non-adrenaline item uses on a per-turn flag; `_apply_item` and `_advance_turn` no longer set or clear the flag. Observation layout unchanged (the flag slot is now always 0) so existing checkpoints still load. |
| Verification | All 38 tests pass. Random-vs-random mean episode length grew from 11.4 → 12.6 steps, consistent with more item use. |
| Expected effect | **Game fidelity improved**, but per literature this does NOT address the plateau — the plateau is about valuation, not expressiveness. |

---

## R1 — literature review (background-agent report)

Artifact: `rl/docs/research_notes.md`.

**Top 3 recommendations, ranked by `impact / difficulty`:**

1. **Main-exploiter agent + meta-Nash opponent sampling** (Vinyals 2019,
   McAleer 2022). Add a dedicated agent whose sole job is to exploit the
   current main; mix with league via meta-Nash weights instead of FIFO.
   Low-medium difficulty; targets the exact failure mode.
2. **Determinised IS-MCTS with PPO prior + value** (Cowling 2012 /
   Schmid 2023). A few-days build using the already-trained value net;
   mechanically fixes the self-shoot decision.
3. **Best-response DQN for exploitability** (Timbers 2022). Diagnostic
   rather than direct fix — gives a quantitative "how big is the hole".

Skipped: transformer/LSTM (obs already summarises POMDP info), full ReBeL
(multi-month), LLM-as-policy (overkill + slow), NFSP (slower than league
on our size).

---

## E4 (planned) — main-exploiter agent

| | |
|---|---|
| Hypothesis | Adding a dedicated main-exploiter to the league will punish the "never shoot self" equilibrium, forcing the main to adopt the EV-correct behaviour. |
| Setup (proposed) | Train main (lr=1e-3, ent=0.01, h=256) for 10M steps. In parallel, a main-exploiter PPO that samples only the current-main as opponent. Opponent pool for main: 70 % rule-based + main-exploiter's best snapshot, 30 % older main snapshots (meta-Nash approximation). |
| Success criteria | `P(SHOOT_SELF)` > 0.3 when `P(blank) ≥ 0.75` in the behavioural probe; head-to-head vs `sweep_rr_winner` > 55 %. |
| Risks | Exploiter might collapse to always-self-shoot → train a main that over-reacts. Mitigation: cap exploiter's weight in main's pool. |

Not yet launched — awaiting user sign-off.
