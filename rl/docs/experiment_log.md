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

## E4 — population-based league (4 ent_coefs × 3 generations)

**Context shift before launch**: the CFR baseline on simple_buckshot showed
that SHOOT_OPPONENT ≈ Nash for most states, so the original E4 plan
(main-exploiter punishing "never shoot self") was superseded. The real
gap exposed by the expanded 22-scenario probe was the **combo-master vs
info-gatherer archetype split** — ent=0.01 committed to saw+shoot
combos, ent=0.02 stalled on phone/glass, neither integrated both.

| | |
|---|---|
| Hypothesis | Training 4 ent_coefs simultaneously and seeding each gen's opponent pool with the previous gen's finals will cross-pollinate strategies, merging the combo-master and info-gatherer archetypes into a single stronger agent. |
| Setup | 4 agents × 3 generations × 2M steps = 24M total gradient steps. ent_coef ∈ {0.005, 0.01, 0.02, 0.03}, lr=1e-3, hidden=256. Gen N+1's pool = rule-based + 4 finals from gen N. 4× H100, ~33 min wall time. |
| Success criteria | (a) Champion beats best sweep agent >55% head-to-head; (b) champion handles ≥1 new scenario class (survival items) that prior agents missed. |
| Directory | `rl_runs/league_v1/` (12 finals, round-robin.json, per-agent analysis.md) |

### Result

**Champion: A_ent005_gen2** (mean WR 0.531 over 11 opponents × 300 eps each).

Full ranking (12 finals, 12×11 round-robin):

| Rank | Agent | Mean WR |
|---|---|---|
| 1 | A_ent005_gen2 | 0.531 |
| 2 | B_ent010_gen2 | 0.530 |
| 3 | A_ent005_gen1 | 0.528 |
| 4 | D_ent030_gen1 | 0.526 |
| 5 | B_ent010_gen1 | 0.525 |
| 6 | C_ent020_gen1 | 0.524 |
| 7 | C_ent020_gen3 | 0.522 |
| 8 | C_ent020_gen2 | 0.520 |
| 9 | A_ent005_gen3 | 0.518 |
| 10 | D_ent030_gen2 | 0.518 |
| 11 | D_ent030_gen3 | 0.517 |
| 12 | B_ent010_gen3 | 0.511 |

**No rock-paper-scissors cycles above 0.55** (ranking is a clean total order).

### Interpretation

- **Partial success (a)**: 22-scenario probe shows all 4 gen-3 agents now
  execute the saw/cuff/adrenaline combos that were archetype-split before
  — `saw_when_known_live` 0.94–0.98, `cuff_saw_combo` 0.66–0.93,
  `adrenaline_steal_smoke_when_low` 1.00 across all four. Cross-pollination
  worked for offensive combos.
- **Failure (b)**: `beer_when_certain_death_next_shot` and
  `inverter_save_from_known_live` — at 1HP with a known-live next shell —
  all 4 agents still `SHOOT_OPPONENT` (0.95–1.00) instead of using the
  survival item. The population did not discover self-preservation.
- **Ceiling effect**: 2pp spread across 12 agents (0.511 → 0.531). Gen 3
  is not systematically stronger than gen 1; in fact **gen 1 agents occupy
  ranks 3, 4, 5, 6**. 24M extra steps of league training produced almost
  no net gain.
- **No cycles**: unlike classic StarCraft-style league dynamics, our
  population is consistent — there's a single direction of strength. This
  suggests the game is sufficiently symmetric that diverse ent_coefs all
  converge to variants of the same (suboptimal) strategy.

### Next step

The plateau is **value-function myopia**, not exploration. The agent
valuates "shoot opp for 50% chance to kill" higher than "use beer to
guarantee surviving the next shot", because the reward signal (sparse,
end-of-episode) doesn't differentiate survival turns from wasted turns.

Candidate experiments (not yet launched):
1. **Survival-sensitive exploiter**: train a PPO agent whose opponent is
   the league champion, rewarded +1 for killing champion OR surviving 10
   shells with HP. If it discovers beer-survival, fold back into main
   pool.
2. **Extend horizon via value-bootstrap**: increase `gamma` from 0.99 to
   0.995 or 0.999 to make the value function care more about future turns.
3. **Opponent modelling / belief-based**: give the agent a learned prior
   over opponent's HP/items (the current obs has these fields but they're
   just raw features). Likely a bigger architecture change.

---

## E5 — dense HP-delta reward shaping (α = 0.05)

| | |
|---|---|
| Date | 2026-04-20 |
| Run dir | `rl_runs/E5_hp_shaping/` |
| Hypothesis | A per-step shaping reward `α·(Δhp_me − Δhp_opp)` would densify the sparse ±1 terminal signal enough for the agent to learn that BEER / INVERTER preserve HP in lethal-shot states. |
| Setup | 3M steps, lr=1e-3, ent=0.01, hidden=256, α=0.05; `--extra-opponent-ckpts` = all 4 A/B/C/D gen-2 champions (full league pool: 4 rule + up-to-8 self snapshots + 4 league finals = 12). ~16 min on 1× H100. |
| Result | Training converged cleanly (final lr ~1e-6, KL ~2e-7). Mean WR vs 12-agent league = **0.517** (vs champion `A_ent005_gen2` baseline 0.531; SE ≈ 0.014 across 12×200 eps → inside noise). |

### Behavioral probe: E5 vs `A_ent005_gen2`

22 scenarios; only differences shown.

| Scenario | Champion argmax | E5 argmax | Same? |
|---|---|---|---|
| `beer_when_certain_death_next_shot` | `SHOOT_OPPONENT` (1.00) | `SHOOT_OPPONENT` (1.00) | ✗ identical failure |
| `inverter_save_from_known_live` | `SHOOT_OPPONENT` (1.00) | `SHOOT_OPPONENT` (1.00) | ✗ identical failure |
| `handcuff_then_lethal` | `USE_HANDCUFF` (0.79) | `USE_HANDCUFF` (0.98) | ✓ sharper |
| `inverter_known_blank` | `SHOOT_SELF` (1.00) | `SHOOT_SELF` (0.69) | ◇ noisier |
| `pills_when_desperate` | `USE_PILLS` (0.99) | `USE_PILLS` (1.00) | ✗ same wrong (pills = 60% suicide) |
| *(all other 17 scenarios)* | same | same | — |

### Interpretation

HP-shaping α=0.05 did **not** change behavior on the survival-item
scenarios. The mechanism fails for a structural reason: `SHOOT_OPPONENT`
on a known-live shell gives **immediate** `Δhp_opp = −1` (positive shaped
reward), while `USE_BEER` / `USE_INVERTER` give **zero** `Δhp` this step —
their value is the *counterfactual* avoided damage next turn, which the
shaping term never observes. So α amplifies the agent's offensive bias
rather than breaking it.

The deeper issue is **experience, not reward**: P(agent at 1HP ∧ next
shell known live ∧ has BEER) is exponentially rare across natural
episode dynamics. In 3M steps the agent likely never faced this exact
configuration, so whatever reward signal we attached, there's no
gradient to shape.

### Next step → E6

Low-HP curriculum. Force `hp_start = 1` in ~20% of resets (and `hp_start
= 2` in another 10%) to multiply the frequency of defensive states by
~10×. Keep α=0.05 shaping so that once the agent *is* in a low-HP state,
the (small) incentive to preserve HP is present. This targets the
exploration gap directly rather than trying to increase the shaping
signal until it dominates correct play.

---

## E6 — low-HP curriculum + α=0.05 shaping

| | |
|---|---|
| Date | 2026-04-20 |
| Run dir | `rl_runs/E6_low_hp_curriculum/` |
| Hypothesis | α=0.05 HP shaping *plus* `low_hp_prob=0.3` curriculum (agent's HP clamped to 1 in 30% of resets) would give the agent enough experience in defensive-item states to learn BEER/INVERTER usage. |
| Setup | 3M steps; same hyperparams as E5; `low_hp_prob=0.3`, `hp_shaping=0.05`; 4 league champions in opponent pool. ~16 min on 1× H100. |
| Result | Mean WR vs 12-agent league = **0.495** (slightly *below* E5's 0.517 and champion's 0.531). Training converged normally. |

### Behavioral probe: E6 vs E5 vs champion

| Scenario | Champion | E5 | E6 | Notes |
|---|---|---|---|---|
| `beer_when_certain_death_next_shot` | SHOOT_OPPONENT 1.00 | SHOOT_OPPONENT 1.00 | SHOOT_OPPONENT 1.00 | same failure |
| `inverter_save_from_known_live` | SHOOT_OPPONENT 1.00 | SHOOT_OPPONENT 1.00 | SHOOT_OPPONENT 1.00 | same failure |
| `handsaw_lethal` | correct (0.97) | correct | correct (0.72) | *offensive combo degraded* |
| `glass_when_uncertain` | correct (1.00) | correct | correct (0.64) | *info-gather weakened* |

Survival failures **identical**. Two previously-strong offensive/info
scenarios got noisier. Curriculum + shaping together made the overall
policy slightly *worse*.

### Interpretation

The curriculum *did* produce experience in 1HP states — the agent now
sees them on ~30% of resets vs. <1% natively — but the policy still
shoots. Why:

The HP-shaping term α=0.05 accumulates across a 15-20 step episode.
Worst-case |Δhp| per step = 2 (lethal shot under damage_mult=2), typical
magnitudes ≈ 1. Over an episode the shaped return can reach ±0.75 to
±1.5 — comparable to or larger than the terminal ±1.0 that actually
encodes "won / lost". So the terminal signal no longer *dominates*:
winning by dealing cumulative damage along the way yields nearly the
same reward as winning by surviving. The agent optimizes shaped
cumulative reward and never receives a clear gradient that "losing
because you didn't BEER" is worse than "losing while still aggressive".

That matches the degraded probes: shaping pulled the agent toward
higher average Δhp-per-step, which rewards myopic aggression at the
expense of combos (handsaw sequence) and info plays (glass, whose
"reward" is not Δhp at all).

### Next step → E7

**Remove shaping entirely** (α=0), keep curriculum, raise γ from 0.995
to 0.999 for longer-horizon credit assignment so the terminal ±1 can
propagate to early-episode decisions. This is the cleanest test of
whether the curriculum alone, with a purely sparse terminal signal, is
enough to teach survival.

---

## E7 — curriculum-only + γ=0.999 (no shaping)

| | |
|---|---|
| Date | 2026-04-20 |
| Run dir | `rl_runs/E7_curriculum_only/` |
| Hypothesis | E5/E6 added shaping that diluted the terminal signal. Strip shaping back to zero, keep the low-HP curriculum, raise γ 0.995 → 0.999 so the sparse ±1 can credit-assign across long episodes. If the survival blindspot is purely a credit-assignment problem, longer horizons should fix it. |
| Setup | 3M steps; lr=1e-3; ent=0.01; hidden=256; `low_hp_prob=0.3`; `hp_shaping=0.0`; γ=0.999; 4 league champions in pool. ~16 min on 1× H100. |
| Result | Final policy WR vs heuristics: random 0.94, aggressive 0.84, conservative 0.75. Terminal `mean_return_50 = -0.08` (regressed from peak +0.20 at u=625). Internal round-robin: `policy_final` ranks **5th of 9** snapshots (mean_wr=0.539); peak is `snapshot_u625` (0.573). Three rock-paper-scissors triples detected. |

### Behavioral probe: survival failures persist

| Scenario | E5 / E6 | E7 | V-estimate at state |
|---|---|---|---|
| `beer_when_certain_death_next_shot` | SHOOT_OPPONENT 1.00 | SHOOT_OPPONENT 1.00 | **−0.837** |
| `inverter_save_from_known_live` | SHOOT_OPPONENT 1.00 | SHOOT_OPPONENT 1.00 | **−0.712** |
| `pills_when_desperate` | USE_PILLS 1.00 | USE_PILLS 1.00 | −0.586 |
| `pills_vs_sure_kill` | correct | SHOOT_OPPONENT 1.00 | +0.967 |
| `cuff_saw_combo` | correct | USE_HANDSAW 0.79 → SHOOT_OPP 0.08 | +0.985 |

The critic now correctly assigns large negative value to 1HP+known-live
states (E5/E6's V-estimates were closer to 0). γ=0.999 *did* propagate
the loss signal back. But the **policy still picks SHOOT_OPPONENT** —
i.e., the agent *knows* it's losing in this state but still picks the
self-kill action with p=1.00.

### Interpretation: this is policy collapse, not credit assignment

The actor and critic disagree consistently in the survival scenarios:
critic says V ≈ −0.7 (almost-certain loss), actor still puts ~all mass
on SHOOT_OPPONENT. The standard PPO update can't fix this:

1. The action is *masked-legal*, so it stays in the support.
2. Past on-policy rollouts almost never selected USE_BEER /
   USE_INVERTER in this exact state (P(visit) ≈ 0 even with curriculum).
3. With no on-policy data for the alternative, the advantage estimate
   for USE_BEER is essentially noise — sometimes negative, sometimes
   positive. Over the full update window, gradients pull toward the
   *visited* action regardless of its expected return, because that's
   the one with consistent advantage signal.

This is the **mode collapse** failure mode: once entropy is annealed
(ent_coef=0.01 from u≥100) and the policy commits to SHOOT_OPPONENT in
this state, no future rollout will ever pick BEER, so the actor cannot
update toward it. Curriculum brought the *state* into the visit
distribution; it did not bring the *good action in that state* into the
visit distribution.

### Next step → E8 (user-requested) and E9 (likely real fix)

E8 will test the user's 4-component asymmetric shaping (damage / heal /
round-survive bonuses). Hypothesis: small dense rewards for *the right
behavior* (e.g., heal_bonus when HP restored) might give USE_SMOKE /
USE_BEER positive immediate reward they currently lack. Calibrated so
shaped budget ≈ 0.3-0.5 per win << terminal ±1.

But the deeper fix is **forcing visits to the alternative action**.
That's E9: a scenario-replay buffer that resets ~15% of episodes
directly into "agent at 1HP + known-live + has BEER" states and lets
on-policy exploration produce the contrast (BEER survives → wins;
SHOOT_OPPONENT here → instant -1). Until USE_BEER appears in rollouts
with non-zero frequency in this state, the actor cannot learn to
prefer it regardless of reward shape.

---

## E8 — user-proposed 4-component shaping (damage + heal + survive)

| | |
|---|---|
| Date | 2026-04-20 |
| Run dir | `rl_runs/E8_4comp_shaping/` |
| Hypothesis | User proposal: big + for win, medium + for damage dealt (β=0.05), medium + for HP restored (γ=0.10), small + for round survived (δ=0.15), big − for loss. Terminal ±1 still dominates (shaped budget calibrated to ≈0.3-0.5 per episode). The *asymmetric* structure might give dense reward for the *right* behavior (e.g., heal on SMOKE) without the accumulating bias E5/E6 had. |
| Setup | 3M steps; lr=1e-3; ent=0.01; hidden=256; γ=0.999; shaping β=0.05, γ=0.10, δ=0.15; 4 league champions in pool. ~25 min on 1× H100. |
| Result | Final policy WR vs heuristics: random 0.94, aggressive 0.80, conservative 0.72. Mean `return_50` near terminal ≈ +1.0 (shaped). Internal round-robin: **monotone, no RPS cycles** (unlike E7). policy_final ranks #3 of 9, peak is snapshot_u1250. |

### Behavioral probe: survival blindspot persists

| Scenario | E7 | E8 | V-estimate change |
|---|---|---|---|
| `beer_when_certain_death_next_shot` | SHOOT_OPP 1.00 (V=−0.84) | SHOOT_OPP 1.00 (V=**−0.54**) | softer V |
| `inverter_save_from_known_live` | SHOOT_OPP 1.00 (V=−0.71) | SHOOT_OPP 1.00 (V=**−0.50**) | softer V |
| `pills_when_desperate` | USE_PILLS 1.00 | USE_PILLS 1.00 | same |
| `cuff_saw_combo` | HANDSAW 0.79 / HANDCUFF 0.13 | **HANDCUFF 0.70** / HANDSAW 0.28 | *fixed* — plays the correct order |
| `smoke_when_low_hp` | correct (1.00) | correct (1.00) | same |
| `handsaw_lethal` | correct | correct | same |

### Interpretation

E8 improved **training stability** (monotone round-robin, no RPS
dynamics) and **combo play** (`cuff_saw_combo` now prefers the correct
cuff-first sequence — heal/survive bonuses apparently made the critic
value longer tactical sequences). The critic's V estimates for s*
states got less negative (−0.84 → −0.54), suggesting the shaping
sprinkled some signal that the state isn't a *certain* loss.

But — consistent with the E7 post-mortem — the **policy still picks
SHOOT_OPPONENT with p=1.00** in the two survival scenarios. No amount
of asymmetric dense reward fixed the visit-distribution problem: since
USE_BEER / USE_INVERTER were never sampled in rollouts in these exact
states, their advantage estimate stays at 0 (noise). heal_bonus only
helps *once the agent already picks USE_SMOKE*; it does nothing when
the actor puts p=0 on the heal action.

This is the predicted result from the E7 post-mortem: mode collapse
needs a *visit-forcing* mechanism, not a reward reshaping.

### Next step → E9

Scenario replay. `scenario_replay_prob=0.15` mutates 15% of resets to
force (HP=1, known-live-next, one defensive item {BEER/SMOKE/INVERTER})
states. With ~15% of rollouts starting in s*, the policy will sample
alternative actions often enough that advantages become non-degenerate.
No shaping for E9 — pure scenario replay on terminal ±1, with γ=0.999.


---

## E9 — scenario replay (15%): visit-forcing without shaping

| | |
|---|---|
| Date | 2026-04-20 |
| Run dir | `rl_runs/E9_scenario_replay/` |
| Hypothesis | Predicted fix from E7 post-mortem. Force 15% of resets into "agent at HP=1, known-live next, single defensive item {BEER/SMOKE/INVERTER}" states. With USE_BEER / USE_INVERTER sampled in these states at ε-rate, PPO can compute a real advantage and pull the policy away from SHOOT_OPPONENT. No shaping, γ=0.999, terminal ±1 only. |
| Setup | 3M steps; lr=1e-3 (default anneal); ent_coef=0.01 (annealed from 0.1 at u<100); hidden=256; γ=0.999; `--scenario-replay-prob 0.15`; league pool with strong_baseline. ~25 min on 1× H100 GPU 1. |
| Headline | WR vs baselines: random 0.93, aggressive 0.76, conservative 0.77. Return50 stabilized around +0.1-+0.25 but noisy (shaping removed). |
| **Outcome** | **Survival blindspot UNCHANGED.** Behavioral probe shows SHOOT_OPPONENT p=0.999 at 1HP+BEER (V=−0.72), SHOOT_OPPONENT p=0.999 at 1HP+INVERTER (V=−0.69). Critic correctly values the state as losing; actor cannot escape its commitment. |

### Diagnosis: actor entropy collapse, not visit failure

The scenario injection worked mechanically (11/11 tests green,
including injection verification). But the metrics show the actor
hasn't been moving for most of training:

| Update | ent | approx_kl | clipfrac |
|---|---|---|---|
| 1 | 1.00 | 8.1e-3 | 0.15 |
| 100 | 0.26 | 5.2e-3 | 0.05 |
| 500 | 0.14 | 1.3e-3 | 0.02 |
| 1000 | 0.14 | 3.8e-4 | 0.00 |
| 1400 | 0.15 | 3.0e-5 | **0.00** |
| 1463 | 0.16 | 6.8e-9 | **0.00** |

After u100 the entropy flatlines at 0.14-0.16; by u1000 `clipfrac=0`
and `kl<1e-3`, meaning the PPO update ratio is ~1.0 almost
everywhere. By u1400 KL is 1e-5 — the actor is frozen. The LR
schedule has decayed to 6e-7 as well, contributing.

So scenario replay *did* force the state visit. But at the injected
state, the actor samples SHOOT_OPPONENT with p=0.999, and that
distribution is reinforced whenever the action-under-injection
happens to lead to a positive rollout return (which it can — opponent
retaliation is stochastic). USE_BEER is sampled with p=0.001 —
statistically ~1 of every 1000 injection rollouts — yielding an
advantage estimate dominated by noise. Since `clipfrac=0`, the policy
gradient ratio ≈ 1 and the clip doesn't activate to pull the policy
toward USE_BEER even when its advantage is positive.

This is **not** the visit-distribution failure of E7. Visits happen.
But the *entropy floor* is too low: p(USE_BEER|s*) = 0.001 means a
15%·0.001 = 1.5e-4 rollout rate in the target scenario. Even across 3M
timesteps (~300k episodes), that's only ~50 rollouts of USE_BEER in s*
— far from enough to overcome the policy's prior commitment.

### Next step → E10: escape entropy collapse

Two orthogonal fixes to try in E10:

1. **Higher entropy floor**: raise `ent_coef` from 0.01 to 0.05 (or
   0.1) and remove the anneal schedule. At ent_coef=0.05 with
   current logits, sampling diversity in collapsed states roughly
   doubles per bit of entropy gained.
2. **Disable LR anneal** (or set cosine to floor at 1e-4 instead of
   ~0): the current near-zero LR means gradients can't move the actor
   even when they're non-zero.

Also bump `--scenario-replay-prob` from 0.15 to 0.30. Combined with
higher entropy, USE_BEER sampling rate in s* should go from 1.5e-4 to
roughly 30% · 5% = 1.5e-2 — a 100× increase in alternative-action
rollouts, which should give PPO enough advantage signal to move.

---

## E10 — escape entropy collapse (ent=0.05, no-anneal, replay=0.30)

| | |
|---|---|
| Date | 2026-04-20 |
| Run dir | `rl_runs/E10_escape_collapse/` |
| Hypothesis | E9 diagnosed actor collapse, not visit failure. Three orthogonal fixes: higher `ent_coef` (0.01→0.05), disable LR anneal (so gradients stay effective), bump `scenario_replay_prob` (0.15→0.30). Target: p(defensive_action \| s*) ≫ 1e-3 so PPO can accumulate advantage signal. |
| Setup | 3M steps; lr=3e-4 flat (`--no-anneal-lr`); ent_coef=0.05; scenario_replay=0.30; hidden=256; γ=0.999; league pool with strong_baseline. ~25 min on 1× H100 GPU 1. |
| Headline | WR vs baselines: random 0.94, aggressive 0.76, conservative 0.71. Return50 ≈ −0.1 to +0.1 (more exploration → slightly worse vs easy baselines, expected tradeoff). |
| **Outcome** | **Partial success.** The 1HP+INVERTER blindspot **solved**: policy now picks USE_INVERTER with p=0.937 (E9: 0.001). The 1HP+BEER case improved from p=0.001 to p=0.381 but SHOOT_OPPONENT still wins the argmax at 0.619. |

### Training dynamics — collapse fixed

| Update | ent | approx_kl | clipfrac | lr |
|---|---|---|---|---|
| 100 | 0.40 | 1.1e-2 | 0.08 | 3e-4 |
| 500 | 0.32 | 4.3e-3 | 0.04 | 3e-4 |
| 1000 | 0.29 | 2.2e-3 | 0.03 | 3e-4 |
| 1400 | 0.30 | 2.2e-3 | 0.03 | 3e-4 |
| 1463 | 0.30 | 1.4e-3 | 0.02 | 3e-4 |

Contrast with E9: ent stays around 0.3 (not 0.14), clipfrac stays
around 2-3% (not 0%), LR stays at 3e-4 (not 6e-7), and KL stays in
1e-3 range (not 1e-8). The actor is actively updating through the end
of training. *This* is the regime we wanted.

### Behavioral probe A/B (E8 → E9 → E10)

| Scenario | E8 | E9 | E10 | |
|---|---|---|---|---|
| `beer_when_certain_death_next_shot` | SHOOT 1.00 | SHOOT 0.999 | **SHOOT 0.62 / BEER 0.38** | partial |
| `inverter_save_from_known_live` | SHOOT 1.00 | SHOOT 0.999 | **INVERTER 0.94** | ✓ fixed |
| `smoke_when_low_hp` | SMOKE 1.00 | SMOKE 1.00 | SMOKE 1.00 | kept |
| `cuff_saw_combo` | CUFF 0.70 | CUFF ~0.7 | CUFF 1.00 | kept/improved |
| `handsaw_lethal` | ✓ | ✓ | ✓ | kept |

### Why BEER is stickier than INVERTER

Both actions have the same "mask is legal" status and similar raw
state features. Candidate explanations:

1. **Asymmetric downstream reward**: USE_INVERTER flips slot0 from
   LIVE to BLANK, so the agent's *very next* action is SHOOT_SELF on
   a now-blank shell → free turn, guaranteed survival, and possibly a
   kill attempt with the next live. USE_BEER ejects slot0 entirely,
   passes turn to opp. Opp acts next. From the actor's credit
   perspective, INVERTER is a higher-immediate-value move in s*, so
   the advantage is larger and easier to learn.
2. **Pre-training bias**: early in training SHOOT_OPPONENT beats
   everything against random-play (opp usually just dies), so all
   "shoot vs item" bandits start SHOOT-favored. Breaking that prior
   for BEER takes more updates than for INVERTER because INVERTER has
   cleaner downstream credit.

### Next steps

Priority: confirm E10 is a stable regime (not a lucky seed). Two
low-risk follow-ups:

- **E11a**: same config, seed=2, 3M steps — reproducibility check.
- **E11b**: same config, **6M steps** — extrapolate whether BEER
  flips with more updates at the same entropy/LR. If yes, that's the
  cheapest path forward.

Alternative if BEER still doesn't flip at 6M: slightly bias the
scenario replay toward BEER states (e.g. sample {BEER: 0.5, INVERTER:
0.25, SMOKE: 0.25} instead of uniform 1/3), or run a targeted
evaluation of what the agent actually *does* in live BEER scenarios
(are we measuring policy output correctly, or is the action being
chosen differently under env stochasticity?).
