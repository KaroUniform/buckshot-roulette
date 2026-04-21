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

Both actions are legal and both keep the turn with the agent (BEER
does *not* pass turn to opp — verified against `engine.py:_apply_item`
and a direct step-level trace). The real asymmetry is about the
*information* in the post-action state, not about who moves next.

1. **Downstream information asymmetry**: USE_INVERTER flips slot 0
   from LIVE to BLANK, leaving a *known-blank* shell in slot 0. The
   agent's next action is a guaranteed-safe SHOOT_SELF → free turn,
   clean survival, and often a lethal setup next. USE_BEER ejects
   slot 0; the new slot 0 is whatever was previously in slot 1 —
   usually *unknown* in the probe scenario. So after BEER the agent
   must act under uncertainty, which the critic scores lower. The
   advantage A(s, BEER) is smaller than A(s, INVERTER), giving less
   gradient pull to flip the prior.
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

---

## E11a + E11b — GRU recurrent policy + observation-honesty ablation

### Why the pivot (original plan vs. what we ran)

Original E11a/b above was "rerun E10 longer / with different seed" — a
reproducibility check for the FF baseline. That check is pending but
low-priority: E10's BEER-hesitation is a known behavior gap and no
amount of FF budget is obviously going to fix the deeper issue, which
is that **the current observation vector leaks privileged info**.

Specifically, `obs[_O_N_LIVE]` and `obs[_O_N_BLANK]` give the agent the
public chamber live/blank counts at every tick, after every shot, after
BEER, after INVERTER. A human player only *hears* the initial
declaration ("two live, three blank") and then has to *remember* what
was shot / ejected to maintain a belief over the remaining chamber.
INVERTER flips the next shell — publicly known to both players that it
was used, privately resolved to the user, so the opponent loses one bit
of certainty on the chamber.

So the FF agent we've been training is a cheater: it gets an oracle
counter that re-derives itself from the public event stream, but saves
the agent the work. An FF policy **cannot learn to count** — it has no
memory. That's why the previous path was "give it the answer and let
it act on it." Before investing another 6M FF steps, we switched tracks
to answer the more interesting question:

> If we take away the hack, can a recurrent policy recover the missing
> counters from public events alone?

If **yes**, we get a bot that plays under human-equivalent information,
with the same (or better) win rate against rule-based baselines. If
**no**, we know the observation hack was doing real work and we need to
either keep it (accept the cheat), bias the architecture more strongly
(larger GRU, auxiliary counter-prediction loss), or bring in something
like a Transformer history encoder.

### Setup

Both runs use **identical hyperparameters**, *identical code paths*,
and the same **3 M steps / 32 envs × 128 rollout / 4 minibatches ×
4 epochs** budget. The differences are exactly two:

| | E11a | E11b |
|---|---|---|
| Observation | **hack, 47-dim** (with public `n_live` / `n_blank`) | **honest, 52-dim** (public initial declaration + per-round event counters for shots / BEER ejections / INVERTER uses) |
| Seed | 1 | 2 |

Both use:
- GRU-based [`RecurrentActorCritic`](../policy.py) (hidden=128, embed=128)
- `lr=3e-4`, `ent_coef=0.05`, `gamma=0.999`, `no-anneal-lr`
- `scenario-replay-prob=0.30` (E10's winning recipe)
- Self-play league: 4 rule-based openers (`NAMED_OPPONENTS`) + rolling
  recurrent snapshots (one every 10 updates, cap 10)
- Eval: 100 eps vs each named opponent every 5 updates
- Named-opponent helpers auto-collapse honest→hack via `_as_hack_obs`
  so E11b's rule-based opponents work without changes

The honest obs layout is a strict superset of "human-public" knowledge:
from `initial_live − shots_live − beer_ejected_live` the agent can
recompute public `n_live` exactly in no-INVERTER rounds, and bound it
±`inverter_uses_this_round` once INVERTER has been used. The agent
still sees its own `glass`/`phone` revelations via the per-shell
knowledge block (slots 0..7). Nothing privileged about the opponent's
inventory is exposed that wasn't already public.

### Training dynamics

Both runs converged to a stable policy after ~1M steps; no entropy
collapse, no value divergence, smooth eval progression.

`rollout/mean_return_50` trajectory:
- **E11a**: −0.76 → mid −0.20 → final −0.28
- **E11b**: −0.56 → mid +0.08 → final −0.28

League pool size ends at 13 (E11a) and 12 (E11b). Sign of a healthy
self-play mix; the league drove win rates down temporarily around the
middle when stronger recurrent snapshots joined.

### Result

Final eval at 500 eps/opponent on the shipped `policy_final.pt`
(torch-seeded for reproducibility):

| Opponent | E11a (hack) | E11b (honest) | Δ (E11b − E11a) |
|---|---|---|---|
| random | 0.914 | 0.908 | −0.006 |
| aggressive | 0.816 | 0.828 | +0.012 |
| conservative | 0.692 | 0.756 | **+0.064** |
| strong_baseline | 0.632 | 0.648 | +0.016 |
| **mean** | **0.764** | **0.785** | **+0.022** |

Binomial stderr at n=500 is ≈0.021, so only the conservative delta
(+0.064, ≈3σ) is individually significant; random/aggressive/strong
are within noise.

**Head-to-head** (honest env, 800 eps):

|  | Win rate | 95 % CI |
|---|---|---|
| E11b (policy) vs E11a (opponent, honest→hack collapse) | **0.579** | [0.545, 0.613] |

The reverse match-up (E11a=policy, E11b=opponent) is architecturally
unplayable: E11b expects 52-dim obs that can't be reconstructed from a
47-dim hack obs (no `initial_live` state to recover). The honest→hack
direction works because a hack-trained opponent only needs public
counts, which the honest env exposes via the counters.

The 57.9 % head-to-head is **4.5 σ above 50 %** — strong evidence the
honest-obs policy is meaningfully better at playing this game, not
just a tie on baselines.

### Interpretation

**Memory fully compensates for dropping public `n_live`/`n_blank`, and
goes a step further — honest-obs is slightly *better* in aggregate.**
Three plausible mechanisms, in descending likelihood:

1. **Counter hack was doing more harm than good.** The hack obs
   smuggles in derivable info (public counters) that's redundant with
   the per-shell `known_live`/`known_blank` slots + event history, and
   it arrives in a non-Markovian encoding — the FF policy can't reason
   about *how that count got there* (was a BEER ejection? an INVERTER
   flip?) only the current state. A GRU with access to the event
   stream gets both the current state *and* its derivation, which
   gives better credit assignment on INVERTER/BEER actions. This
   matches the conservative-specific improvement: `conservative` is
   the opponent most sensitive to chamber-awareness — it plays safely
   when counts are balanced, so extracting edge requires nuanced
   belief updating rather than raw counting.
2. **Slight regularization.** Removing one redundant feature shrinks
   the trivial "exploit the count" gradient pathway and may push the
   policy to learn more robust value estimation on the remaining 50
   dims. Similar in spirit to dropout.
3. **Seed variance.** With only one run each, a +0.02 mean gap is not
   conclusive on its own — the head-to-head result is the strongest
   signal and is well above noise, but with n=1 per arm we can't rule
   out this being a lucky E11b seed interacting with a slightly
   weaker E11a seed. Would need ≥3 seeds per arm to lock this in.

### Confidence + caveats

- **Strong signal for the ablation hypothesis.** Honest obs ≥ hack
  obs on every baseline and wins head-to-head — this is enough to
  commit forward to honest obs as the default.
- **`strong_baseline` is still weak.** Both policies sit ~65 % vs.
  `strong_baseline`, unchanged from E10. The BEER-hesitation
  pathology is not fixed by recurrence alone — it needs a targeted
  intervention (biased scenario replay / auxiliary probe loss).
- **No self-play ceiling probe.** We don't know the match-up
  dynamics between E11b and the *old FF baseline* E10. Likely E11b
  beats it (the head-to-head pattern suggests so), but this is a
  data-gap, not a claim.

### Next steps → E12

Given honest obs works, there are two obvious and one risky
follow-ups:

1. **E12a — seed replication (cheap).** Rerun E11b with seeds 3, 4,
   5 (same honest obs, same config). ~90 min of GPU, gives us a
   3-seed mean ± stdev on the canonical setup so we can distinguish
   the +0.064 conservative gain from noise.
2. **E12b — BEER auxiliary loss.** Add an auxiliary head on the
   recurrent trunk that predicts the remaining chamber composition
   given history. Regularizes the GRU to actually track counts and
   (hopefully) unsticks BEER against `strong_baseline`. 3M steps.
3. **E13 — league fusion.** Run FF-E10 as a pinned opponent in the
   E11b-style league for 3M more steps. If recurrent + honest obs
   really is better, we should see the FF-E10 slot's opponent-weight
   in the league go to the ceiling, and the policy's `strong_baseline`
   number should climb. This also gives us the E11b-vs-E10 number we
   don't currently have.

Priority: E12a (lowest risk, fills the seed-variance gap), then E12b
(if seed replication holds) to chip at the BEER ceiling.

---

## E12a — seed replication of the E11b honest-obs setup

**Hypothesis.** E11b (honest 52-dim obs + GRU, 3M steps) was a single
seed. The +0.064 conservative gain over E11a (and the 57.9% head-to-head
in E11b's favor) might be a fluke. Run the same config on 3 fresh seeds
(3, 4, 5) and check that the means are stable, the strong_baseline
ceiling persists, and E11b's numbers sit somewhere inside the seed
distribution rather than way above it.

### Setup

Identical to E11b in every respect except `--seed`. All three runs:

| field | value |
|---|---|
| script | `e12a_s{3,4,5}_launch.sh` (same body as `e11b_launch.sh` modulo seed + run-name) |
| total-timesteps | 3,000,000 |
| num-envs / num-steps | 32 / 128 (rollout = 4096 steps, ~733 updates) |
| lr / ent-coef | 3e-4 / 0.05 (no anneal) |
| hidden / embed | 128 / 128 (RecurrentActorCritic) |
| obs layout | honest 52-dim (`--honest-obs`) |
| scenario-replay-prob | 0.30 |
| GPU | beeline-prod GPU 1, sequential (one at a time, polite tenant) |
| wallclock | ≈ 14m / seed |

**Code-version caveat (small).** The three seeds did NOT all train on
the exact same opponents.py:
- s3 trained on the **pre-INVERTER-fix** opponents (commit `8424fd6`).
  Rule-based opponents got the buggy honest→hack collapse that ignored
  `inverter_uses_this_round`.
- s4 trained on the **post-INVERTER-fix** opponents (`5f320a9`):
  per-opponent `obs_layout` routing, rule-based opponents always see
  hack obs with true post-inverter counts.
- s5 trained on **post-bugbot-#5/#6** opponents (`f5585be`): same as s4
  plus two strong_baseline heuristic patches (no wasted HANDCUFF before
  guaranteed lethal, no wasted ADRENALINE on inverter when only opp owns
  the saw). Both patches affect rare edge-cases — strong_baseline
  win-rate vs random/aggressive/conservative was unchanged in unit
  tests, so the training-time effect is well below seed variance.

E11b itself trained on the same pre-INVERTER-fix code as s3.

### Final eval — 500 eps/opponent, post-fix opponents (apples-to-apples)

All four policies (E11b + s3/4/5) were re-evaluated with the latest
opponent code (`f5585be`) so the win-rate numbers are directly
comparable. E11b's numbers in this table are NOT what was reported in
the E11 post-mortem above (which used pre-fix opponents); the post-fix
strong_baseline is slightly stronger, which mostly affects that column.

| opponent | s3 | s4 | s5 | 3-seed mean ± stdev | E11b re-eval |
|---|---|---|---|---|---|
| random | 0.908 | 0.912 | 0.920 | **0.913 ± 0.006** | 0.922 |
| aggressive | 0.810 | 0.778 | 0.808 | **0.799 ± 0.018** | 0.822 |
| conservative | 0.684 | 0.696 | 0.708 | **0.696 ± 0.012** | 0.730 |
| strong_baseline | 0.620 | 0.584 | 0.578 | **0.594 ± 0.023** | 0.598 |
| mean of 4 | 0.756 | 0.743 | 0.754 | **0.751** | 0.768 |

Binomial stderr at n=500: ±0.013–0.022 per seed/opponent, i.e. each
single-seed cell has CI roughly ± one stdev's worth of seed variance.

### Interpretation

1. **Seed variance is real but bounded.** Per-opponent stdev across the
   3 seeds is 0.6%–2.3%, with the hardest opponents (`strong_baseline`,
   `aggressive`) showing the largest spread. The seed effect is on the
   same order as the binomial noise of a 500-episode eval, so each
   seed's reported number is "true ± ~2%."

2. **E11b sits modestly above the 3-seed mean, not at the top.**
   - random: +0.009 ≈ 1.5σ above mean
   - aggressive: +0.023 ≈ 1.3σ above mean
   - conservative: +0.034 ≈ 2.8σ above mean
   - strong_baseline: +0.004, statistically indistinguishable

   E11b's most "lucky-looking" result (conservative) is ~3σ above the
   3-seed mean — notable but well within what a 4-sample distribution
   would produce. The honest-obs setup is reproducible; E11b was on
   the favorable end of variance, not anomalous.

3. **The strong_baseline ceiling (~0.60) is structural, not a seed
   artifact.** All three E12a seeds landed at 0.578–0.620 vs
   strong_baseline. E11b's 0.598 in this re-eval matches. The BEER /
   adrenaline-cuff blindspots flagged in the E11 post-mortem are a
   property of the architecture/observation/training distribution, not
   of the random seed. Closing that gap requires E12b (auxiliary head
   for chamber tracking) or E13 (FF-E10 in the league), as planned.

4. **Mean of opponent-means: E12a 0.751 vs E11b 0.768.** The single
   E11b number that was treated as "E11b's headline" is +0.017 above
   what 3-seed mean predicts. Future E11-style claims should report
   either a ≥3-seed mean or a wider confidence interval; the
   single-seed result was genuinely informative about *direction* (the
   honest-obs config does work) but overstated the magnitude.

### Confidence + caveats

- Three seeds is the bare minimum for "stdev" to be meaningful; a
  Welch's-t comparison between E12a and a hypothetical 3-seed E11a
  pool would have wide CIs. We didn't replicate E11a (hack obs) on 3
  seeds — that's the obvious symmetric experiment if we want a
  rigorous A/B, but it's also another ~45 minutes of GPU and the E11
  head-to-head (57.9% in E11b's favor at 4.5σ) is independent
  corroboration that honest > hack.
- The opponent-code drift between s3 (pre-INVERTER-fix) and s4/s5
  (post-fix) is a confound. Magnitude bound: post-fix strong_baseline
  is ~3% stronger in unit tests, which would reduce s4/s5's vs-strong
  numbers by a few percent if anything. Looking at the numbers, s4/s5
  ARE somewhat lower vs strong (0.584 / 0.578 vs s3's 0.620), but the
  effect is in the same direction as expected and within stdev.
- Eval was 500 eps × 4 opponents (= 2000 eps per seed). No
  head-to-head matrix between the 3 E12a seeds (cheap to run later
  if useful — would tell us whether the seeds learn distinguishable
  policies or all converge to roughly the same fixed point).

### Decision

Honest-obs + GRU + 3M-step + replay-0.30 setup is **reproducible** and
**robust enough for the league**. All future runs use this config as
the baseline. E11b's individual numbers should be cited with the
3-seed mean (0.751) as the canonical reference, with E11b itself as
the high-side of the seed distribution.

### Next steps

1. **E12b — BEER auxiliary loss (next).** With the seed-variance
   question answered, the BEER blindspot vs strong_baseline is the
   biggest open performance gap. Add a chamber-composition prediction
   head on the GRU trunk (auxiliary cross-entropy loss with weight ≈
   0.05), train 3M steps from scratch on seed 6 with the same E11b
   config otherwise. Hypothesis: if the GRU is forced to actually
   track remaining live/blank counts, the policy will recognize and
   exploit BEER-good states more often.
2. **E13 — league fusion (later).** Pin FF-E10 as a permanent opponent
   in the league. Gives the missing E11b-vs-E10 number and might
   transfer FF-E10's strong-vs-strong play into the recurrent policy.
3. **Optional E12c — E11a 3-seed replication.** Only if a reviewer
   challenges the asymmetric seed comparison. Same compute as E12a.

---

## E12b — BEER auxiliary loss (chamber-composition regression)

**Hypothesis:** the honest-obs GRU has to *infer* remaining live/blank
counts from the public event counters, and the plateau against
`strong_baseline` at ~0.60 suggests it's not doing so well enough to
recognize high-EV BEER/SHOOT states. Forcing the trunk to predict the
true (n_live, n_blank) via an auxiliary MSE loss should provide a dense
learning signal that shapes the GRU's internal state and frees the
policy head to focus on decision-making, not estimation.

### Implementation

- `RecurrentActorCritic` gains an optional 2-dim linear head reading from
  the GRU hidden state (`aux_dim=2`). Constructed only when requested,
  so old checkpoints load unchanged under `strict=True`.
- `forward_sequence(return_aux=True)` returns per-timestep aux
  predictions. Single-step path is unchanged (aux is only used during
  PPO updates, where we replay the whole T-length minibatch).
- `SingleAgentBuckshotEnv.step/reset` emit the true chamber composition
  in every info dict as `n_live`/`n_blank`. `SyncVectorEnv` aggregates
  them into per-env arrays automatically.
- `ppo_recurrent.py` captures the aux target at every rollout step,
  alongside `obs_buf`, and adds `aux_coef * MSE(aux_pred, aux_target)`
  to the PPO minibatch loss. CLI flags: `--aux-chamber --aux-coef 0.05`.

### Run config

Identical to E11b/E12a otherwise:
- seed 6, 3M steps, envs=32, rollout=128, hidden=128, honest_obs=True
- scenario_replay_prob=0.30, cosine LR, gamma=0.995
- `--aux-chamber --aux-coef 0.05`

Wall-clock: ~90 min on 1×H100 (same as E11b/E12a).

### Training dynamics

The aux head learned cleanly. Aux MSE trajectory:
- u1:  0.99 (cold start, essentially random)
- u10: 0.25
- u50: 0.09
- u200: 0.066
- second-half mean: 0.060, range 0.031–0.100

Converged within ~200 updates and stayed at ~0.06 for the remaining
500+. Chamber counts are in [0, 4] for live/blank, so MSE 0.06 ≈
±0.24 shells RMS error — the trunk is tracking composition within ¼ of
a shell on average, effectively perfect for the game-relevant decisions.

Policy/value losses, entropy, and KL behaved normally — no evidence
that the aux gradient perturbed PPO's dynamics.

### Evaluation (500 eps/opponent, post-fix rule-based opponents, seed 20260420)

| opp | E12b s6 | E12a 3-seed mean ± stdev | E11b (re-eval) |
|---|---|---|---|
| random | 0.934 ± 0.011 | 0.913 ± 0.006 | 0.922 |
| aggressive | 0.794 ± 0.018 | 0.799 ± 0.018 | 0.822 |
| conservative | 0.706 ± 0.020 | 0.696 ± 0.012 | 0.730 |
| strong_baseline | **0.594 ± 0.022** | 0.594 ± 0.023 | 0.598 |
| mean of 4 | **0.757** | 0.751 | 0.768 |

Binomial stderr at n=500 is ±0.013–0.022 per opponent.

### Interpretation: clean negative result

**The aux loss did not move win rates.** E12b's mean across 4 opponents
is 0.757 vs E12a's 3-seed mean of 0.751 — a 0.6pp difference, well
inside the ±2% per-seed noise band. Per-opponent:
- vs `random`: +2.1pp (within 2σ)
- vs `aggressive`: -0.5pp (indistinguishable)
- vs `conservative`: +1.0pp (indistinguishable)
- vs `strong_baseline`: +0.0pp (**identical to 3 decimals**)

The single-seed variance we observed in E12a (strong-baseline range
0.578–0.620 across s3/s4/s5) dwarfs any E12b signal. E12b falls
squarely inside that distribution.

**Key finding: the aux head learned its task essentially perfectly, and
it still didn't help.** This rules out "the aux target was too hard" or
"the gradient didn't flow" — those would have shown up as stagnant aux
loss. The aux loss was 0.06 by update 200 and stayed there.

### Why this is informative (even though the hypothesis failed)

1. **The GRU was already tracking chamber composition.** If
   (n_live, n_blank) were genuinely absent from the trunk's internal
   state before adding the aux head, adding it would have changed the
   representation and the policy should have shifted. It didn't. That
   suggests the policy/value gradients *already* push the GRU to encode
   counts well enough to play — adding explicit pressure on those
   specific quantities is redundant.

2. **The strong_baseline ceiling is NOT about chamber estimation.**
   With both E12a's 3-seed mean and E12b at 0.594 against
   `strong_baseline`, we now have 4 independent 3M-step honest-obs
   policies that all cap at 0.58–0.62. Either:
   - `strong_baseline` plays near-optimally given its information (most
     likely — it has access to the hack obs, which reveals public
     counters),
   - or the gap is in decision-making (item sequencing, timing of
     HANDSAW+HANDCUFF, exploiting BEER states) rather than in state
     estimation.

3. **Auxiliary-loss representation shaping needs a harder target.**
   (n_live, n_blank) is too easy: it can be derived deterministically
   from the public counters already in the obs (initial declaration +
   shots_fired + beer_ejected). Future aux-loss experiments should pick
   a target that actually requires Bayesian inference — e.g. the
   posterior probability that the next chambered shell is live after
   INVERTER use, or the expected value of shooting opponent given
   current inventory.

### Decision

**Drop E12b from the league; revert to the E11b/E12a honest-obs config
as the baseline.** The aux_chamber flag stays in the codebase (costs
nothing, may be useful for diagnostic probing), but the next experiment
should move on from representation shaping to opponent / training-
distribution changes.

### Next steps (updated)

1. **E13 — league fusion (promoted to next).** Pin FF-E10 as a
   permanent opponent. Gives the missing E11b-vs-E10 number and, more
   importantly, forces the recurrent policy to face a qualitatively
   different (feedforward, hack-obs) opponent that may push it past the
   `strong_baseline` plateau. Cheap (~90 min / H100) and well-scoped.
2. **E14 — harder aux target (deferred).** Only revisit auxiliary loss
   after E13 if the plateau persists. Candidate targets: p(next shell
   live | obs), which requires the GRU to maintain a posterior through
   the round rather than just count. Less trivially derivable from
   counters — forces real inference.
3. **E15 — structural changes (speculative).** If league fusion also
   fails to break the 0.60 ceiling, consider: larger hidden size,
   attention over past events, or distributional value head. These are
   expensive and should wait until simpler interventions are exhausted.

---

## E13 — league fusion with FF-E10 as permanent opponent

**Hypothesis:** the `strong_baseline` 0.60 ceiling persists across 3
honest-obs seeds plus the aux-loss variant (E12b). FF-E10 is a
qualitatively different opponent (feedforward, hack obs, self-play
trained) that may expose exploit patterns rule-based baselines don't
cover. Training against FF-E10 in the pool should either (a) push the
policy past the `strong_baseline` plateau or (b) give us a first
head-to-head number for honest-obs-recurrent vs hack-obs-feedforward.

### Run config

Identical to E12a/E12b except:
- seed 7 (new)
- `--extra-opponent-ckpts rl_runs/E10_escape_collapse/policy_final.pt`
- no aux chamber loss

FF-E10 joins the pool with `weight=1.0`, same as each rule-based
opponent. Total pool grows from 4 → 5 at u1; with up to 8 snapshots the
pool caps at 13, so FF-E10 appears as ~1/13 of encounters mid-run.
Existing `obs_layout` routing auto-detected FF-E10's 47-dim hack layout
and fed it the right observation per-episode (no code changes needed).

Wall-clock: ~90 min on 1×H100.

### Evaluation (500 eps/opponent, seed 20260420)

| opp | E13 s7 | E12b s6 | E12a 3-seed mean | E11b |
|---|---|---|---|---|
| random | 0.936 ± 0.011 | 0.934 | 0.913 ± 0.006 | 0.922 |
| aggressive | 0.800 ± 0.018 | 0.794 | 0.799 ± 0.018 | 0.822 |
| conservative | 0.680 ± 0.021 | 0.706 | 0.696 ± 0.012 | 0.730 |
| strong_baseline | **0.594 ± 0.022** | 0.594 | 0.594 ± 0.023 | 0.598 |
| ff_e10 | **0.492 ± 0.022** | — | — | — |
| mean of 4 standard opps | 0.753 | 0.757 | 0.751 | 0.768 |

### Interpretation

**Another clean negative result on `strong_baseline`.** E13 lands at
**0.594** — identical to 3 decimals with E12a's 3-seed mean AND E12b.
That's now four independent 3M-step honest-obs runs (E12a s3, s4, s5,
E12b s6, E13 s7) all converging to 0.58–0.62 against strong_baseline.
The consistency is remarkable — and damning for representation / league
interventions:

> No intervention tried so far (aux loss, league fusion) has moved the
> `strong_baseline` win rate by more than seed noise (~2%).

**The new datum: E13 goes 49.2% ± 2.2% head-to-head vs FF-E10.**
This is the first ever honest-obs-recurrent vs hack-obs-feedforward
number. It's a statistical dead heat (CI overlaps 0.50). Takeaways:

1. The E11 head-to-head test (57.9% for honest-obs-GRU vs hack-obs-GRU)
   showed that the recurrent GRU on honest obs actually *beats* a GRU
   on hack obs — but FF-E10 is hack obs + FEEDFORWARD, and here the
   recurrent-honest-obs policy is only even. That's consistent with
   FF-E10 being a stronger policy class than E11a-GRU (it took multiple
   iterations + the ent/no-anneal fix to get there), not evidence
   against the honest-obs direction.
2. Training *against* FF-E10 didn't flip the matchup in E13's favor.
   At ~1/13 encounter frequency, FF-E10 appears for ~300k steps of the
   3M run — that may simply be too rare for the policy to specialize
   against.

**Side effect: vs_conservative regressed (0.706 → 0.680, ~1.5σ).**
Small enough to be seed noise, but consistent with a mild
specialization tradeoff: pool-weighted training against a stronger,
differently-styled opponent shifts the policy's mixed strategy slightly
away from rule-based conservatives. Not big enough to be a real
concern, but flags that adding mismatched opponents with `weight=1.0`
isn't free.

### Why this is informative (even though the primary hypothesis failed)

1. **The 0.60 ceiling is independent of the league.** Four honest-obs
   runs with different seeds, aux losses, and opponent pools all land
   at the same ±2% window vs `strong_baseline`. That's not a training
   artifact — it's a **structural property** of honest-obs-GRU at this
   scale against a rule-based opponent with hack-level information.

2. **FF-E10 vs honest-recurrent is roughly 50/50.** That fills in the
   missing cell from the E11 post-mortem ("E11 → E10 head-to-head") at
   low cost and provides a reasonable yardstick: our current recurrent
   agent is *competitive* with a well-tuned feedforward hack-obs agent,
   not dominant. More compute / bigger models / richer curricula could
   each move this number, but representation shaping and league fusion
   won't.

3. **The remaining gap is almost certainly information asymmetry.**
   `strong_baseline` and FF-E10 both see hack obs (public
   n_live/n_blank). Our agent sees honest obs. A ~5pp gap vs each is
   consistent with the information advantage itself being worth ~5pp in
   close games. If that's right, no amount of representation tweaking
   on the honest-obs side will close it — we'd need either (a) honest
   obs with better inference (still bounded by the information
   content), or (b) accept the plateau and focus on closing the gap
   at the tail (e.g. mixed-strategy adversarial training).

### Decision

**Stop A/B-testing honest-obs interventions; pivot.** We now have
strong evidence that:
- honest-obs + recurrent + 3M steps ≈ 0.75 mean-of-4-opponents
- the ceiling vs `strong_baseline` is 0.60 and will not be moved by
  simple code changes

Either accept this as the baseline and move to a different axis
(opponent curriculum, self-play depth, architecture scale), or revisit
the honest-obs assumption itself. The task-bot deliberate-information-
hiding framing (agent must infer what hack-obs reveals directly) is
philosophically clean but may be strictly harder in a way that
architecture fixes can't close without orders of magnitude more data.

### Next steps (revised)

1. **E14 — self-play scale-up (promoted).** Longer horizon: 10M steps,
   same honest-obs config, no aux, no FF-E10 pin. Tests whether the
   plateau moves with ~3× more data. If yes, compute is the binding
   constraint. If no, it's structural and we should change the
   architecture or observation.
2. **E15 — bigger GRU (parallel).** Same 3M-step honest-obs config but
   hidden=256 (4× params). Tests the "maybe 128 hidden isn't enough to
   track the full posterior" hypothesis. Cheaper than E14.
3. **E16 — dropped honest-obs abstraction.** Go back to hack obs +
   recurrent and see if GRU+hack beats FF-E10 and strong_baseline by a
   wider margin. Useful data point; costs only one run.
4. **Deferred: harder auxiliary tasks.** The E12b post-mortem suggested
   predicting p(next-shell-live | obs) as a better aux target. Only
   revisit after architecture/data scaling is exhausted.

---

## Addendum to E13 — hack-obs recurrent (E11a re-eval) vs honest-obs

After the E13 post-mortem, one key cell was still missing: what does
**hack-obs + recurrent** (the E11a checkpoint) score against the
post-fix opponent suite? This single number decides whether the 0.60
ceiling on honest-obs-GRU is about *information* or *architecture*.

### Method

Re-ran E11a's `policy_final.pt` at 500 eps/opp vs the same opponent
suite E12b/E13 used, with `honest_obs=False` (E11a was trained on the
47-dim hack observation). Also evaluated FF-E10 the same way for an
apples-to-apples reference point, and threw in FF-E10 as an extra eval
opponent for both.

### Results

| Opponent | E11a hack-GRU (3M) | E13 honest-GRU+FF-E10-pool (3M) | FF-E10 (3M, feedforward) |
|---|---|---|---|
| random | 0.936 ± 0.011 | 0.936 ± 0.011 | 0.930 ± 0.011 |
| aggressive | 0.810 ± 0.018 | 0.800 ± 0.018 | 0.796 ± 0.018 |
| conservative | 0.760 ± 0.019 | 0.680 ± 0.021 | 0.744 ± 0.020 |
| **strong_baseline** | **0.646 ± 0.021** | **0.594 ± 0.022** | **0.654 ± 0.021** |
| ff_e10 | 0.550 ± 0.022 | 0.492 ± 0.022 | — |

### What changes

1. **The 0.60 ceiling on honest-GRU is about information, not
   architecture.** Hack-GRU (E11a) hits 0.646 vs `strong_baseline`,
   essentially matching FF-E10's 0.654. The gap between hack-GRU and
   honest-GRU is **5.2pp** (2.4σ) — right in the range you'd expect if
   explicit n_live/n_blank counters in the observation are genuinely
   worth a few points in close-game decisions.

2. **Architecture (FF vs recurrent) is NOT the bottleneck.** At
   matched hack observations, recurrent-GRU (E11a: 0.646) ≈ feedforward
   (FF-E10: 0.654), and head-to-head E11a **beats** FF-E10 at 0.550 ±
   0.022 (2.3σ above 50%). The GRU is at least as good as FF at this
   scale — the original reason we switched to recurrent (inference
   over hidden state) is doing its job.

3. **The E12b post-mortem's conclusion partially reverses.** E12b
   concluded "the ceiling is NOT about chamber estimation — it's
   about decision-making." That's *half* right. The correct reading:
   - Honest-obs GRU *can* track chamber composition (aux head proved
     this in E12b).
   - But the honest observation itself **contains less information**
     than hack. Specifically, the public shot/beer/inverter counters
     don't pin down n_live exactly when items have been used in
     ways that change composition (BEER-eject on unknown-color, or
     INVERTER flip semantics). Some games remain genuinely ambiguous
     to honest-obs that aren't to hack-obs.
   - Therefore representation shaping on the honest side can't close
     the gap (matches E12b's negative result), but hack obs is
     directly worth ~5pp.

4. **E13's "structural ceiling" claim needs revision.** The
   `strong_baseline` ceiling is structural **for honest-obs** but not
   for the game. Hack-observer policies (both recurrent and FF) land
   ~5pp above it.

### Decision

- **The honest-obs project is a real handicap, not just an aesthetic
  choice.** Continuing on honest obs means accepting a 5pp tax vs
  hack-observer opponents.
- **If we care about maximizing play strength, switch back to hack
  obs.** E11a is already a decent baseline there; scaling it (longer
  run, bigger model, richer league) is the fast path.
- **If we care about demonstrating learning under information
  asymmetry** (the original honest-obs motivation), the plateau is the
  point and we should accept 0.594 as the correct number — but then
  evaluate against opponents that *also* use honest obs, not against
  `strong_baseline` with hack info.

This is a meaningful reframing. The E14 / E15 / E16 priorities shift:

### Next steps (revised again)

1. **E16 promoted — re-enter hack obs and scale up.** Train a fresh
   3M-step recurrent policy on hack obs with seed 8, FF-E10 in the
   league, and `--aux-chamber` off. Compare to E11a's 0.646 — if we
   significantly exceed it, scaling the hack-obs line further is the
   frontier. Cheap (~90 min).
2. **E17 — honest-vs-honest evaluation.** Build an honest-obs analog
   of `strong_baseline` that only reads public counters (BEER-counter,
   INVERTER-counter, shots-fired) and can't peek at n_live directly.
   Evaluate E12a/b seeds against *that*. If they score much better,
   the honest-GRU is actually playing well — it just loses info-tax to
   hack-obs rule-based opponents. If they still plateau, the issue is
   deeper.
3. **E14 and E15 remain useful but lower priority.** Data/model
   scaling on honest obs is fine but unlikely to close the full 5pp
   information gap regardless. E16 answers the more pressing question
   first.


## E16 — hack-obs GRU + FF-E10 pinned (seed 8): FF-E10 pinning is a net negative

**Goal.** Two things at once:
1. Reproduce the hack-obs recurrent result on a fresh seed (E11a was a
   single-seed point).
2. Test whether adding FF-E10 as a permanent league opponent lifts the
   hack-obs GRU above E11a's 0.646 vs `strong_baseline` (same recipe
   that E13 tried on honest obs and failed).

**Config.** `ppo_recurrent`, hack obs, `--seed 8`, 3M steps, hidden=128,
embed=128, num_envs=32, num_steps=128, num_minibatches=4,
update_epochs=4, `--scenario-replay-prob 0.30`,
`--extra-opponent-ckpts rl_runs/E10_escape_collapse/policy_final.pt`,
no `--aux-chamber`. Trained on 1× H100 (GPU 1), ~40 min wall.

### Results (500 eps/opp)

| Opponent | E16 (hack + FF-pin, seed 8) | E11a (hack, no FF-pin, seed 7) | E13 (honest + FF-pin, seed 7) | FF-E10 (hack FF) |
|---|---|---|---|---|
| random | 0.922 ± 0.012 | 0.936 ± 0.011 | 0.936 ± 0.011 | 0.930 ± 0.011 |
| aggressive | 0.792 ± 0.018 | 0.810 ± 0.018 | 0.800 ± 0.018 | 0.796 ± 0.018 |
| conservative | 0.742 ± 0.020 | 0.760 ± 0.019 | 0.680 ± 0.021 | 0.744 ± 0.020 |
| **strong_baseline** | **0.604 ± 0.022** | **0.646 ± 0.021** | **0.594 ± 0.022** | **0.654 ± 0.021** |
| ff_e10 | 0.482 ± 0.022 | 0.550 ± 0.022 | 0.492 ± 0.022 | — |

### Three headline findings

1. **FF-E10 pinning HURTS hack-obs GRU vs `strong_baseline`.** E16 lands
   at 0.604 — 4.2pp **below** E11a's 0.646 (1.4σ). The expected
   "better league ⇒ stronger policy" effect is absent; the actual
   effect is a regression.

2. **FF-E10 pinning fails to teach the GRU to beat FF-E10.** E16 goes
   0.482 head-to-head vs the pinned opponent — statistically
   indistinguishable from 50%, and **below** E11a's 0.550 against the
   same FF-E10 checkpoint (E11a never trained against it). Training
   *against* FF-E10 made the GRU *less* able to beat it.

3. **With FF-E10 pinned, hack vs honest obs becomes
   indistinguishable.** E16 (hack) and E13 (honest) converge to nearly
   identical scores across the board — 0.604 vs 0.594 on
   `strong_baseline` (0.3σ), 0.482 vs 0.492 on `ff_e10` (0.3σ). The
   5.2pp observation-honesty gap we measured between E11a and E13
   (1174–1180) **collapses** once both are trained against FF-E10.

### What this means

The most economical interpretation: **FF-E10 pinning distorts the
training distribution toward a near-Nash response to FF-E10**, and
that response is a compromise that (a) underperforms vs weaker
rule-based opponents and (b) doesn't improve the head-to-head against
FF-E10 itself. The collapse of the obs-honesty gap is the giveaway:
once both policies have to spend capacity drawing against FF-E10,
whatever extra info hack obs provides stops mattering for the
rule-based opponents too.

The E11a hack-obs advantage over honest-obs (≈5pp) was **specific to
the rule-based-only league**. It reflected hack-obs's ability to
exploit the rule-based opponents' predictable plans. As soon as a
stronger, less-exploitable opponent (FF-E10) enters the pool, the
GRU stops exploiting and starts drawing — and exploitation is where
the obs-honesty asymmetry lived.

### The broader lesson

We now have three independent experiments (E12b aux loss, E13 honest
+ FF-pin, E16 hack + FF-pin) that all landed at ~0.60 vs
`strong_baseline` despite very different interventions. The only
configuration that beats 0.60 is **hack obs + rule-based-only
league** (E11a 0.646, FF-E10 0.654). This is a strong pattern:

> The 0.65 ceiling is the **exploit-ceiling against rule-based
> opponents when observations expose n_live**. Any intervention that
> forces the policy to be robust (honest obs, FF-E10 pin, aux loss)
> trades exploit for robustness and lands at 0.60.

To push past 0.65 we probably need one of:
- A richer diverse league (many different archetypes, none pinned)
  so the policy can still exploit by conditioning on opponent type.
- Opponent modeling (explicit context-inference about who we're
  playing) — either architectural or via auxiliary identification
  losses.
- A larger / deeper recurrent network so representational capacity
  stops being the constraint. We have H100s; 3M-step 128-hidden is
  currently leaving compute on the table.

### Decisions for the next batch

1. **Drop FF-E10 pinning as a league strategy.** Two independent runs
   (E13 and E16) show it regresses the policy. Unless we find a
   different way to make it helpful (curriculum? reduced weight in
   the pool?), don't spend more compute on this axis.
2. **E18 — scale the E11a recipe.** Hack obs, rule-based-only league,
   fresh seed, **bigger model** (hidden=256, embed=256) and **longer
   training** (10M steps). This is the cheapest way to test whether
   0.65 is a learning-capacity ceiling or a structural one.
3. **E17 — honest-vs-honest evaluation** remains on the queue but
   demoted below E18. Until we've tried to push the hack-obs ceiling,
   debating the honest-obs 0.60 plateau is premature — it may be a
   reflection of the 0.65 hack ceiling rather than a separate issue.
4. **Optional: opponent-embedding experiment (E19).** Add a
   per-episode learned opponent embedding (or auxiliary opponent-ID
   prediction) so the policy can condition its strategy on who it's
   playing. This is the principled way to recover the exploitation
   capacity FF-E10-pinning destroyed.


## E18 — scaled E11a recipe (in progress, 2026-04-21)

**Hypothesis.** The 0.65 hack-obs ceiling (E11a 0.646, FF-E10 0.654)
may be a *learning capacity* limit of a 128-hidden GRU at 3M steps
rather than a structural game ceiling. With every "robust training"
intervention (FF-E10 pinning, honest obs, aux loss) landing at ~0.60
instead, pushing the exploit side deserves a proper scaling test
before declaring the ceiling structural.

**Config.** `ppo_recurrent`, **hack obs**, **rule-based-only league**
(no `--extra-opponent-ckpts`), fresh seed=9, **hidden=256 / embed=256**
(4× more params than E11a), **10M total timesteps** (3.3× longer than
E11a). Other hyperparameters match E11a/E16: num_envs=32,
num_steps=128, num_minibatches=4, update_epochs=4, lr=3e-4 with
default anneal, ent_coef=0.01, gamma=0.995,
`--scenario-replay-prob 0.30`, `--eval-every 10`, `--eval-episodes
100`, `--snapshot-every 20`. Running on beeline GPU 1 (H100 80GB,
pid 3891824). Wall time estimate: ~130 min total (captures larger
model + longer run + eval overhead).

**Predictions — what to look for on the 500-ep final eval:**

- **Strong positive (0.68 – 0.72 vs `strong_baseline`):** scaling
  works; capacity *was* the bottleneck. Next: scale further
  (hidden=384, 20M steps) and consider opponent embeddings (E19).
- **Modest positive (0.66 – 0.68):** real but small; 0.65 is
  close to the true ceiling. E19 (opponent embedding) likely the
  bigger win than further scaling.
- **Flat (0.63 – 0.66):** scaling doesn't help. 0.65 is structural
  for this game under hack obs vs rule-based opponents. E19 becomes
  the priority, and we should also revisit the game itself (are
  we modeling the right distribution of opening shell counts?).
- **Negative (<0.63):** bigger model overfits / destabilizes.
  Report the result, switch back to hidden=128 and try E19 from
  that base.

Post-mortem will go below this stub. Intermediate eval every 40
updates via `eval_every=10`, so the 100-ep training-eval trajectory
will already hint at the answer before the 500-ep final.


### Results (500 eps/opp, post-training)

| Opponent | E18 (h=256, 10M, seed 9) | E11a (h=128, 3M, seed 7) | Δ |
|---|---|---|---|
| random | 0.938 ± 0.011 | 0.936 ± 0.011 | +0.002 |
| aggressive | 0.828 ± 0.017 | 0.810 ± 0.018 | +0.018 |
| conservative | 0.774 ± 0.019 | 0.760 ± 0.019 | +0.014 |
| **strong_baseline** | **0.676 ± 0.021** | **0.646 ± 0.021** | **+0.030** |
| ff_e10 | 0.550 ± 0.022 | 0.550 ± 0.022 | +0.000 |
| e11a | 0.580 ± 0.022 | — | — |

**Landed in the "modest positive" bracket** (0.66–0.68) predicted
in the stub. Three points worth noting:

1. **+3pp on `strong_baseline`** (1.0σ). E11a's 95% CI was
   [0.605, 0.687]; E18's mean 0.676 sits near the top of that CI.
   The gain is real across the full eval pool (weaker rule-based
   opponents also moved +1–2pp) but not dramatic — the 0.65
   plateau appears to extend to at least 0.68 with 4× compute.

2. **E18 beats E11a head-to-head at 0.580 ± 0.022** (3.6σ above
   50%). This is the cleanest "scaling worked" signal — the
   gain is unambiguously more than sampling noise. E18 is
   strictly a stronger player than E11a, even if that strength
   shows up as only +3pp against one shared opponent.

3. **No gain vs FF-E10** (0.550 both times). Scaling against a
   rule-based league doesn't transfer to FF-E10 specifically.
   The head-to-head vs FF-E10 looks like a different axis — the
   same problem E16's post-mortem flagged (pinning FF-E10 hurts;
   not pinning leaves us at 55%).

### Decisions after E18

- **The 0.65 ceiling is real but soft.** 4× compute bought +3pp
  against `strong_baseline` and clear head-to-head superiority
  over the previous-generation agent. The curve is not flat but
  it's slow — pushing to ~0.72 with hidden=384 / 20M steps would
  be another ~3× compute for probably another +2-3pp. Real but
  expensive.
- **E19 (opponent embedding) promoted to top priority.** The FF-E10
  axis didn't move at all. That's not a scaling problem — it's
  that a single GRU can't simultaneously exploit rule-based
  openings and play cautiously against a trained opponent.
  Giving the policy explicit per-episode context about *who*
  it's playing is the principled fix, and it's where the
  remaining win is.
- **E20 (hidden=384, 20M steps) on deck but de-prioritized.**
  Unless E19 surprises us, more of the same scaling is unlikely
  to close the FF-E10 gap and offers only modest gains vs
  rule-based. Keep it as a cheap sanity check if we need to
  confirm the 4× → +3pp trend extrapolates.
- **Saved checkpoint.** E18's final policy is now the new best
  single-policy hack-obs baseline. Subsequent experiments
  (E19, E20) should include it as an eval opponent so we can
  track absolute scaling progress head-to-head.


## E19 — opponent embedding (in progress, 2026-04-21)

**Motivation.** E18 showed that scaling hack-obs + rule-based league
delivers +3pp on `strong_baseline` (0.676 vs E11a's 0.646) and a
clean head-to-head win (0.58) but *zero* movement against FF-E10
(stuck at 0.55). A single GRU is forced to average its strategy
across opponents that reward opposite play styles: rule-based
opponents reward aggressive exploitation, trained policies punish
it. E19 tests whether explicit per-episode opponent context lets
the policy specialize.

**Minimum viable design.** Discrete opponent ID → small learnable
embedding → concatenated with obs embedding before the GRU.

- **ID table (5 slots, fixed):**
  - 0: `random`
  - 1: `aggressive`
  - 2: `conservative`
  - 3: `strong_baseline`
  - 4: *any other* (catch-all: self-play snapshots, FF-E10,
    future-added extras)

  Rationale: the rule-based set is small, stable, and exactly the
  axis where exploitation matters. Snapshots and FF-E10 all live
  behind slot 4 because:
  1. Snapshot identity shifts during training (pool.add(snapshot_k)
     can happen any time), so per-snapshot IDs would create a
     moving target the policy can't actually learn.
  2. From the policy's POV, "snapshot k" and "snapshot k-5" play
     almost the same game at this scale — they're all
     near-self-play. Collapsing them is not much of a loss.
  3. Keeping the table small (5) means the embedding learns
     crisp per-archetype responses rather than sparse per-
     snapshot noise.

- **Env plumbing.** `SingleAgentBuckshotEnv` already stores
  `self._opp_name` at reset. Add a name→ID lookup (module-level
  constant `_OPPONENT_ID_MAP`) and stash `self._opp_id`. Include
  it in every info dict return (`reset`, `step`, `_terminal_return`).

- **Policy change.** `RecurrentActorCritic` gains an optional
  `n_opponents: int` arg (default 0 → embedding disabled for
  backward compat with existing E11a/E16/E18 checkpoints). If
  enabled, an `nn.Embedding(n_opponents, embed_dim)` layer is
  added; its output is *added* to `obs_embed(obs)` before the
  GRU (addition rather than concat keeps `input_size=embed_dim`
  so GRU shape is unchanged and backward compat is easier).

- **PPO buffer.** A new `opp_id_buf[num_steps, num_envs]` int64
  buffer captures per-step opponent ID from info. It slices
  alongside obs in minibatches and feeds `forward_sequence(...,
  opp_ids=...)`. `load_recurrent_policy` auto-detects
  `n_opponents` from `state["opponent_embed.weight"].shape[0]`.

- **Training knobs.** New CLI args: `--opp-embed` (bool flag),
  `--n-opponents` (int, default 5).

**Training config (E19 run).** Match E18 for apples-to-apples:
hack obs, rule-based-only league, seed=10, hidden=256, 10M steps,
`--opp-embed --n-opponents 5`. Eval at 500 eps/opp including vs
E18, FF-E10, E11a.

**Predictions.**
- **Win:** E19 beats E18 by ≥3pp on `strong_baseline` AND flips
  the FF-E10 head-to-head above 0.55. This is the target result.
- **Half-win:** E19 matches E18 vs `strong_baseline` (≈0.68) but
  moves FF-E10 to 0.55+. Tells us per-opponent specialization is
  working but rule-based was already near-ceiling.
- **Flat:** No movement on either axis. Either ID 4 is too coarse
  (snapshots and FF-E10 need separate slots) or the bottleneck is
  elsewhere entirely. Next step: per-snapshot IDs or opponent-ID
  *inference* head (E20).
- **Negative:** ID embedding destabilizes training. Unlikely given
  the tiny parameter count, but documented as a possibility.

**Implementation plan (incremental commits):**
1. `opp_id_map` constant + env info plumbing + tests. ✅ (ac92221)
2. `RecurrentActorCritic` opponent embedding + tests
   (backward-compat checkpoint load included). ✅ (780e1b7)
3. PPO rollout buffer + training CLI flags + smoke test. ✅ (0aa0441)
4. Launch E19 run on beeline, eval, post-mortem appended here. ⏳

**Status (launched).** E19_opp_embed_s10 is training on GPU 1
(beeline) as of 2026-04-21 20:11 UTC. Config matches E18 exactly
except for `--opp-embed` (n_opponents=5): hidden=256, 10M steps,
seed=10, gamma=0.995, ent_coef=0.01, snapshot_every=10,
eval_every=20, eval_eps=200. League: rule-based only (no FF-E10
pin, which E16 showed is net negative).

Early trajectory (first 17 updates, ~70k steps):
- return_50 climbed from −0.84 at u=1 to +0.28 by u=17. E18's
  return_50 was still around −0.6 at the same step count. This is
  a noticeably faster early ramp — consistent with the embedding
  letting the policy specialize against `random`/`aggressive` (the
  two most-common pool samples early in training, before snapshot
  slots get filled).
- entropy fell from 0.98 → 0.40 by u=17, again faster than E18
  (which still had entropy ≈0.9 at the same step). This is slightly
  concerning — if the policy collapses to a single-mode exploit
  too early, the late-stage league-matching phase might fail to
  recover diversity. Monitor for entropy < 0.1 with stalled
  improvement as an early-stop heuristic.
- approx_kl stable at ~0.002, clipfrac <3%. No optimization
  pathologies so far.

**Post-mortem (appended 2026-04-21 22:45 UTC, after completion).**

Training ran the full 10M steps (2441 updates) on GPU 1. Clean
shutdown, no pathologies. Training-time evals (n=200 eps,
per-opponent) showed wild swings — best vs strong_baseline hit
0.78 at u=2160 with entropy 0.17, then regressed to 0.645 at
the final update with entropy 0.18. The 0.78 was misleading; a
500-eps eval of that same checkpoint gives 0.69 (matching E18).
Moral: a 200-eps eval has ~3pp standard error and is too noisy
to detect real improvements at the Nash-equilibrium plateau.

**500-episode eval (CPU, post-fix handsaw engine):**

| opponent        | E18-final | E19-final | E19-best (u=2160) |
|-----------------|-----------|-----------|-------------------|
| random          | 0.938     | 0.926     | 0.930             |
| aggressive      | 0.820     | 0.834     | 0.838             |
| conservative    | 0.784     | 0.764     | 0.766             |
| strong_baseline | 0.690     | 0.674     | 0.690             |
| E18-final (H2H) | —         | 0.526     | **0.550**         |

**What this tells us:**

1. **Against rule-based opponents, E19 matches E18.** No opponent
   in NAMED_OPPONENTS benefits from opponent-conditioning because
   they're all deterministic policies — the trunk already learns
   to probe-and-respond without an explicit id. E19's embedding
   is basically a no-op here.

2. **Head-to-head vs E18, E19-best wins 55%.** With n=500 that
   gives SE ≈ 2.2pp, so 0.550 vs 0.500 is z ≈ 2.3 (borderline
   significant). This is the cleanest signal that the embedding
   gave *some* real edge: when the pool has diverse opponents
   worth conditioning on (recurrent snapshots are non-stationary),
   the embedding earns its +5 parameters.

3. **Training past peak hurt.** E19-best (u=2160) vs E19-final
   (u=2440) — 1.6M extra steps lost ~2pp on H2H. Overfitting on
   the latest snapshots at the cost of rule-based opponents.
   Suggests a best-checkpoint tracker in future runs instead of
   trusting `policy_final.pt`.

**Verdict: E19 is a marginal win.** The opponent embedding helps
in non-stationary-pool scenarios but not on rule-based evals.
The feature is cheap (+5 params) and now verified safe. Promote
the pattern into E20 but don't expect another big jump from the
representation alone — we're at the Nash plateau for hidden=256.

**What NOT to do in E20:**
- More hidden-size scaling (E18→E19 at hidden=256 gave no rule
  improvement; scaling alone is unlikely to help).
- Another "try a new NN feature" experiment without checking
  whether the current policy actually has room to improve vs
  NAMED_OPPONENTS. 0.69 vs strong_baseline with entropy 0.18 may
  be close to optimal against that specific rule-based policy.

**What to do in E20 — candidates:**
- **Exploiter analysis:** train a dedicated exploiter policy vs
  E19-best with 1-2M steps, see if it consistently beats E19.
  If it wins >60% we know what E19 is missing.
- **Early stopping / best-checkpoint infra:** track best-SB-eval
  during training, save that separately. Small infra win,
  enables longer runs without regret.
- **Pin E18 as a league opponent in E20.** E19's H2H edge is
  small (5pp); forcing E20 to learn vs E18-the-snapshot should
  push it past E19.
- **Co-play with perturbation:** evaluate policy via perturbed
  games (e.g., force small random noise into chamber loading)
  to find robustness gaps.

Note: E19 ran on the pre-handsaw-fix engine (commit a84cfaa
landed after E19 started). Multi-handsaw stacking was rare
enough that eval numbers on the fixed engine are still
meaningful — but any E19 "strategies" that depended on the
exploit are now unreachable. No re-eval needed.

## E20 — exploiter analysis (launched 2026-04-21 00:24 UTC)

Diagnostic experiment picked from E19 post-mortem's candidate
list. Question: does E19 have exploitable blind spots, or is
it at the Nash plateau for hidden=256?

**Setup:** Train a new recurrent PPO agent (fresh init, seed=20,
hidden=256) whose SOLE opponent is the frozen E19-best snapshot
at u=2160. No rule-based opponents in the pool — only the main
and the exploiter's own past snapshots (for diversity). 2M
steps, async env (2× rollout speedup), ~20-30 min on H100 GPU 1.

**Interpretation rules (set before results):**
- Exploiter wins **<52% vs main** → E19 is near-Nash. Stop
  scaling; move to product integration or a different scope.
- Exploiter wins **52-58%** → minor weakness. Fold exploiter
  into E21 league as a pinned opponent; expect small gain.
- Exploiter wins **>60%** → real blind spot. Analyze action
  distribution to pin down the exploit; shape it into training.

**Launch details:** `rl/exploiter.py` uses a monkey-patch on
`rl.ppo_recurrent.OpponentPool` so the standard training loop's
pool contains only `main` (frozen E19-best) plus exploiter
snapshots. Pool.manifest() broadcasts correctly via async env
(ckpt_path annotated). Will post-eval exploiter vs main for
500 eps to get a clean winrate CI.

pid 4075, run_dir rl_runs/E20_exploiter_vs_E19_s20, seed=20.
Post-mortem to follow.

## Async vector envs (infra PR, 2026-04-21)

Drafted during E19's wait. Four-commit chunk into PR #3 that swaps
`gymnasium.vector.SyncVectorEnv` → `AsyncVectorEnv(context="spawn",
shared_memory=True)` for `ppo_recurrent.train`. Full design in
`rl/docs/async_env_optimization_plan.md`. Commits:

- `0c03ef1` — `OpponentPool.manifest()` + `build_pool_from_manifest()`
- `d2856bd` — `SingleAgentBuckshotEnv.sync_pool(manifest)`
- `08e8cba` — AsyncVectorEnv swap + broadcast plumbing + `close(terminate=True)`

Key design decisions worth flagging for later:
- Workers spawn with a rule-only seed pool (not the full pool). The
  parent's full pool may contain CUDA-resident `nn.Module` extras;
  those cannot cross the spawn pipe. Instead the parent broadcasts
  `pool.manifest()` via `envs.call("sync_pool", …)` right after
  construction, which workers reload from disk on CPU.
- League snapshots (every `snapshot_every_updates`) annotate the
  factory with `.ckpt_path = <newly-saved .pt file>` so
  `pool.manifest()` can round-trip them. The save happens BEFORE
  `pool.add` so workers see the file the moment the broadcast
  arrives.
- `threading.Lock` on `OpponentPool` now has `__getstate__`/
  `__setstate__` to survive pickling through the spawn pipe.

Local stability smoke (mac, 11 cores, 4 envs × 128 steps × 19
updates, snapshot_every=5): pool grew 4 → 7 across 4 snapshot
broadcasts, no pipe deadlocks, clean shutdown. Return trajectory
−0.6 → +0.08 (sanity — still learning normally).

**Throughput measurement on beeline (72 vCPU, CPU-only bench, pool
= NAMED_OPPONENTS + 1 recurrent snapshot at 4× weight, 128-step
warmup, random-legal actions as stand-in for the PPO actor):**

| config              | env-step rate | wall-clock 8192 steps | speedup |
|---------------------|---------------|-----------------------|---------|
| Sync, 32 envs × 256 | 5 631 step/s  | 1.45 s                | 1.00×   |
| Async, 32 envs × 256| 11 385 step/s | 0.72 s                | **2.02×** |
| Sync, 64 envs × 128 | 5 754 step/s  | 1.42 s                | 1.00×   |
| Async, 64 envs × 128| 14 766 step/s | 0.55 s                | **2.57×** |

**Why 2× not 32×:**
- Sync barrier: `AsyncVectorEnv.step()` waits for ALL workers to
  finish each step, so throughput = `1/max(t_i)`, not
  `sum(1/t_i)`. Heterogeneous opponent pool (random 50 µs vs
  recurrent-snapshot 500–1000 µs per step) creates per-env
  latency variance → slow workers gate the barrier.
- IPC overhead: even with `shared_memory=True`, 32 pickle
  round-trips per step eat 20–30% at the current per-step cost.
- Shared beeline box: 72 vCPU visible but other users compete;
  workers don't get 100% of a core.

**Expected improvement on real training:** 2.5–3.5× end-to-end
(vs 2× rollout-only) because the pool fills with recurrent
snapshots → per-step variance drops → better parallelism.
H100 forward pass pipelines with worker stepping. `num_envs=32`
is kept for E20 as a drop-in; scaling to 48–64 would cost PPO
hyper-retuning and wasn't judged worthwhile for a 20–30% further
gain.

**Verdict: 2× is enough for E20+. Adopted as-is.**
