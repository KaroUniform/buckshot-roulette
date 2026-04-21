# Strong rule-based AI v1 — `strong_baseline`

## Summary

`strong_baseline_opponent` is a hand-engineered, deterministic-given-RNG
Buckshot Roulette AI that lives at
[`rl/opponents.py`](../opponents.py) and is registered in
`NAMED_OPPONENTS` under the key `"strong_baseline"`.

Designed as a **long-lived baseline** for two purposes:

1. A non-collapsible reference point that future RL agents must beat.
2. A diverse warm-start opponent in the league pool (auto-included via
   `NAMED_OPPONENTS`, see `rl/ppo.py:147`).

The AI is intentionally compact (~400 lines) and uses **only** the public
observation + action mask + RNG — no peeking at engine internals.

## Win-rate benchmarks (n=1000, seed=2024, 95% CI)

| Opponent                   | Win rate | 95% CI  | wins/1000 | Brief target |
|----------------------------|---------:|--------:|----------:|-------------:|
| `random`                   |   0.914  | ±0.017  | 914       | ≥ 0.60 ✓ (+0.31) |
| `aggressive`               |   0.723  | ±0.028  | 723       | ≥ 0.55 ✓ (+0.17) |
| `conservative`             |   0.646  | ±0.030  | 646       | ≥ 0.55 ✓ (+0.10) |
| `strong_baseline` (mirror) |   0.505  | ±0.031  | 505       | ~0.50 sanity ✓ |
| League champion `A_ent005_gen2` (~5M steps PPO + league) | 0.393 | ±0.030 | 393 | aim ≥ 0.45, miss by 0.06 |

Run via `python -c "..."` in the project root; full reproduction script
embedded in this commit's PR description.

### Cross-baseline win-rate grid (n=300 each, seed=2024)

| agent vs       | random | aggressive | conservative | strong_baseline |
|----------------|-------:|-----------:|-------------:|----------------:|
| random         |   –    |   0.177    |   0.177      |   0.087         |
| aggressive     | 0.843  |     –      |   0.490      |   0.270         |
| conservative   | 0.807  |   0.567    |     –        |   0.327         |
| strong_baseline| 0.923  |   0.730    |   0.623      |     –           |

Interpretation: `strong_baseline` strictly dominates every other
rule-based opponent across the full grid. Mirror match is balanced
(0.505 ± 0.031), which sanity-checks the determinism (small drift comes
from per-RNG tie-breaking on 50/50 shells).

## Probe scenarios

Ran the 22-scenario battery from
[`rl/analyze_policy.py`](../analyze_policy.py) (originally built to
inspect PPO policies). Of 22 scenarios, **20 produce the textbook
optimal action**, and the remaining 2 are arguably correct (the
"expected" hint over-prescribes):

| Scenario                                    | Picked              | Verdict |
|---------------------------------------------|---------------------|---------|
| `blank_majority_no_items`                   | `SHOOT_SELF`        | optimal |
| `live_majority_no_items`                    | `SHOOT_OPPONENT`    | optimal |
| `50_50_no_items`                            | `SHOOT_OPPONENT`    | Nash (CFR finding) |
| `glass_when_uncertain`                      | `USE_GLASS`         | optimal |
| `handsaw_lethal`                            | `USE_HANDSAW`       | optimal |
| `smoke_when_low_hp`                         | `USE_SMOKE`         | optimal |
| `handcuff_then_lethal`                      | `USE_HANDCUFF`      | optimal |
| `inverter_known_blank`                      | `USE_INVERTER`      | optimal |
| `saw_when_known_live`                       | `USE_HANDSAW`       | optimal |
| `saw_when_known_blank`                      | `SHOOT_SELF`        | optimal (saw avoided) |
| `smoke_useless_at_full_hp`                  | `SHOOT_OPPONENT`    | smoke masked, picks alt |
| `beer_when_certain_death_next_shot`         | `USE_BEER`          | optimal (survival) |
| `phone_when_many_shells`                    | `USE_PHONE`         | optimal |
| `adrenaline_steal_saw_for_kill`             | `USE_ADRENALINE`    | optimal (step 1 of 2) |
| `adrenaline_steal_smoke_when_low`           | `USE_ADRENALINE`    | optimal (step 1 of 2) |
| `pills_vs_sure_kill`                        | `SHOOT_OPPONENT`    | optimal (no pills suicide) |
| `pills_when_desperate`                      | `SHOOT_OPPONENT`    | optimal (no pills suicide) |
| `cuff_saw_combo`                            | `USE_HANDCUFF`      | optimal (chain start) |
| `inverter_save_from_known_live`             | `USE_INVERTER`      | optimal (life save) |
| `glass_peek_unseen_all_live`                | `SHOOT_OPPONENT`    | acceptable¹ |
| `opp_cuffed_go_aggressive`                  | `USE_HANDSAW`       | optimal |
| `phone_redundant_with_glass_info`           | `SHOOT_OPPONENT`    | acceptable² |

¹ `glass_peek_unseen_all_live`: chamber is `[True, True, True]`, p_live = 1.0
unconditionally. Glass is genuinely redundant — every shot is live, so
shooting opp is correct. The "expected" hint is conservative.

² `phone_redundant_with_glass_info`: pos 0 already known via glass; phone
reveals a *random* later position with diminishing value when slot 0 is
the only one we'll fire this turn. Going straight to a shot is fine.

For comparison, the 5M-step PPO league champion `A_ent005_gen2` passes
13/22 of the same scenarios (notably **fails** `beer_when_certain_death`,
`inverter_save_from_known_live`, and `inverter_known_blank` — the
"survival blindspot" documented in
[`rl/docs/research_notes.md`](research_notes.md)). The strong baseline
addresses these head-on.

## Design philosophy

A cascade of priority gates with **early-exit** on first match. Roughly:

1. **Adrenaline-pick mode** — if the engine is waiting for a `PICK_<X>`
   choice, pick the opponent's item that maximises this turn's value
   (saw for lethal, smoke for heal, info if uncertain, BEER/INVERTER
   for survival).
2. **Next shell known LIVE** —
   - one-shot kill available → cuff-then-shoot, or saw-then-shoot
   - cannot kill but would die next turn → BEER → INVERTER → CUFF →
     ADRENALINE survival cascade
   - else heal/saw setup, then shoot opponent
3. **Next shell known BLANK** —
   - INVERTER converts to a likely-live shot (always profitable while
     opp is alive)
   - else burn the blank with a self-shot
4. **Unknown next shell** —
   - critical-survival: 1HP + p_live ≥ 0.5 → glass/phone for info,
     smoke for cushion, adrenaline for steal
   - info gathering when uncertainty is genuine (`0.25 ≤ p_live ≤ 0.85`)
   - heal when wounded
   - adrenaline to fill missing roles (info, smoke, saw)
   - saw before a likely-lethal shot (only when not wasted)
   - cuff before a likely-live shot if there's a real follow-up kill
   - shot via Nash-aligned tie-break: `p_live ≥ 0.5` → `SHOOT_OPPONENT`,
     else `SHOOT_SELF` with risk-adjusted thresholds at low HP

### Anti-patterns explicitly avoided

These mirror failure modes observed in PPO sweeps (E1–E4 in
`experiment_log.md`):

- **`USE_HANDSAW` on a known blank.** Saw resets after the blank shot →
  pure waste. Strong baseline gates saw on `slot0_live ∧ ¬excess_dmg`.
- **`USE_HANDSAW` on excess damage** (e.g., opp already at 1HP, raw shot
  kills). Saw consumed for nothing. Gated on
  `_can_one_shot_opp(with_saw=False) == False`.
- **`USE_PILLS` when alternatives exist.** 60% chance of -1HP (often
  suicide), 40% of +2. Strong baseline never picks PILLS unless it's
  the only legal action; the cascade always finds a shot first.
- **`SHOOT_SELF` at p_live ≈ 0.4 with no chainable follow-up.** Coin-flip
  on tempo for negligible expected gain. Tie-break only commits to
  `SHOOT_SELF` when `p_live ≤ 0.35` or there's a saw / glass / phone /
  cuff'd-opp follow-up that compounds the free turn.
- **`SHOOT_SELF` at 1HP for moderate p_live.** A self-shot at 1HP and
  p_live > 0.2 risks instant death for marginal EV. Strong baseline
  diverts to `SHOOT_OPPONENT` in this band.

### What it does *not* model

- **Opponent modelling / bluff inference.** No tracking of what items
  the opponent picked, no inference of opp's hidden knowledge.
  Treats opp as black-box-but-rational-enough; misses exploits like
  "opp always cuffs before shooting" or "opp wasted their saw, so I
  can self-shoot freely now."
- **Multi-step planning.** Each call returns one action; no game-tree
  search, no rollouts. The cascade is greedy with respect to the
  current state.
- **Belief updating from opponent moves.** If opp uses GLASS / PHONE we
  don't infer anything about which positions they revealed. Human
  experts learn from this; our AI does not.
- **Dynamic threshold adjustment.** All probability cutoffs (0.5, 0.35,
  0.45, 0.85) are fixed. A learner could fine-tune these per game phase.

These omissions are intentional — they keep the code reviewable and
prevent the baseline from accidentally encoding strategic heuristics
the RL policy is supposed to discover. The ~6% gap to `A_ent005_gen2`
on the head-to-head benchmark is mostly attributable to these missing
opponent-aware components plus tactical patterns PPO discovered through
game-tree exploration that aren't in the cascade.

## Honest weaknesses

1. **Loses to the league champion** (~39% win rate). The 5M-step PPO
   champion uses items 4–10× more aggressively than this baseline (see
   action-mix diagnostic in commit history): USE_INVERTER 102 vs 8 over
   matched samples, USE_HANDCUFF 138 vs 74, USE_HANDSAW 109 vs 68.
   Champion exploits the baseline's conservative item-burn pattern.
   Mitigations attempted (more aggressive INVERTER on known-blank,
   looser cuff threshold, BEER on uncertain high-p_live) gave +1–4 pp
   each but with high variance (n=500 samples) and didn't break 0.45.
2. **Mirror match is balanced** (50.5%) — the AI is not noticeably
   non-deterministic, so two copies playing each other rely on rng
   tie-breaking. Acceptable for a baseline.
3. **No tempo-aware adrenaline use** — adrenaline triggers only when
   opp has a directly-useful item; misses cases where stealing a
   neutral item denies opp a future combo. Champion does this.
4. **Pills strategy is binary** — never picks pills. In genuinely
   desperate states (known-live, no items, no shots that win), pills
   may be EV-positive vs accepting the loss. We never test this.

## Files added / modified

- **Added** `rl/opponents.py::strong_baseline_opponent` (+ helpers).
- **Added** `rl/opponents.py::NAMED_OPPONENTS["strong_baseline"]`
  registration so it auto-joins the training pool in
  [`rl/ppo.py:147`](../ppo.py).
- **Added** `rl/test_opponents.py` — 19 tests covering legality (1000
  episode stress), determinism, registry presence, 12 probe
  scenarios, and 3 win-rate guards (vs random, aggressive,
  conservative).
- **Modified** `rl/round_robin.py` — `--include-baselines` flag for
  inserting rule-based competitors into a checkpoint round-robin
  (e.g., grounding neural results against `strong_baseline`).
- **Added** this document.

## Reproducing the benchmarks

```bash
# Unit tests (legality 1000 eps + 12 probes + 3 win-rate floors)
python -m rl.test_opponents

# Existing tests (sanity that nothing broke)
python -m rl.test_engine
python -m rl.test_single_agent_env

# Round-robin including baselines
python -m rl.round_robin --league-dir rl_runs/league_v1 \
    --episodes 500 --include-baselines strong_baseline aggressive
```

## Future work

Improving the baseline's vs-champion winrate further would require
either (a) opponent modelling (track opp's item history, infer their
known_shells from item usage), or (b) a small game-tree search at
decision time (depth-2 minimax over `me_acts × shell_outcomes ×
opp_acts`). Both are out of scope for a "rule-based reference point";
they belong in a search-augmented agent or in the RL policy itself.

The cleanest measurable next step is to use this baseline as a
**training opponent** for a fresh PPO run and check whether the
resulting agent breaks 0.55 vs `A_ent005_gen2`. If yes, the baseline
contributed to the league diversity in a measurable way.
