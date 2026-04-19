# Research notes — breaking the PPO self-play plateau

Literature review performed mid-session (2026-04-20) after PPO + league
self-play plateaued at ~70 % mean win-rate vs rule-based baselines and a
tournament showed `phase3_u5000` (~20M steps) barely beat
`sweep_rr_winner` (~2M steps) — **10× compute for ~3 % win-rate gain**.

The plateau is characterised by the agent's refusal to ever choose
`SHOOT_SELF` even when `P(blank) = 0.75` (the EV-optimal move). This is a
local optimum that self-play is structurally bad at escaping: since both
sides avoid self-shots, the loss surface has no gradient pushing either
policy toward the correct behavior.

## What the literature says

### 1. Self-play local optima — the actual problem

**Top finding: this is a known pathology of vanilla league training.**
The fix is **dedicated exploiter agents** (Vinyals et al., *Grandmaster
level in StarCraft II using multi-agent reinforcement learning*, Nature
2019; refined in McAleer et al., *Anytime PSRO for Two-Player Zero-Sum
Games*, 2022):

- **Main agents** train vs. a meta-Nash mixture of league opponents
  (normal PPO + league).
- **Main-exploiters** train only to beat the current main — they'll find
  and ruthlessly exploit holes like "main never shoots self".
- **League-exploiters** train to beat *anyone* in history.

When the main encounters an exploiter who figured out "shoot self on
blank-majority", it's forced to either adopt that behavior itself or
defend against it — breaking the symmetric local optimum.

A **simpler partial fix**: sample opponents from the league using a
meta-Nash weighting instead of uniform/FIFO. 20-line change, captures
much of the PSRO benefit.

### 2. Inference-time search

**Second-best finding.** A stripped-down Information-Set MCTS using the
trained PPO policy as prior and the value head at leaves (determinized
PIMC variant; roots in Cowling et al. 2012; modernised in Schmid et al.,
*Player of Games*, Science 2023) can **mechanically fix** the self-shoot
decision: the search sees the EV directly and acts on it, even if the
policy network is still biased.

Full ReBeL / SoG is a multi-month project; determinised PIMC with the
existing value net is a few-days job.

### 3. Exploitability as a progress metric

Train a dedicated best-response DQN against the frozen main (Timbers et
al., *Approximate exploitability*, IJCAI 2022). Its win-rate is a live
exploitability estimate — tells you *quantitatively* how big the
"never-shoots-self" hole is, and lets you track whether an intervention
closed it.

## What the literature says NOT to bother with

- **LSTM / transformer policies**: our observation already summarises
  hidden info (counts + known-shell flags); recurrence buys at most a
  few pp. (Ni et al., *Recurrent Model-Free RL can be a Strong Baseline*,
  ICML 2022.)
- **NFSP / Deep CFR**: works but is slower than PPO+league on games of
  this size, and we already have league scaffolding.
- **LLM-as-policy**: overkill for a 19-action, 47-feature game;
  inference cost kills self-play.
- **Full ReBeL / Student of Games**: multi-month build for marginal gain
  over a cheaper PIMC approximation.
- **Expanding action space to allow item-chaining**: do this for **game
  fidelity** (the real Steam game allows it), but it does **not** break
  the self-play plateau — the plateau is a valuation problem, not an
  expressiveness one.

## Ranked next experiments

1. **Main-exploiter + meta-Nash opponent sampling** — low-medium
   difficulty, directly targets the pathology, uses existing
   `OpponentPool` infrastructure. Expected impact: +5-10 pp win-rate
   and a principled "exploiter can't find a hole" progress signal.
2. **Determinized IS-MCTS with PPO prior + value** — medium difficulty,
   leverages already-trained checkpoints, will fix the self-shoot
   decision immediately. Expected impact: visible jump on blank-majority
   scenarios even before policy network is retrained.
3. **Exploitability best-response DQN** — medium difficulty, mainly
   diagnostic; makes it measurable whether (1) and (2) actually helped.

Each of the above is a tractable multi-day project and fits into the
existing `rl/` codebase with minimal architectural change.
