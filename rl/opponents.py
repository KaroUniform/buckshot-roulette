"""Opponent policies for self-play training and evaluation.

An Opponent is anything implementing `act(obs, mask, rng) -> int`:
  - obs: float32 observation tensor (numpy)
  - mask: int8 action mask (numpy), 1 where legal
  - rng: numpy Generator for stochasticity

These are intentionally thin so they're cheap to call inside the env loop and
trivial to swap (random / rule-based / frozen neural net all conform).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from rl.engine import Action, Item, NUM_ITEMS

OpponentFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


# ---- Layout helpers (tied to engine.observation()) ----
# Scalar slots in the obs vector — keep in sync with BuckshotEngine.observation()
_O_MY_HP = 0
_O_MY_MAX_HP = 1
_O_OPP_HP = 2
_O_OPP_MAX_HP = 3
_O_N_SHELLS = 4
_O_N_LIVE = 5
_O_N_BLANK = 6
_O_DAMAGE_MULT = 7
_O_IS_MY_TURN = 8
_O_OPP_CUFFED = 9
_O_MY_CUFFED = 10
_O_ADRENALINE_ACTIVE = 11
_O_USED_NON_ADRENALINE = 12

_INV_OFFSET = 13
_OPP_INV_OFFSET = _INV_OFFSET + NUM_ITEMS  # 22
# Per-shell knowledge blocks (max_shells = 8 slots each; engine.shells_range=(2,8))
_MAX_SHELLS = 8
_KNOWN_LIVE_OFFSET = _OPP_INV_OFFSET + NUM_ITEMS  # 31
_KNOWN_BLANK_OFFSET = _KNOWN_LIVE_OFFSET + _MAX_SHELLS  # 39


# Honest-obs layout (52-dim): core section is 18 scalars wide (5 extra for the
# event counters), followed by the same inventory / known-shells blocks.
#
# Rule-based opponents below are written against the 47-dim hack layout and
# expose `obs_layout = "hack"`. The env (`SingleAgentBuckshotEnv`) reads that
# attribute and asks the engine for a hack-layout obs for the opponent, even
# when the agent itself is on honest obs. Reason: an earlier hand-crafted
# honest→hack collapse that derived `n_live/n_blank` from event counters was
# off by up to ±inverter_uses after any INVERTER, since the honest obs
# deliberately hides the (private) flip direction. Rather than leak info or
# silently corrupt the rule-based baselines, we let opponents declare their
# required layout and route accordingly.
_HONEST_OBS_DIM = 52


def random_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


random_opponent.obs_layout = "hack"  # type: ignore[attr-defined]


def aggressive_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Always shoot opponent; prefer handsaw before shooting; otherwise shoot.
    No info-gathering, no self-shots. A "rush" baseline."""
    if mask[int(Action.USE_HANDSAW)] and obs[_O_DAMAGE_MULT] < 2:
        return int(Action.USE_HANDSAW)
    if mask[int(Action.SHOOT_OPPONENT)]:
        return int(Action.SHOOT_OPPONENT)
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


aggressive_opponent.obs_layout = "hack"  # type: ignore[attr-defined]


def conservative_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Naive EV-style heuristic. Uses info items when available; shoots self
    if blanks > lives, opponent otherwise. Heals when low. No bluff modeling.

    This mirrors the popular wisdom from the Buckshot Roulette community,
    so the trained agent should be able to *exploit* it if it learns nuance.
    """
    n_live = obs[_O_N_LIVE]
    n_blank = obs[_O_N_BLANK]
    my_hp = obs[_O_MY_HP]
    my_max = obs[_O_MY_MAX_HP]

    # Heal first if hurt and have a smoke
    if my_hp < my_max and mask[int(Action.USE_SMOKE)]:
        return int(Action.USE_SMOKE)

    # Gather info if uncertain
    if n_live > 0 and n_blank > 0:
        if mask[int(Action.USE_GLASS)]:
            return int(Action.USE_GLASS)
        if mask[int(Action.USE_PHONE)]:
            return int(Action.USE_PHONE)

    # Cuff opponent before lethal shot
    if mask[int(Action.USE_HANDCUFF)]:
        return int(Action.USE_HANDCUFF)

    # Saw before opponent shot
    if mask[int(Action.USE_HANDSAW)] and obs[_O_DAMAGE_MULT] < 2:
        return int(Action.USE_HANDSAW)

    # Shoot decision
    if n_blank > n_live and mask[int(Action.SHOOT_SELF)]:
        return int(Action.SHOOT_SELF)
    if mask[int(Action.SHOOT_OPPONENT)]:
        return int(Action.SHOOT_OPPONENT)

    # Fallback
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


conservative_opponent.obs_layout = "hack"  # type: ignore[attr-defined]


# ---- Strong rule-based baseline ----

def _slot0_known(obs: np.ndarray) -> tuple[bool, Optional[bool]]:
    """Return (is_known, is_live_if_known) for the next shell (slot 0).

    Uses the per-shell knowledge block written by engine.observation().
    `is_live_if_known` is None when the slot is not known.
    """
    if obs[_KNOWN_LIVE_OFFSET] > 0.5:
        return True, True
    if obs[_KNOWN_BLANK_OFFSET] > 0.5:
        return True, False
    return False, None


def _prob_live(obs: np.ndarray) -> float:
    """P(next shell is live) given public counts and any revealed knowledge
    about the next shell specifically. Ignores information about later shells
    (that wouldn't sharpen P for slot 0 without a full belief update, and we
    don't need it for this baseline)."""
    known, is_live = _slot0_known(obs)
    if known:
        return 1.0 if is_live else 0.0
    n_live = float(obs[_O_N_LIVE])
    n_blank = float(obs[_O_N_BLANK])
    n_tot = n_live + n_blank
    if n_tot <= 0.0:
        return 0.5  # degenerate; shouldn't happen during a legal-action query
    # Subtract knowledge about non-slot-0 shells (they can't be slot 0, so they
    # don't influence P(slot 0 is live)). This does NOT change P directly, but
    # makes the counts we use consistent with what's left unknown.
    n_shells = int(obs[_O_N_SHELLS])
    # Known-live/blank at non-zero positions reduce the pool of "random" shells
    later_known_live = 0
    later_known_blank = 0
    for i in range(1, min(n_shells, _MAX_SHELLS)):
        if obs[_KNOWN_LIVE_OFFSET + i] > 0.5:
            later_known_live += 1
        elif obs[_KNOWN_BLANK_OFFSET + i] > 0.5:
            later_known_blank += 1
    unknown_live = max(0.0, n_live - later_known_live)
    unknown_blank = max(0.0, n_blank - later_known_blank)
    denom = unknown_live + unknown_blank
    if denom <= 0.0:
        # All non-slot-0 positions accounted for → slot 0 is whichever is left
        if unknown_live > 0:
            return 1.0
        if unknown_blank > 0:
            return 0.0
        return 0.5
    return float(unknown_live / denom)


def _opp_has(obs: np.ndarray, item: Item) -> bool:
    return obs[_OPP_INV_OFFSET + int(item)] > 0.5


def _me_has(obs: np.ndarray, item: Item) -> bool:
    return obs[_INV_OFFSET + int(item)] > 0.5


def _can_one_shot_opp(obs: np.ndarray, with_saw: bool) -> bool:
    """Can we deal enough damage this shot to kill the opponent?

    Considers current damage_mult (already set if we used saw earlier this
    turn) plus an optional pending saw we haven't yet applied.
    """
    opp_hp = int(round(float(obs[_O_OPP_HP])))
    dmg_mult = int(round(float(obs[_O_DAMAGE_MULT])))
    if with_saw:
        dmg_mult *= 2
    return dmg_mult >= opp_hp


def _fallback(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        # Engine promises at least SHOOT_* is legal mid-game; still be defensive.
        return 0
    return int(rng.choice(legal))


def _tiebreak_shoot(obs: np.ndarray, mask: np.ndarray, p_live: float) -> int:
    """Choose between SHOOT_OPPONENT and SHOOT_SELF given the live probability.

    Decision rule (Nash-aligned per research_notes.md, with adjustments
    for finite-HP tempo concerns):
      - p_live >= 0.5  → SHOOT_OPPONENT (expected damage is maximised)
      - p_live <  0.5  → SHOOT_SELF when:
            (a) p_live <= 0.35  (clearly blank-majority, free turn likely), OR
            (b) we have a useful follow-up (saw or known-live elsewhere), OR
            (c) opp is already cuffed and we can stack damage.
        Otherwise SHOOT_OPPONENT — at p ∈ [0.4, 0.5) the free-turn value
        is too small to risk -1HP. Self-shot at low p_live is essentially a
        coin-flip on tempo, while shoot_opp guarantees we don't take damage.
    On exact 0.5 with mixed legality we prefer SHOOT_OPPONENT (CFR finding).
    """
    if p_live >= 0.5 and mask[int(Action.SHOOT_OPPONENT)]:
        return int(Action.SHOOT_OPPONENT)
    if p_live < 0.5 and mask[int(Action.SHOOT_SELF)]:
        my_hp = float(obs[_O_MY_HP])
        # Risk-adjusted: at 1HP, almost never self-shot unless near-certain blank
        if my_hp <= 1 and p_live > 0.2:
            if mask[int(Action.SHOOT_OPPONENT)]:
                return int(Action.SHOOT_OPPONENT)
        # Standard threshold: aggressive self-shoot only when likely blank
        if p_live <= 0.35:
            return int(Action.SHOOT_SELF)
        # Borderline (0.35 < p_live < 0.5): self-shot only with useful follow-up
        opp_cuffed = obs[_O_OPP_CUFFED] > 0.5
        has_chainable_item = (
            _me_has(obs, Item.HANDSAW)
            or _me_has(obs, Item.GLASS)
            or _me_has(obs, Item.PHONE)
            or opp_cuffed
        )
        if has_chainable_item:
            return int(Action.SHOOT_SELF)
        # No chainable follow-up → shoot_opp for guaranteed tempo
        if mask[int(Action.SHOOT_OPPONENT)]:
            return int(Action.SHOOT_OPPONENT)
        return int(Action.SHOOT_SELF)
    # Fallback if for some reason the preferred action is masked
    if mask[int(Action.SHOOT_OPPONENT)]:
        return int(Action.SHOOT_OPPONENT)
    if mask[int(Action.SHOOT_SELF)]:
        return int(Action.SHOOT_SELF)
    return -1  # caller must handle


def _strong_baseline_decision(
    obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> int:
    """Core decision logic. Separated so we can reuse it during an
    adrenaline-triggered "effectively-stole-an-item" follow-up without
    re-entering the full dispatch (adrenaline handled at top level)."""

    n_shells = int(round(float(obs[_O_N_SHELLS])))
    my_hp = int(round(float(obs[_O_MY_HP])))
    my_max = int(round(float(obs[_O_MY_MAX_HP])))
    opp_hp = int(round(float(obs[_O_OPP_HP])))
    dmg_mult = int(round(float(obs[_O_DAMAGE_MULT])))
    adren = obs[_O_ADRENALINE_ACTIVE] > 0.5
    opp_cuffed = obs[_O_OPP_CUFFED] > 0.5

    slot0_known, slot0_live = _slot0_known(obs)
    p_live = _prob_live(obs)

    # ============================================================
    # ADRENALINE-PICK MODE: only PICK_<item> actions are legal.
    # Decide which opp item to steal based on current state.
    # ============================================================
    if adren:
        # Order of preference when adrenaline_active:
        # 1. HANDSAW if we can lethal with it and next is live
        # 2. HANDCUFF if we have a lethal plan and opp isn't cuffed
        # 3. SMOKE if we're hurt
        # 4. INVERTER if slot0 is known live and we need to save our life
        # 5. BEER if slot0 is known live and we'd otherwise die
        # 6. GLASS if chamber uncertain
        # 7. PHONE if chamber uncertain
        # 8. PILLS last resort
        # Fall back to any legal pick.
        want_saw_first = (
            mask[int(Action.PICK_HANDSAW)]
            and slot0_known and slot0_live
            and dmg_mult < 2
            and _can_one_shot_opp(obs, with_saw=True)
        )
        if want_saw_first:
            return int(Action.PICK_HANDSAW)
        # Cuff: only worth stealing+using when there will actually BE a follow-up
        # turn for opp to skip. If the next shot already kills opp (directly or via
        # our saw), the cuff is consumed but never fires — opp dies before their
        # next turn. So only steal cuff when the kill is at least 2 shots out.
        kill_in_one_now = slot0_known and slot0_live and _can_one_shot_opp(obs, with_saw=False)
        kill_in_one_with_saw = (
            slot0_known and slot0_live and dmg_mult < 2
            and _can_one_shot_opp(obs, with_saw=True)
            and (_me_has(obs, Item.HANDSAW) or mask[int(Action.PICK_HANDSAW)])
        )
        # Saw advances kill but doesn't finish (opp_hp 3 or 4, saw → 1 or 2):
        # cuff IS useful here because we need a second shot to finish.
        kill_in_two_with_saw = (
            slot0_known and slot0_live and dmg_mult < 2
            and (_me_has(obs, Item.HANDSAW) or mask[int(Action.PICK_HANDSAW)])
            and opp_hp - 2 >= 1 and opp_hp - 2 <= 2
        )
        if (
            mask[int(Action.PICK_HANDCUFF)]
            and not opp_cuffed
            and kill_in_two_with_saw
            and not (kill_in_one_now or kill_in_one_with_saw)
        ):
            return int(Action.PICK_HANDCUFF)
        # Smoke when hurt
        if mask[int(Action.PICK_SMOKE)] and my_hp < my_max:
            return int(Action.PICK_SMOKE)
        # Defensive: if known-live next shell and we're about to die, grab inverter
        if mask[int(Action.PICK_INVERTER)] and slot0_known and slot0_live and my_hp <= max(1, dmg_mult):
            return int(Action.PICK_INVERTER)
        if mask[int(Action.PICK_BEER)] and slot0_known and slot0_live and my_hp <= max(1, dmg_mult):
            return int(Action.PICK_BEER)
        # Info value when chamber uncertain
        if not slot0_known and n_shells >= 2:
            if mask[int(Action.PICK_GLASS)]:
                return int(Action.PICK_GLASS)
            if mask[int(Action.PICK_PHONE)] and n_shells >= 3:
                return int(Action.PICK_PHONE)
        # Grab anything useful even in stable state
        if mask[int(Action.PICK_HANDCUFF)] and not opp_cuffed:
            return int(Action.PICK_HANDCUFF)
        if mask[int(Action.PICK_HANDSAW)] and dmg_mult < 2 and p_live >= 0.5:
            return int(Action.PICK_HANDSAW)
        if mask[int(Action.PICK_GLASS)]:
            return int(Action.PICK_GLASS)
        if mask[int(Action.PICK_PHONE)]:
            return int(Action.PICK_PHONE)
        if mask[int(Action.PICK_BEER)] and p_live < 0.5:
            return int(Action.PICK_BEER)
        # Truly nothing fits — take whatever; PICK_PILLS last resort.
        for a in range(int(Action.PICK_HANDSAW), int(Action.PICK_INVERTER) + 1):
            if mask[a] and a != int(Action.PICK_PILLS):
                return a
        if mask[int(Action.PICK_PILLS)]:
            return int(Action.PICK_PILLS)
        return _fallback(mask, rng)

    # ============================================================
    # LETHAL-NOW: next shell known LIVE and we can immediately kill.
    # Priority: saw (if it's the lethal piece) → shoot.
    # NOTE: Do NOT cuff before a kill — opp dies on the shot and the cuff's
    # skip_next_turn never fires. Cuff is only worth using when opp will
    # survive long enough to have a turn skipped (handled in (b') below).
    # ============================================================
    if slot0_known and slot0_live:
        # (a) Already primed with saw (or raw dmg) to kill
        if _can_one_shot_opp(obs, with_saw=False) and mask[int(Action.SHOOT_OPPONENT)]:
            return int(Action.SHOOT_OPPONENT)
        # (b) Saw makes this lethal — top priority. No cuff: opp dies on the shot.
        if (
            mask[int(Action.USE_HANDSAW)]
            and dmg_mult < 2
            and _can_one_shot_opp(obs, with_saw=True)
        ):
            return int(Action.USE_HANDSAW)
        # (b') Saw advances the kill (opp_hp 3 or 4 → drops to 1 or 2 after
        # 2-dmg shot). NOT wasted — sets up a one-shot next turn. Skip when
        # opp_hp is so high that 2 dmg doesn't change the picture much (e.g. 4 → 2,
        # still need another shot anyway → fine), but DO use saw when it makes
        # opp lethal-on-one-shot for next turn.
        if (
            mask[int(Action.USE_HANDSAW)]
            and dmg_mult < 2
            and opp_hp - 2 >= 1   # not wasting all of saw on excess
            and opp_hp - 2 <= 2   # progresses to one-shot range
        ):
            if (
                mask[int(Action.USE_HANDCUFF)]
                and not opp_cuffed
                and n_shells >= 2
            ):
                return int(Action.USE_HANDCUFF)
            return int(Action.USE_HANDSAW)
        # (c) Can't kill this shot — do we need to survive?
        # If opp can't be killed and we're 1HP, we lose this exchange. Consider
        # defensive items before committing to a live shot.
        # NOTE: dmg_mult is OUR saw multiplier (from our HANDSAW). The engine
        # resets damage_mult to 1 after a shot fires, so whatever opp does next
        # turn starts from mult=1 — their retaliation deals 1 dmg even if we
        # sawed. So the survival check is just my_hp <= 1.
        will_die_if_opp_shoots_next = my_hp <= 1
        cannot_kill_opp = not _can_one_shot_opp(obs, with_saw=False)
        if cannot_kill_opp and will_die_if_opp_shoots_next:
            # BEER ejects the live shell entirely → survives
            if mask[int(Action.USE_BEER)]:
                return int(Action.USE_BEER)
            # INVERTER flips slot0 live→blank → self-shoot for free turn
            if mask[int(Action.USE_INVERTER)]:
                return int(Action.USE_INVERTER)
            # HANDCUFF buys a full round before opp can retaliate
            if mask[int(Action.USE_HANDCUFF)] and not opp_cuffed:
                return int(Action.USE_HANDCUFF)
            # ADRENALINE: try to steal opp beer/inverter to survive
            if mask[int(Action.USE_ADRENALINE)]:
                # Only worth burning adrenaline if the opp actually has a survival tool
                if (
                    _opp_has(obs, Item.BEER)
                    or _opp_has(obs, Item.INVERTER)
                    or _opp_has(obs, Item.SMOKE)
                    or _opp_has(obs, Item.HANDCUFF)
                    or _opp_has(obs, Item.HANDSAW)  # could use saw for lethal
                ):
                    return int(Action.USE_ADRENALINE)
        # (d) Opp can't kill us next turn → heal/info first if possible, then shoot.
        if mask[int(Action.USE_SMOKE)] and my_hp < my_max:
            return int(Action.USE_SMOKE)
        if mask[int(Action.USE_HANDSAW)] and dmg_mult < 2:
            # Don't saw unless it's meaningful: only when it moves opp closer to death
            # (not wasted on excess damage). Since we can't one-shot even with saw,
            # saw still doubles the damage → worth it if it gets opp into one-shot range
            # for NEXT turn. Conservatively: use saw only if it takes opp HP below our
            # current one-shot threshold on the follow-up.
            opp_hp_after_saw_shot = opp_hp - 2
            if opp_hp_after_saw_shot > 0 and opp_hp_after_saw_shot <= 2:
                # Saw burns now, shoots for 2, opp at 1HP next → next turn easier.
                return int(Action.USE_HANDSAW)
        # (e) Cuff before shot for tempo if we have follow-up damage
        if (
            mask[int(Action.USE_HANDCUFF)]
            and not opp_cuffed
            and n_shells >= 2
            and opp_hp <= 2  # a two-shot sequence plausibly kills
        ):
            return int(Action.USE_HANDCUFF)
        # (f) Adrenaline for saw if opp has it
        if (
            mask[int(Action.USE_ADRENALINE)]
            and _opp_has(obs, Item.HANDSAW)
            and dmg_mult < 2
            and _can_one_shot_opp(obs, with_saw=True)
        ):
            return int(Action.USE_ADRENALINE)
        # (g) Default: live → shoot opp
        if mask[int(Action.SHOOT_OPPONENT)]:
            return int(Action.SHOOT_OPPONENT)
        return _fallback(mask, rng)

    # ============================================================
    # KNOWN BLANK: slot 0 is definitely blank → free self-shot (no damage,
    # keep turn, burn a blank shell). This is nearly always correct unless
    # there's a useful item to burn on the blank (rare). Skip any saw (wasted
    # on blank — saw resets after a blank shot too).
    # ============================================================
    if slot0_known and slot0_live is False:
        # INVERTER on a known-blank slot 0 = "free upgrade": guaranteed live
        # next. Always worth using if we can shoot opp profitably — the
        # forfeited free-self-shot is worth less than guaranteed offensive
        # damage. Skip only if blank-self-shot would convert to a critical
        # tempo we'd lose (almost never).
        if mask[int(Action.USE_INVERTER)]:
            # Invert blank→live, then shoot opp for guaranteed dmg_mult damage.
            # Always worth trading the "free self-shot" here: lethal kills
            # (with/without saw) and general chip-damage all fall under the
            # same decision, so the earlier lethality-specific branches were
            # unreachable duplicates.
            return int(Action.USE_INVERTER)
        # If opp can be lethal'd next turn and we can cuff now, stall.
        # (Note: slot 0 will reveal as blank after self-shot; subsequent
        # positions may still be known via phone/glass, but we don't bank on it.)
        if (
            mask[int(Action.USE_HANDCUFF)]
            and not opp_cuffed
            and n_shells >= 2
            and opp_hp <= dmg_mult * 2
            and (_me_has(obs, Item.HANDSAW) or dmg_mult >= 2)
        ):
            return int(Action.USE_HANDCUFF)
        # ADRENALINE: steal opp's inverter to convert this blank into a kill.
        # Saw justification only counts if WE own a saw — adrenaline was already
        # spent stealing the inverter, so we can't also steal opp's saw on the
        # same turn. Without our own saw, the post-invert shot deals 1 damage,
        # not 2 — and we'd burn both adrenaline and the free blank self-shot.
        if mask[int(Action.USE_ADRENALINE)] and _opp_has(obs, Item.INVERTER):
            if _can_one_shot_opp(obs, with_saw=False) or (
                _me_has(obs, Item.HANDSAW)
                and dmg_mult < 2
                and _can_one_shot_opp(obs, with_saw=True)
            ):
                return int(Action.USE_ADRENALINE)
        # Default: burn the blank safely (free turn). Don't smoke here — smoke
        # before a free turn wastes the free turn. After self-shot we'll get
        # another chance to smoke.
        if mask[int(Action.SHOOT_SELF)]:
            return int(Action.SHOOT_SELF)
        if mask[int(Action.SHOOT_OPPONENT)]:
            return int(Action.SHOOT_OPPONENT)
        return _fallback(mask, rng)

    # ============================================================
    # UNKNOWN slot 0: decide whether to gather info, heal, or commit.
    # ============================================================

    # (1) CRITICAL-SURVIVAL: we're at 1HP and opp could plausibly kill. Use a
    # survival item if available rather than gambling on p_live > 0.5.
    will_die_if_live = my_hp <= max(1, dmg_mult)
    if will_die_if_live and p_live >= 0.5:
        # Try to gather info FIRST so we know whether we even need to defend
        if mask[int(Action.USE_GLASS)] and n_shells >= 2:
            return int(Action.USE_GLASS)
        if mask[int(Action.USE_PHONE)] and n_shells >= 3:
            return int(Action.USE_PHONE)
        # Smoke is a great cushion if available
        if mask[int(Action.USE_SMOKE)] and my_hp < my_max:
            return int(Action.USE_SMOKE)
        # ADRENALINE to steal info/survival
        if mask[int(Action.USE_ADRENALINE)]:
            if (
                _opp_has(obs, Item.GLASS)
                or _opp_has(obs, Item.PHONE)
                or _opp_has(obs, Item.SMOKE)
                or _opp_has(obs, Item.BEER)
                or _opp_has(obs, Item.INVERTER)
            ):
                return int(Action.USE_ADRENALINE)

    # (2) Info gathering when slot is uncertain and shells remain. Glass is
    # strictly better than Phone for slot 0 (Phone reveals a random position).
    # Trigger when info is meaningful: at least 2 shells, p_live in genuine
    # uncertainty range, OR opp/we are in lethal range and we want certainty.
    info_value_high = (
        n_shells >= 2
        and 0.25 <= p_live <= 0.85  # genuine uncertainty (extended upper bound)
    )
    # Don't gather info if we're on the verge of winning anyway and confident
    if opp_hp <= dmg_mult and p_live >= 0.7:
        info_value_high = False

    # Special case: opp in one-shot range AND we're uncertain → glass is gold.
    # If we glass and it's live → confirmed lethal. If blank → free turn.
    opp_in_one_shot = _can_one_shot_opp(obs, with_saw=False) or (
        _me_has(obs, Item.HANDSAW) and dmg_mult < 2 and _can_one_shot_opp(obs, with_saw=True)
    )
    if opp_in_one_shot and 0.2 <= p_live <= 0.85 and n_shells >= 2:
        if mask[int(Action.USE_GLASS)]:
            return int(Action.USE_GLASS)

    if info_value_high:
        if mask[int(Action.USE_GLASS)]:
            return int(Action.USE_GLASS)
        if mask[int(Action.USE_PHONE)] and n_shells >= 3:
            return int(Action.USE_PHONE)

    # (3) Heal when damaged. Lower-priority than info because info can avoid
    # damage entirely. Higher-priority before committing to a shot when we're
    # hurt (HP buffer vs lethal). Skip if it'd waste tempo (we're at full or
    # near-max).
    if mask[int(Action.USE_SMOKE)]:
        # Always smoke if we have multiple smokes and aren't full
        if my_hp <= my_max // 2 and my_hp < my_max:
            return int(Action.USE_SMOKE)
        # Or if we just need more buffer before a likely-live shot
        if my_hp < my_max and p_live >= 0.5:
            return int(Action.USE_SMOKE)

    # (4) Adrenaline when opp has tools we lack and the situation is uncertain
    if mask[int(Action.USE_ADRENALINE)]:
        # Steal info if uncertain and we have none
        if (
            0.3 < p_live < 0.7
            and n_shells >= 2
            and not _me_has(obs, Item.GLASS)
            and not _me_has(obs, Item.PHONE)
            and (_opp_has(obs, Item.GLASS) or _opp_has(obs, Item.PHONE))
        ):
            return int(Action.USE_ADRENALINE)
        # Steal smoke if injured and we have none
        if (
            my_hp <= my_max // 2
            and not _me_has(obs, Item.SMOKE)
            and _opp_has(obs, Item.SMOKE)
        ):
            return int(Action.USE_ADRENALINE)
        # Steal saw for a high-prob lethal setup
        if (
            dmg_mult < 2
            and _opp_has(obs, Item.HANDSAW)
            and not _me_has(obs, Item.HANDSAW)
            and _can_one_shot_opp(obs, with_saw=True)
            and p_live > 0.6  # good odds it actually connects
        ):
            return int(Action.USE_ADRENALINE)

    # (5) Saw before a likely-live lethal shot. The kill payoff (winning the
    # game) makes saw worth burning even at moderate p_live ≥ 0.5 when saw
    # turns the next shot into a one-shot kill.
    if (
        mask[int(Action.USE_HANDSAW)]
        and dmg_mult < 2
        and not _can_one_shot_opp(obs, with_saw=False)  # otherwise saw is wasted
    ):
        # (5a) Saw makes it a one-shot kill — fire at p_live ≥ 0.5
        if p_live >= 0.5 and _can_one_shot_opp(obs, with_saw=True):
            return int(Action.USE_HANDSAW)
        # (5b) Saw advances opp into one-shot range — needs higher p_live
        if p_live >= 0.7 and 1 <= opp_hp - 2 <= 2:
            return int(Action.USE_HANDSAW)

    # (6) Cuff if we have a real follow-up that lethals opp; skip otherwise.
    # Conditions:
    #   - opp not already cuffed; multiple shells remain.
    #   - p_live decent (cuff is wasted if we self-shoot anyway and opp gets to act).
    #   - opp is in plausible reach: one-shot now, one-shot with saw, or
    #     two-shots (cuff buys us 2 consecutive shots).
    if (
        mask[int(Action.USE_HANDCUFF)]
        and not opp_cuffed
        and n_shells >= 2
        and p_live >= 0.5
    ):
        # Direct: any path to kill within 2 of our shots
        can_kill_in_one = _can_one_shot_opp(obs, with_saw=False) or (
            _me_has(obs, Item.HANDSAW) and dmg_mult < 2 and _can_one_shot_opp(obs, with_saw=True)
        )
        # Two consecutive 1-damage shots can kill a 2HP opp
        can_kill_in_two = opp_hp <= dmg_mult + 1
        if can_kill_in_one or can_kill_in_two:
            return int(Action.USE_HANDCUFF)


    # (7) PILLS: explicitly avoid. 60% chance of -1 HP (often suicide), 40% +2.
    # We only consider pills as a true last resort. The cascade naturally
    # falls through to a shot below; pills is never chosen.

    # (8) Commit to a shot based on p_live (Nash-aligned).
    shoot = _tiebreak_shoot(obs, mask, p_live)
    if shoot >= 0:
        return shoot
    return _fallback(mask, rng)


def strong_baseline_opponent(
    obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> int:
    """A carefully-tuned rule-based Buckshot Roulette AI.

    Strategy outline (cascade of priority gates; first match wins):
      1. Adrenaline-pick mode → pick the opp item that maximises current
         turn's value (saw for lethal, smoke for heal, info if uncertain).
      2. Next shell known LIVE:
           - can one-shot opp (with saw or raw) → cuff/saw then shoot_opp
           - can't kill this turn AND we'd die next → USE_BEER / USE_INVERTER
             / USE_HANDCUFF / USE_ADRENALINE (in that priority)
           - otherwise smoke/saw setup then shoot_opp
      3. Next shell known BLANK:
           - smoke if hurt, cuff if lethal plan, INVERTER if can convert
           - else SHOOT_SELF for free turn
      4. Unknown slot 0:
           - heal-critical, info-gather (glass > phone), heal if wounded
           - adrenaline to steal missing info/smoke/saw
           - saw/cuff pre-commit if p_live high enough and non-wasted
           - shoot per Nash rule: p_live >= 0.5 → shoot_opp else shoot_self

    Design notes:
      * Uses only obs + mask + rng; no peeking at engine internals. All
        shell knowledge comes from the observation's known_live/blank
        blocks (positions 31..46).
      * Never wastes HANDSAW on a known blank or on already-lethal damage —
        a concrete anti-pattern observed in prior heuristics and early PPO.
      * Never uses PILLS (60% self-damage) — the expected value is negative
        in almost every legal state; we return PILLS only as a hard fallback.
      * On a 50/50 ties breaks toward SHOOT_OPPONENT (Nash-aligned per
        rl/docs/research_notes.md and CFR on simple_buckshot).
      * BEER/INVERTER on known-live is the 'survival' play the research notes
        flagged as the gap even 5M-step PPO champions miss.
    """
    action = _strong_baseline_decision(obs, mask, rng)
    if action < 0 or not mask[action]:
        return _fallback(mask, rng)
    return int(action)


strong_baseline_opponent.obs_layout = "hack"  # type: ignore[attr-defined]


# Registry for easy lookup
NAMED_OPPONENTS: dict[str, OpponentFn] = {
    "random": random_opponent,
    "aggressive": aggressive_opponent,
    "conservative": conservative_opponent,
    "strong_baseline": strong_baseline_opponent,
}


def make_frozen_policy_opponent(policy, device: str = "cpu") -> OpponentFn:
    """Wraps a torch policy network into an opponent function.

    The policy must expose `.act(obs_tensor, mask_tensor) -> action: int`.
    A fresh forward pass is run per call — fine for training collectors but
    avoid calling on huge batches; vectorize at the env level instead.

    The returned callable sets `.obs_layout` based on the policy's input
    width (47 → 'hack', 52 → 'honest'). `SingleAgentBuckshotEnv` reads this
    attribute and asks the engine for the matching layout, so the opponent
    always receives exactly the obs shape the policy was trained on.
    """
    import torch

    expected_dim = int(policy.body[0].in_features)
    layout = "honest" if expected_dim == _HONEST_OBS_DIM else "hack"

    def fn(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        if obs.shape[0] != expected_dim:
            raise ValueError(
                f"FF opponent expects obs_dim={expected_dim} but got "
                f"{obs.shape[0]} — env should have routed {layout} layout."
            )
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)
            action = policy.act(obs_t, mask_t)
        return int(action.item())

    fn.obs_layout = layout  # type: ignore[attr-defined]
    return fn


class _FrozenRecurrentOpponent:
    """Stateful callable wrapping a RecurrentActorCritic for opponent play.

    One instance per env-episode — hidden state is reset per episode via
    the factory (a new instance is created each time env.reset samples
    this opponent). Do NOT share a single instance across envs: their
    hidden states would clobber each other.
    """

    def __init__(self, policy, device: str = "cpu") -> None:
        self.policy = policy
        self.device = device
        self._h = None  # lazy-init on first call once we know batch=1
        self._expected_dim = int(policy.obs_dim)
        # Declared to the env so it can fetch a matching-layout obs.
        self.obs_layout = "honest" if self._expected_dim == _HONEST_OBS_DIM else "hack"

    def __call__(
        self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> int:
        import torch

        if obs.shape[0] != self._expected_dim:
            raise ValueError(
                f"Recurrent opponent expects obs_dim={self._expected_dim} "
                f"but got {obs.shape[0]} — env should have routed "
                f"{self.obs_layout} layout."
            )
        obs_t = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        mask_t = torch.from_numpy(mask).to(self.device).unsqueeze(0)
        if self._h is None:
            self._h = self.policy.initial_hidden(1, device=self.device)
        # Opponent is called only on its own turns inside one episode, so
        # there's no "previous step ended episode" event to signal here.
        done_prev = torch.zeros(1, device=self.device)
        action, new_h = self.policy.act_stateful(obs_t, mask_t, self._h, done_prev)
        self._h = new_h
        return int(action.item())


def make_frozen_recurrent_opponent_factory(policy, device: str = "cpu") -> Callable:
    """Returns a FACTORY that produces a fresh stateful opponent per env-episode.

    `OpponentPool` stores the factory as if it were an opponent fn; the
    SingleAgentBuckshotEnv detects the `_is_factory` marker on reset and
    calls it to get an independent instance. This avoids hidden-state
    collisions between parallel envs in SyncVectorEnv that happen to
    sample the same recurrent snapshot.
    """

    layout = "honest" if int(policy.obs_dim) == _HONEST_OBS_DIM else "hack"

    def factory() -> _FrozenRecurrentOpponent:
        return _FrozenRecurrentOpponent(policy, device=device)

    factory._is_factory = True  # type: ignore[attr-defined]
    factory.obs_layout = layout  # type: ignore[attr-defined]
    return factory


class OpponentPool:
    """Sampling pool for self-play league.

    Holds a set of named opponents (functions). `sample()` returns one
    according to weights — used by the env at reset time to pick who the
    agent plays this episode.

    Thread-safety: all reads/writes on the pool are serialised through an
    internal `threading.Lock`. Today we only use SyncVectorEnv (synchronous,
    GIL-serialised), so correctness doesn't depend on the lock; it's there
    so that switching to AsyncVectorEnv (or sharing a pool across worker
    threads) doesn't introduce a data race between `add`/`remove` and
    `sample`.
    """

    def __init__(self, base_opponents: Optional[dict[str, OpponentFn]] = None) -> None:
        import threading
        self.opponents: dict[str, OpponentFn] = dict(base_opponents or {})
        self.weights: dict[str, float] = {name: 1.0 for name in self.opponents}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        # threading.Lock is not picklable; AsyncVectorEnv(context="spawn")
        # pickles the pool through the worker setup closure. Drop the lock
        # on serialize and re-create it in the child process below.
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        import threading
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def add(self, name: str, fn: OpponentFn, weight: float = 1.0) -> None:
        with self._lock:
            self.opponents[name] = fn
            self.weights[name] = weight

    def remove(self, name: str) -> None:
        with self._lock:
            self.opponents.pop(name, None)
            self.weights.pop(name, None)

    def sample(self, rng: np.random.Generator) -> tuple[str, OpponentFn]:
        with self._lock:
            names = list(self.opponents.keys())
            if not names:
                raise ValueError("OpponentPool is empty")
            ws = np.array([self.weights[n] for n in names], dtype=np.float64)
            fns = [self.opponents[n] for n in names]
        ws /= ws.sum()
        idx = int(rng.choice(len(names), p=ws))
        return names[idx], fns[idx]

    def __len__(self) -> int:
        with self._lock:
            return len(self.opponents)

    def manifest(self) -> list[tuple[str, str, float, str]]:
        """Serializable description of the pool for AsyncVectorEnv worker sync.

        Returns a list of `(name, kind_spec, weight, obs_layout)` tuples where
        `kind_spec` is one of:
          - `"rule:<named_key>"` — a rule-based opponent registered in
            `NAMED_OPPONENTS`, identified by its key in that registry.
          - `"ckpt:<abs_path>"` — a frozen policy-based opponent whose
            weights live on disk at the given path; workers rebuild it by
            calling `load_recurrent_policy` (or `load_policy` for FF).

        `obs_layout` is "honest" or "hack" (read from the stored callable's
        `.obs_layout` attr that every policy wrapper sets). Rule-based
        opponents get layout "agnostic" — they don't care.

        Why this shape: the manifest needs to round-trip through a pickle
        call across a pipe to subprocess workers that do NOT share Python
        state with the parent. A pure-data tuple survives that trip
        regardless of the closures the parent happens to hold.

        Trained recurrent snapshots added by `ppo_recurrent.train()` carry
        their checkpoint path on the factory as `.ckpt_path` (set by the
        snapshot-save block). Opponents without that attribute are skipped
        with a warning — they cannot be reconstructed without code changes.
        """
        with self._lock:
            rows: list[tuple[str, str, float, str]] = []
            for name, fn in self.opponents.items():
                w = self.weights[name]
                layout = getattr(fn, "obs_layout", "agnostic")
                # Rule-based: find by identity in NAMED_OPPONENTS
                matched_rule = None
                for rule_name, rule_fn in NAMED_OPPONENTS.items():
                    if fn is rule_fn:
                        matched_rule = rule_name
                        break
                if matched_rule is not None:
                    rows.append((name, f"rule:{matched_rule}", w, "agnostic"))
                    continue
                # Policy-based: needs a ckpt_path annotation set by the
                # code that added it to the pool.
                ckpt = getattr(fn, "ckpt_path", None)
                if ckpt is None:
                    # Can't serialize; skip with a warning so caller knows.
                    print(
                        f"[manifest] WARNING: opponent '{name}' has no "
                        f"ckpt_path annotation and is not in NAMED_OPPONENTS; "
                        f"it will NOT be visible to AsyncVectorEnv workers."
                    )
                    continue
                rows.append((name, f"ckpt:{ckpt}", w, layout))
            return rows


def build_pool_from_manifest(
    manifest: list[tuple[str, str, float, str]],
    device: str = "cpu",
) -> OpponentPool:
    """Reconstruct an `OpponentPool` from a manifest produced by
    `OpponentPool.manifest()`. Intended for `AsyncVectorEnv` workers.

    Ckpt-based entries are loaded from disk via `load_recurrent_policy`
    (if the saved weights include `gru.weight_ih_l0`) or `load_policy`
    for the feedforward case. All policies are forced onto `device`
    (default "cpu") regardless of where the parent had them — env
    stepping in workers is CPU-bound, and the parent keeps training
    policy on its GPU.

    Kept as a top-level function (not a classmethod) so it's
    importable from both `ppo_recurrent.py` and the worker-side env.
    """
    import torch
    from rl.engine import NUM_ACTIONS
    from rl.policy import load_policy, load_recurrent_policy

    pool = OpponentPool()
    for name, spec, weight, layout in manifest:
        if spec.startswith("rule:"):
            rule_key = spec[len("rule:"):]
            fn = NAMED_OPPONENTS.get(rule_key)
            if fn is None:
                print(
                    f"[build_pool_from_manifest] WARNING: unknown rule "
                    f"'{rule_key}' for '{name}'; skipping."
                )
                continue
            pool.add(name, fn, weight=weight)
            continue
        if spec.startswith("ckpt:"):
            path = spec[len("ckpt:"):]
            try:
                state = torch.load(path, map_location=device, weights_only=True)
                if "gru.weight_ih_l0" in state:
                    net = load_recurrent_policy(path, NUM_ACTIONS, device=device)
                    for p in net.parameters():
                        p.requires_grad_(False)
                    fac = make_frozen_recurrent_opponent_factory(net, device=device)
                    fac.ckpt_path = path  # type: ignore[attr-defined]
                    pool.add(name, fac, weight=weight)
                else:
                    net = load_policy(path, NUM_ACTIONS, device=device)
                    for p in net.parameters():
                        p.requires_grad_(False)
                    fn = make_frozen_policy_opponent(net, device=device)
                    fn.ckpt_path = path  # type: ignore[attr-defined]
                    pool.add(name, fn, weight=weight)
            except Exception as exc:
                print(
                    f"[build_pool_from_manifest] WARNING: could not load "
                    f"'{name}' from {path}: {exc}"
                )
            continue
        print(
            f"[build_pool_from_manifest] WARNING: unknown spec '{spec}' "
            f"for '{name}'; skipping."
        )
    return pool
