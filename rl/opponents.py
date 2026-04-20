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


def random_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def aggressive_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Always shoot opponent; prefer handsaw before shooting; otherwise shoot.
    No info-gathering, no self-shots. A "rush" baseline."""
    if mask[int(Action.USE_HANDSAW)] and obs[_O_DAMAGE_MULT] < 2:
        return int(Action.USE_HANDSAW)
    if mask[int(Action.SHOOT_OPPONENT)]:
        return int(Action.SHOOT_OPPONENT)
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


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


def _known_live_positions(obs: np.ndarray, n_shells: int) -> list[int]:
    """Positions (0-indexed) where we know the shell is live."""
    out = []
    for i in range(min(n_shells, _MAX_SHELLS)):
        if obs[_KNOWN_LIVE_OFFSET + i] > 0.5:
            out.append(i)
    return out


def _known_blank_positions(obs: np.ndarray, n_shells: int) -> list[int]:
    """Positions (0-indexed) where we know the shell is blank."""
    out = []
    for i in range(min(n_shells, _MAX_SHELLS)):
        if obs[_KNOWN_BLANK_OFFSET + i] > 0.5:
            out.append(i)
    return out


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


def _me_count(obs: np.ndarray, item: Item) -> int:
    return int(round(float(obs[_INV_OFFSET + int(item)])))


def _opp_count(obs: np.ndarray, item: Item) -> int:
    return int(round(float(obs[_OPP_INV_OFFSET + int(item)])))


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
        # Cuff: free lethal setup — if we know next is live and saw+shoot would
        # kill (or damage_mult already ≥ opp_hp), cuff helps us re-shoot next
        # turn without retaliation. Only worth stealing if opp isn't already cuffed.
        can_lethal_now = slot0_known and slot0_live and _can_one_shot_opp(obs, with_saw=False)
        can_lethal_with_saw = (
            slot0_known and slot0_live and dmg_mult < 2
            and _can_one_shot_opp(obs, with_saw=True)
            and (_me_has(obs, Item.HANDSAW) or mask[int(Action.PICK_HANDSAW)])
        )
        if (
            mask[int(Action.PICK_HANDCUFF)]
            and not opp_cuffed
            and (can_lethal_now or can_lethal_with_saw)
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
    # Priority: cuff (if we have both cuff + the current lethal setup)
    # → saw (only if not wasted) → shoot.
    # ============================================================
    if slot0_known and slot0_live:
        # (a) Already primed with saw (or raw dmg) to kill
        if _can_one_shot_opp(obs, with_saw=False) and mask[int(Action.SHOOT_OPPONENT)]:
            # Cuff first only when we have a clear use for the extra tempo
            # (n_shells>=2 so another live attempt may exist). Skip if opp
            # already cuffed or we're at full HP (no urgency to milk extra dmg).
            if (
                mask[int(Action.USE_HANDCUFF)]
                and not opp_cuffed
                and n_shells >= 2
            ):
                return int(Action.USE_HANDCUFF)
            return int(Action.SHOOT_OPPONENT)
        # (b) Saw makes this lethal — top priority
        if (
            mask[int(Action.USE_HANDSAW)]
            and dmg_mult < 2
            and _can_one_shot_opp(obs, with_saw=True)
        ):
            # Cuff first if we have both
            if (
                mask[int(Action.USE_HANDCUFF)]
                and not opp_cuffed
                and n_shells >= 2
            ):
                return int(Action.USE_HANDCUFF)
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
            # (a) Direct: invert → shoot for guaranteed kill
            if _can_one_shot_opp(obs, with_saw=False):
                return int(Action.USE_INVERTER)
            # (b) Invert + saw + shoot = lethal kill
            if (
                _me_has(obs, Item.HANDSAW)
                and dmg_mult < 2
                and _can_one_shot_opp(obs, with_saw=True)
            ):
                return int(Action.USE_INVERTER)
            # (c) General-purpose offensive use: invert blank→live, shoot opp.
            # Trade a "free self-shot" for guaranteed dmg_mult damage to opp.
            # Worth it whenever opp has HP that we can chip down — i.e.,
            # always while opp is alive. This is what champion does.
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
        # ADRENALINE: steal opp's inverter to convert this blank into a kill
        if mask[int(Action.USE_ADRENALINE)] and _opp_has(obs, Item.INVERTER):
            if _can_one_shot_opp(obs, with_saw=False) or (
                (_me_has(obs, Item.HANDSAW) or _opp_has(obs, Item.HANDSAW))
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
    """
    import torch

    def fn(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)
            action = policy.act(obs_t, mask_t)
        return int(action.item())

    return fn


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
