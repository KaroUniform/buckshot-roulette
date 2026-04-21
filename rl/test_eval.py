"""Tests for rl.eval — focused on the recurrent opponent-embedding path.

The pre-fix bug (bugbot 12ff10d6): evaluate_recurrent_policy never passed
opp_ids to act_stateful, so policies trained with the E19 opponent
embedding evaluated with embedding contribution = 0. This file pins
the new contract:
  - For policies WITH n_opponents > 0, eval calls act_stateful with
    opp_ids=tensor([opp_id]).
  - For policies WITHOUT (n_opponents = 0), eval still works and passes
    opp_ids=None (or omits the kwarg) — backward compatible.
"""

from __future__ import annotations

import sys

import torch

from rl.engine import NUM_ACTIONS
from rl.eval import evaluate_recurrent_policy
from rl.opponents import NAMED_OPPONENTS
from rl.policy import RecurrentActorCritic


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_eval_with_embedding_passes_opp_ids():
    """Wrap a recurrent policy and intercept act_stateful calls to verify
    opp_ids is a long tensor (not None) when the policy was constructed
    with n_opponents > 0."""

    class SpyPolicy:
        def __init__(self, inner):
            self.inner = inner
            self.n_opponents = inner.n_opponents
            self.opp_ids_seen: list = []

        def initial_hidden(self, b, device):
            return self.inner.initial_hidden(b, device=device)

        def act_stateful(self, obs, mask, h, done_prev, opp_ids=None):
            self.opp_ids_seen.append(opp_ids)
            return self.inner.act_stateful(obs, mask, h, done_prev, opp_ids=opp_ids)

    inner = RecurrentActorCritic(
        obs_dim=47, n_actions=NUM_ACTIONS, hidden=32, n_opponents=5,
    ).eval()
    for p in inner.parameters():
        p.requires_grad_(False)

    spy = SpyPolicy(inner)
    evaluate_recurrent_policy(
        spy,
        opponents={"random": NAMED_OPPONENTS["random"]},
        n_episodes=2,
        seed=7,
        device="cpu",
    )

    _assert(len(spy.opp_ids_seen) > 0, "spy never received any act_stateful calls")
    for opp_ids in spy.opp_ids_seen:
        _assert(opp_ids is not None,
                "opp_ids must be a tensor for embedding-equipped policy, got None")
        _assert(opp_ids.dtype == torch.long,
                f"opp_ids must be long, got {opp_ids.dtype}")
        _assert(opp_ids.shape == (1,),
                f"opp_ids must have shape (1,), got {tuple(opp_ids.shape)}")
    print("ok  eval_with_embedding_passes_opp_ids")


def test_eval_without_embedding_passes_none():
    """Policy without opponent embedding must still work and must NOT
    construct a tensor needlessly — opp_ids stays None."""

    class SpyPolicy:
        def __init__(self, inner):
            self.inner = inner
            self.n_opponents = inner.n_opponents
            self.opp_ids_seen: list = []

        def initial_hidden(self, b, device):
            return self.inner.initial_hidden(b, device=device)

        def act_stateful(self, obs, mask, h, done_prev, opp_ids=None):
            self.opp_ids_seen.append(opp_ids)
            return self.inner.act_stateful(obs, mask, h, done_prev, opp_ids=opp_ids)

    inner = RecurrentActorCritic(
        obs_dim=47, n_actions=NUM_ACTIONS, hidden=32, n_opponents=0,
    ).eval()
    for p in inner.parameters():
        p.requires_grad_(False)

    spy = SpyPolicy(inner)
    evaluate_recurrent_policy(
        spy,
        opponents={"random": NAMED_OPPONENTS["random"]},
        n_episodes=2,
        seed=8,
        device="cpu",
    )

    _assert(len(spy.opp_ids_seen) > 0, "spy never received any act_stateful calls")
    for opp_ids in spy.opp_ids_seen:
        _assert(opp_ids is None,
                f"non-embedding policy should receive opp_ids=None, got {opp_ids}")
    print("ok  eval_without_embedding_passes_none")


def main() -> int:
    tests = [
        test_eval_with_embedding_passes_opp_ids,
        test_eval_without_embedding_passes_none,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
