"""Tests for ActorCritic policy. Run: python -m rl.test_policy"""

from __future__ import annotations

import sys

import numpy as np
import torch

from rl.engine import NUM_ACTIONS
from rl.policy import ActorCritic


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_forward_shapes():
    obs_dim = 47
    p = ActorCritic(obs_dim, NUM_ACTIONS)
    obs = torch.randn(8, obs_dim)
    mask = torch.ones(8, NUM_ACTIONS, dtype=torch.int8)
    action, logp, ent, value = p.get_action_and_value(obs, mask)
    _assert(action.shape == (8,), f"action shape {action.shape}")
    _assert(logp.shape == (8,), f"logp shape {logp.shape}")
    _assert(ent.shape == (8,), f"ent shape {ent.shape}")
    _assert(value.shape == (8,), f"value shape {value.shape}")
    print("ok  forward_shapes")


def test_action_masking_actually_masks():
    """With only one legal action, the policy must always pick it."""
    obs_dim = 47
    p = ActorCritic(obs_dim, NUM_ACTIONS)
    obs = torch.randn(64, obs_dim)
    mask = torch.zeros(64, NUM_ACTIONS, dtype=torch.int8)
    target_action = 7
    mask[:, target_action] = 1
    action = p.act(obs, mask)
    _assert((action == target_action).all(), f"Expected all actions == {target_action}, got {action.unique().tolist()}")
    # And the log-prob of the legal action should be ~0 (probability ~1)
    _, logp, _, _ = p.get_action_and_value(obs, mask, action)
    _assert((logp.abs() < 1e-5).all(), f"log-prob of forced action should be ~0, got max {logp.abs().max().item()}")
    print("ok  action_masking_actually_masks")


def test_gradients_flow():
    obs_dim = 47
    net = ActorCritic(obs_dim, NUM_ACTIONS)
    obs = torch.randn(16, obs_dim)
    mask = torch.ones(16, NUM_ACTIONS, dtype=torch.int8)
    action, logp, ent, value = net.get_action_and_value(obs, mask)
    loss = -logp.mean() + value.pow(2).mean() - 0.01 * ent.mean()
    loss.backward()
    grad_norms = [param.grad.norm().item() for param in net.parameters() if param.grad is not None]
    _assert(all(g > 0 for g in grad_norms), "Some grads are zero — gradient flow issue")
    print(f"ok  gradients_flow (mean_grad_norm={np.mean(grad_norms):.4f})")


def test_act_is_deterministic_under_torch_seed():
    obs_dim = 47
    torch.manual_seed(0)
    p = ActorCritic(obs_dim, NUM_ACTIONS)
    obs = torch.randn(4, obs_dim)
    mask = torch.ones(4, NUM_ACTIONS, dtype=torch.int8)
    torch.manual_seed(42)
    a1 = p.act(obs, mask)
    torch.manual_seed(42)
    a2 = p.act(obs, mask)
    _assert((a1 == a2).all(), "act() not reproducible under fixed seed")
    print("ok  act_is_deterministic_under_torch_seed")


def main() -> int:
    tests = [
        test_forward_shapes,
        test_action_masking_actually_masks,
        test_gradients_flow,
        test_act_is_deterministic_under_torch_seed,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            import traceback
            print(f"ERR  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
