"""Tests for ActorCritic policy. Run: python -m rl.test_policy"""

from __future__ import annotations

import sys

import numpy as np
import torch

from rl.engine import NUM_ACTIONS
from rl.policy import ActorCritic, RecurrentActorCritic, load_recurrent_policy


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


def test_recurrent_forward_shapes():
    obs_dim = 47
    p = RecurrentActorCritic(obs_dim, NUM_ACTIONS, hidden=64)
    B = 8
    obs = torch.randn(B, obs_dim)
    mask = torch.ones(B, NUM_ACTIONS, dtype=torch.int8)
    h0 = p.initial_hidden(B)
    done_prev = torch.zeros(B)
    action, logp, ent, value, h1 = p.get_action_and_value(obs, mask, h0, done_prev)
    _assert(action.shape == (B,), f"action shape {action.shape}")
    _assert(logp.shape == (B,), f"logp {logp.shape}")
    _assert(ent.shape == (B,), f"ent {ent.shape}")
    _assert(value.shape == (B,), f"value {value.shape}")
    _assert(h1.shape == (1, B, 64), f"h1 {h1.shape}")
    print("ok  recurrent_forward_shapes")


def test_recurrent_resets_hidden_on_done():
    """If done_prev is True, the hidden that goes into GRU must be zero —
    meaning the output depends only on the current obs, regardless of what
    was in h."""
    obs_dim = 47
    p = RecurrentActorCritic(obs_dim, NUM_ACTIONS, hidden=32)
    p.eval()
    B = 4
    obs = torch.randn(B, obs_dim)
    mask = torch.ones(B, NUM_ACTIONS, dtype=torch.int8)
    h_zero = p.initial_hidden(B)
    h_random = torch.randn(1, B, 32)
    done_all = torch.ones(B)
    with torch.no_grad():
        logits_a, val_a, _ = p.forward_step(obs, h_zero, torch.zeros(B))
        logits_b, val_b, _ = p.forward_step(obs, h_random, done_all)
    _assert(torch.allclose(logits_a, logits_b, atol=1e-6),
            "done_prev=1 should zero hidden; logits must equal h=0 case")
    _assert(torch.allclose(val_a, val_b, atol=1e-6),
            "done_prev=1 should zero hidden; values must equal h=0 case")
    print("ok  recurrent_resets_hidden_on_done")


def test_recurrent_sequence_equals_stepwise():
    """forward_sequence over T steps must match T calls to forward_step
    with carried hidden state."""
    obs_dim = 47
    p = RecurrentActorCritic(obs_dim, NUM_ACTIONS, hidden=48)
    p.eval()
    T, B = 7, 3
    obs_seq = torch.randn(T, B, obs_dim)
    dones_seq = torch.zeros(T, B)
    # Inject a couple of mid-sequence resets
    dones_seq[2, 1] = 1.0
    dones_seq[5, 0] = 1.0
    h_init = torch.randn(1, B, 48)
    with torch.no_grad():
        logits_seq, values_seq = p.forward_sequence(obs_seq, h_init, dones_seq)
        # Manual unroll
        h = h_init.clone()
        logits_manual = []
        values_manual = []
        for t in range(T):
            logits_t, val_t, h = p.forward_step(obs_seq[t], h, dones_seq[t])
            logits_manual.append(logits_t)
            values_manual.append(val_t)
        lm = torch.stack(logits_manual, dim=0)
        vm = torch.stack(values_manual, dim=0)
    _assert(torch.allclose(logits_seq, lm, atol=1e-6), "logits seq != manual unroll")
    _assert(torch.allclose(values_seq, vm, atol=1e-6), "values seq != manual unroll")
    print("ok  recurrent_sequence_equals_stepwise")


def test_recurrent_memory_affects_output():
    """Sanity: with different prior context, current-step logits should differ.
    Otherwise memory is wasted capacity."""
    obs_dim = 47
    p = RecurrentActorCritic(obs_dim, NUM_ACTIONS, hidden=32)
    p.eval()
    obs_now = torch.randn(1, obs_dim)
    with torch.no_grad():
        h1 = p.initial_hidden(1)
        # Context A: one prior step of random obs
        _, _, h_a = p.forward_step(torch.randn(1, obs_dim), h1, torch.zeros(1))
        logits_a, _, _ = p.forward_step(obs_now, h_a, torch.zeros(1))
        # Context B: different prior step
        _, _, h_b = p.forward_step(torch.randn(1, obs_dim), h1, torch.zeros(1))
        logits_b, _, _ = p.forward_step(obs_now, h_b, torch.zeros(1))
    _assert(not torch.allclose(logits_a, logits_b, atol=1e-6),
            "Memory of different prior contexts should change current logits")
    print("ok  recurrent_memory_affects_output")


def test_recurrent_gradients_flow():
    obs_dim = 47
    p = RecurrentActorCritic(obs_dim, NUM_ACTIONS, hidden=32)
    T, B = 5, 4
    obs_seq = torch.randn(T, B, obs_dim)
    mask = torch.ones(T, B, NUM_ACTIONS, dtype=torch.int8)
    dones_seq = torch.zeros(T, B)
    h_init = p.initial_hidden(B)
    logits_seq, values_seq = p.forward_sequence(obs_seq, h_init, dones_seq)
    masked = logits_seq.masked_fill(mask == 0, -1e8)
    logp = torch.log_softmax(masked, dim=-1).sum()
    loss = -logp + values_seq.pow(2).mean()
    loss.backward()
    grad_norms = [
        param.grad.norm().item()
        for param in p.parameters()
        if param.grad is not None
    ]
    _assert(all(g > 0 for g in grad_norms),
            "Some recurrent grads are zero — BPTT broken somewhere")
    print(f"ok  recurrent_gradients_flow (mean_grad_norm={np.mean(grad_norms):.4f})")


def test_recurrent_checkpoint_round_trip():
    """Train-time save/load must restore exact behavior."""
    import tempfile, os
    obs_dim = 47
    torch.manual_seed(11)
    p = RecurrentActorCritic(obs_dim, NUM_ACTIONS, hidden=40, embed=56)
    p.eval()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rp.pt")
        torch.save(p.state_dict(), path)
        q = load_recurrent_policy(path, NUM_ACTIONS, device="cpu")
    B = 3
    obs = torch.randn(B, obs_dim)
    mask = torch.ones(B, NUM_ACTIONS, dtype=torch.int8)
    h = p.initial_hidden(B)
    done = torch.zeros(B)
    with torch.no_grad():
        l1, v1, _ = p.forward_step(obs, h, done)
        l2, v2, _ = q.forward_step(obs, h, done)
    _assert(torch.allclose(l1, l2, atol=1e-6), "loaded policy logits drift")
    _assert(torch.allclose(v1, v2, atol=1e-6), "loaded policy values drift")
    _assert(q.hidden == 40 and q.obs_dim == obs_dim,
            f"loader auto-detect wrong: hidden={q.hidden} obs_dim={q.obs_dim}")
    print("ok  recurrent_checkpoint_round_trip")


def main() -> int:
    tests = [
        test_forward_shapes,
        test_action_masking_actually_masks,
        test_gradients_flow,
        test_act_is_deterministic_under_torch_seed,
        test_recurrent_forward_shapes,
        test_recurrent_resets_hidden_on_done,
        test_recurrent_sequence_equals_stepwise,
        test_recurrent_memory_affects_output,
        test_recurrent_gradients_flow,
        test_recurrent_checkpoint_round_trip,
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
