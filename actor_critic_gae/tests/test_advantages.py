import pytest
import torch
from actor_critic_gae.advantages import discounted_returns, td_advantages, gae_advantages

GAMMA = 0.99


def make_rollout(T=6, N=3, seed=0):
    """Random [T, N] rollout with a termination, a truncation, and both flags at once.

    next_values[t] must equal values[t+1] on every step that is not done,
    otherwise the telescoping in test 2 does not hold. So build one chain of
    T+1 state values and overwrite next_values only where an episode ended.
    """
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(T, N, generator=g)
    chain = torch.randn(T + 1, N, generator=g)
    values = chain[:-1].clone()
    next_values = chain[1:].clone()

    terminated = torch.zeros(T, N, dtype=torch.bool)
    truncated = torch.zeros(T, N, dtype=torch.bool)
    terminated[2, 0] = True
    truncated[3, 1] = True
    terminated[1, 2] = True
    truncated[1, 2] = True            # both flags: termination must win

    done = terminated | truncated
    other = torch.randn(T, N, generator=g)
    next_values[done] = other[done]   # true final state differs from the next row's start
    return rewards, values, next_values, terminated, truncated


def test_lambda_zero_equals_td():
    r, v, nv, term, trunc = make_rollout()
    gae = gae_advantages(r, v, nv, term, trunc, GAMMA, lam=0.0)
    td = td_advantages(r, v, nv, term, GAMMA)
    assert torch.allclose(gae, td, atol=1e-6)


def test_lambda_one_equals_return_minus_value():
    r, v, nv, term, trunc = make_rollout()
    gae = gae_advantages(r, v, nv, term, trunc, GAMMA, lam=1.0)
    G = discounted_returns(r, nv, term, trunc, GAMMA)
    assert torch.allclose(gae, G - v, atol=1e-5)


@pytest.mark.skip(reason="TODO: expected values not written yet")
def test_hand_computed_case():
    # TODO: one environment, T = 3, no episode end, gamma = 0.5, lam = 0.5.
    # Choose simple numbers for rewards, values, next_values, work out
    # delta_0..delta_2 and A_2, A_1, A_0 on paper, then assert against them.
    ...


@pytest.mark.skip(reason="TODO: asserts not written yet")
def test_truncation_keeps_bootstrap_but_cuts_chain():
    # TODO: use make_rollout(), then check two things at the truncated step (3, 1):
    #   (a) changing next_values[3, 1] changes A[3, 1]            -> bootstrap is kept
    #   (b) changing rewards/values in rows after step 3 for env 1
    #       does not change A[3, 1] or anything before it          -> chain is cut
    ...