import torch

from actor_critic_gae.networks import ActorNet, CriticNet

OBS_DIM, N_ACTIONS, HIDDEN = 4, 2, 16


def test_actor_output_shape():
    net = ActorNet(OBS_DIM, N_ACTIONS, HIDDEN)
    out = net(torch.randn(5, OBS_DIM))
    assert out.shape == (5, N_ACTIONS)


def test_critic_output_shape():
    net = CriticNet(OBS_DIM, HIDDEN)
    out = net(torch.randn(5, OBS_DIM))
    assert out.shape == (5,)


def test_actor_and_critic_share_no_parameters():
    actor = ActorNet(OBS_DIM, N_ACTIONS, HIDDEN)
    critic = CriticNet(OBS_DIM, HIDDEN)
    actor_ids = {id(p) for p in actor.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)


def test_critic_is_not_squashed():
    torch.manual_seed(0)
    net = CriticNet(OBS_DIM, HIDDEN)
    out = net(torch.randn(5, OBS_DIM) * 100)
    assert ((out < 0) | (out > 1)).any()
