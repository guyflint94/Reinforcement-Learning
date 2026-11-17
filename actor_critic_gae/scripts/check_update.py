import torch
from actor_critic_gae.networks import ActorNet, CriticNet
from actor_critic_gae.rollout import make_vec_env, RolloutCollector
from actor_critic_gae.update import a2c_update

envs = make_vec_env("CartPole-v1", num_envs=8)
actor, critic = ActorNet(4, 2, 32), CriticNet(4, 32)
rollout = RolloutCollector(envs, actor, critic, num_steps=16, seed=0).collect()
opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
a2c_update(actor, critic, opt, rollout, 0.99, 0.95, 0.5, 0.01, 0.5, False)