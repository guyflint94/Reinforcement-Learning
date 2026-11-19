import torch

from actor_critic_gae.networks import ActorNet, CriticNet
from actor_critic_gae.rollout import RolloutCollector, make_vec_env

N, T = 8, 128
torch.manual_seed(0)
actor, critic = ActorNet(4, 2, 32), CriticNet(4, 32)

# Checks 1 and 2: shapes, dtypes, no graph, episode accounting
envs = make_vec_env("CartPole-v1", N)
collector = RolloutCollector(envs, actor, critic, T, seed=0)
n_flags, returns = 0, []
for _ in range(20):
    r = collector.collect()
    n_flags += int((r.terminated | r.truncated).sum())
    returns += r.episode_returns
print("shapes:", tuple(r.obs.shape), tuple(r.actions.shape), tuple(r.next_values.shape))
print("dtypes:", r.obs.dtype, r.actions.dtype, r.terminated.dtype)
print("any tensor requires grad:",
      any(x.requires_grad for x in (r.obs, r.log_probs, r.rewards, r.values, r.next_values)))
print("done flags vs collected returns:", n_flags, len(returns))  # the env raises a flag (terminated or truncated) on the step an episode ends
print("mean return:", sum(returns) / len(returns))

# Check 3: forced truncations, true next state
envs = make_vec_env("CartPole-v1", N, max_episode_steps=10)
collector = RolloutCollector(envs, actor, critic, T, seed=0)
r = collector.collect()
done = (r.terminated | r.truncated)[:-1]
print("truncations seen:", int(r.truncated.sum()))
with torch.no_grad():
    v_of_stored_next_obs = critic(r.obs[1:])      # [T-1, N]
close = torch.isclose(v_of_stored_next_obs, r.next_values[:-1], atol=1e-6)
print("not-done entries all match:", bool(close[~done].all()))
print("done entries that differ:", int((~close[done]).sum()), "of", int(done.sum()))