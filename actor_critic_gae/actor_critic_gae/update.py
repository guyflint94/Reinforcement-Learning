import torch
from actor_critic_gae.rollout import Rollout
from actor_critic_gae.networks import ActorNet, CriticNet
from actor_critic_gae.advantages import td_advantages, gae_advantages, discounted_returns
from torch.distributions.categorical import Categorical


def a2c_update(actor: ActorNet, critic: CriticNet, optimizer: torch.optim.Optimizer,
               rollout: Rollout, gamma: float, lam: float, value_coef: float, entropy_coef: float,
               max_grad_norm: float, normalise_advantages: bool) -> dict[str, float]:

    # calculate the GAE and the critics target G_t
    GAE_advantages = gae_advantages(rollout.rewards, rollout.values, rollout.next_values, rollout.terminated,
                                    rollout.truncated, gamma, lam)
    critics_target = GAE_advantages + rollout.values

    # flatten
    obs = rollout.obs.reshape(-1, rollout.obs.shape[2])
    actions = rollout.actions.reshape(-1)
    GAE_advantages = GAE_advantages.reshape(-1)
    critics_target = critics_target.reshape(-1)

    # advantages normalisation (z-score normalisation)
    if normalise_advantages:
        GAE_advantages = (GAE_advantages - GAE_advantages.mean()) / (GAE_advantages.std() + 1e-8)

    logits = actor(obs)
    probs = Categorical(logits=logits)
    log_probabilities = probs.log_prob(actions)

    entropy = probs.entropy()  # it tells us how much randomness the actor has in its policy for a given observation

    critic_value_est = critic(obs)

    # Sanity check. we did not change the policy yet so the two log probs need to be approx. the same
    old_log_probs = rollout.log_probs.reshape(-1)
    abs_diff = (log_probabilities.detach() - old_log_probs).abs().max().item()
    # print(f"Maximum log-probability difference: {abs_diff:.10f}")

    # policy loss
    policy_loss = -(log_probabilities * GAE_advantages).mean()

    # value loss
    value_loss = ((critic_value_est - critics_target)**2).mean()

    # mean entropy
    mean_entropy = entropy.mean()

    # total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * mean_entropy

    # reset gradients
    optimizer.zero_grad()

    # calculate gradients on backpropagation
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(list(actor.parameters()), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(list(critic.parameters()), max_grad_norm)


    # take one update step
    optimizer.step()

    return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": mean_entropy.item()}

