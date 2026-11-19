from actor_critic_gae.networks import ActorNet, CriticNet
from actor_critic_gae.advantages import gae_advantages
from actor_critic_gae.rollout import Rollout
from torch.distributions.categorical import Categorical
import torch


def ppo_update(actor: ActorNet, critic: CriticNet, optimizer: torch.optim.Optimizer,
               rollout: Rollout, gamma: float, lam: float,
               clip_eps: float, num_epochs: int, minibatch_size: int,
               value_coef: float, entropy_coef: float, max_grad_norm: float,
               normalise_advantages: bool) -> dict[str, float]:

    # calculate the GAE and the critics target G_t
    GAE_advantages = gae_advantages(rollout.rewards, rollout.values, rollout.next_values, rollout.terminated,
                                    rollout.truncated, gamma, lam)
    critics_target = GAE_advantages + rollout.values

    # flatten
    obs = rollout.obs.reshape(-1, rollout.obs.shape[2])
    actions = rollout.actions.reshape(-1)
    GAE_advantages = GAE_advantages.reshape(-1)
    critics_target = critics_target.reshape(-1)
    old_log_probs = rollout.log_probs.reshape(-1)

    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []
    clip_fractions = []

    for epoch in range(num_epochs):
        shuffle_tensor = torch.randperm(len(obs))
        for start in range(0, len(obs), minibatch_size):
            idx = shuffle_tensor[start : start + minibatch_size]

            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_log_probs = old_log_probs[idx]
            mb_GAE_advantages = GAE_advantages[idx]
            mb_critics_target = critics_target[idx]

            if normalise_advantages:
                mb_GAE_advantages = (mb_GAE_advantages - mb_GAE_advantages.mean()) / (mb_GAE_advantages.std() + 1e-8)

            # Actor
            logits = actor(mb_obs)
            probs = Categorical(logits=logits)
            new_log_probs = probs.log_prob(mb_actions)
            entropy = probs.entropy()

            # Critic
            critic_values = critic(mb_obs)

            # calculate the ratio probability r_t
            prob_ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # clipped policy loss
            plain = prob_ratio * mb_GAE_advantages
            clipped = (torch.clamp(prob_ratio, 1 - clip_eps, 1 + clip_eps) * mb_GAE_advantages)
            policy_loss = -torch.min(plain, clipped).mean()

            # value loss
            value_loss = ((critic_values - mb_critics_target) ** 2).mean()

            # mean entropy
            mean_entropy = entropy.mean()

            # Total loss
            total_loss = (policy_loss + value_coef * value_loss - entropy_coef * mean_entropy)

            # reset gradients
            optimizer.zero_grad()

            # calculate gradients on backpropagation
            total_loss.backward()

            # These two lines implement gradient clipping. Their purpose is to prevent excessively
            # large gradients from causing unstable neural network updates during training.
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

            # # take one update step
            optimizer.step()

            # No-grad diagnostics
            with torch.no_grad():
                approx_kl = (mb_old_log_probs - new_log_probs).mean()

                clip_fraction = ((prob_ratio - 1).abs() > clip_eps).float().mean()

            # Store minibatch results
            policy_losses.append(policy_loss.detach().item())
            value_losses.append(value_loss.detach().item())
            entropies.append(mean_entropy.detach().item())
            approx_kls.append(approx_kl.detach().item())
            clip_fractions.append(clip_fraction.detach().item())

    # Return average metrics
    return {
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
        "entropy": sum(entropies) / len(entropies),
        "approx_kl": sum(approx_kls) / len(approx_kls),
        "clip_fraction": sum(clip_fractions) / len(clip_fractions),
        }
