import torch


def discounted_returns(rewards: torch.Tensor, next_values: torch.Tensor,
                       terminated: torch.Tensor, truncated: torch.Tensor,
                       gamma: float) -> torch.Tensor:
    """
    Compute bootstrapped discounted returns (G_t) for all timesteps
    and parallel environments in a rollout.

    Iterates backwards through time, computing:
        G_t = r_t + gamma * following_return

    The following return depends on the episode state:
        - Terminated: 0 (the episode genuinely ended).
        - Truncated: next_values[t] (bootstrap from the critic).
        - Last rollout step: next_values[t] (bootstrap from the critic).
        - Otherwise: G_t[t+1] (continue the return calculation).

    Args:
        rewards:     Rewards at each timestep, shape [T, N].
        next_values: Critic estimates of the next states, shape [T, N].
        terminated:  True where an episode genuinely ended, shape [T, N].
        truncated:   True where an episode was cut short, shape [T, N].
        gamma:       Discount factor.

    Returns:
        Discounted returns G_t, shape [T, N], without gradients.
    """
    G_t = torch.zeros_like(rewards)
    num_of_steps = len(rewards)
    num_of_envs = rewards.shape[1]

    for t in reversed(range(num_of_steps)):
        for i in range(num_of_envs):
            if terminated[t][i]:
                G_t[t][i] = rewards[t][i]
            elif truncated[t][i]:  # reached time-limit
                G_t[t][i] = rewards[t][i] + gamma * next_values[t][i]
            else:
                if t == num_of_steps-1:  # end of rollout
                    G_t[t][i] = rewards[t][i] + gamma * next_values[t][i]
                else:
                    G_t[t][i] = rewards[t][i] + gamma * G_t[t+1][i]
    return G_t


def td_advantages(rewards: torch.Tensor, values: torch.Tensor,
                  next_values: torch.Tensor, terminated: torch.Tensor,
                  gamma: float) -> torch.Tensor:
    """One-step TD error for every step: the TD(0) advantage.

    All float32 [T, N], terminated bool [T, N]. No loop needed (fully vectorised).
    The next-state value counts unless the step ended in a termination.
    Returns float32 [T, N]. No gradients.
    """
    # td_error = rewards + gamma*(1-d) * next_values - values
    # termination indicator d is 1 if terminated, 0 if not
    terminated_float = terminated.float()

    td_error = rewards + gamma*(1-terminated_float)*next_values - values
    return td_error


def gae_advantages(rewards: torch.Tensor, values: torch.Tensor,
                   next_values: torch.Tensor, terminated: torch.Tensor,
                   truncated: torch.Tensor, gamma: float,
                   lam: float) -> torch.Tensor:
    """Generalised Advantage Estimation, A_t, for every step of a rollout.

    Shapes and dtypes as above. Works backwards over time.
    Each step starts from its own TD error (as in td_advantages) and adds
    gamma * lam times the advantage of step t+1, except that nothing is added
    after a termination or a truncation, and at the last step of the rollout.
    Returns float32 [T, N]. No gradients.
    """
    A_t = torch.zeros_like(rewards)  # create the shape of advantage [T, N]. later store values in it
    delta = torch.zeros_like(rewards)
    num_of_steps = len(rewards)

    for t in reversed(range(num_of_steps)):
        # Bootstrap unless the transition truly terminated
        bootstrap_mask = 1.0 - terminated[t].float()
        chain_mask = 1.0 - (truncated[t] | terminated[t]).float()

        delta[t] = rewards[t] + gamma * bootstrap_mask * next_values[t] - values[t]  # one-step TD error
        if t == num_of_steps-1:
            A_t[t] = delta[t]
        else:
            A_t[t] = delta[t] + (gamma*lam) * A_t[t + 1]*chain_mask

    return A_t