import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# 30/10_26
# nn.Identity returns its input unchanged. It doesn't apply any activation function.
# For this REINFORCE policy network, nn.Identity is appropriate because we want raw output scores.
def build_neural_network(size, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(size) - 1):
        act_layer = activation if j < len(size)-2 else output_activation
        layers += [nn.Linear(size[j], size[j+1]), act_layer()]
    return nn.Sequential(*layers)


# # G_t = r_t + gamma * G_(t+1)
def compute_weights(ep_rewards, gamma, reward_to_go):
    weights = [0.0] * len(ep_rewards)
    running = 0.0
    if reward_to_go:
        for t in reversed(range(len(ep_rewards))):
            running = ep_rewards[t] + gamma*running
            weights[t]=running
    else:
        weights = [sum(ep_rewards)] * len(ep_rewards)
    return weights


def train(env_name='CartPole-v1', hidden_layers=[32], lr = 1e-2, render=False, epochs = 100,
          batch_size = 1000, seed=0, gamma=1.0, reward_to_go=True, baseline=False):
    # create the environment
    env = gym.make(env_name, render_mode="human" if render else None)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seeds the environment once
    env.reset(seed=seed)

    # These lines are checks that make sure the environment is compatible with the algorithm's implementation.
    # They aren't essential to the learning process itself, but they're useful for catching mistakes early.
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    obs_dim = env.observation_space.shape[0] # we have 4: Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
    num_of_acts = env.action_space.n # we have 2 possible actions: move right, move left

    logits_net = build_neural_network([obs_dim] + hidden_layers + [num_of_acts])  # our neural network

    # compute action distribution (probabilities object)
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)  # PyTorch converts the logits into probabilities (using softmax):

    # choose an action. sample an action from the policy
    def choose_action(obs):
        return get_policy(obs).sample().item()  # .item() converts a single-value PyTorch tensor into a regular Python number

    # calculate the loss
    def loss_function(obs, act, returns):
        log_prob = get_policy(obs).log_prob(act)
        return -(log_prob * returns).mean()  # We use .mean() to average the loss contributions across all collected time steps

    # logits_net is a PyTorch neural network, and PyTorch automatically tracks its trainable parameters
    # and thus Adam optimizer knows what the parameters are
    optimizer = Adam(logits_net.parameters(), lr=lr)


    mean_returns = []
    """
    Episode: One complete attempt to balance the pole, from reset until the pole falls or the episode otherwise ends.
    Epoch: One training iteration in this particular code. It collects enough experience from one or more episodes and then updates the neural network once.
    """
    # training the policy
    def train_one_epoch():

        # create empty lists to store values along the training
        batch_obs = []
        batch_act = []
        batch_returns = []  # stores one return per episode, mainly for tracking performance.
        batch_weights = []  # holds weights (one per step)
        batch_len = []

        # These lines initialize the variables needed before the agent starts collecting experience.
        obs, info = env.reset()
        done = False
        ep_reward = []  # list for rewards accrued throughout an episode

        # collect experience by acting in the environment with current policy
        while True:

            # save observation
            batch_obs.append(obs.copy())  # stores an independent copy

            # act in the environment
            act = choose_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, terminated, truncated, info = env.step(act)
            """obs - the new state we transition into
               reward - the reward we got for the action we took
               terminated - the episode ended naturally, e.g. the pole fell
               truncated - the episode ended because a time limit was reached
               info - additional information we'll ignore"""

            done = terminated or truncated

            # save action and reward
            batch_act.append(act)
            ep_reward.append(reward)

            if done:
                # if episode is over, record info about episode
                ep_return = sum(ep_reward)
                ep_len = len(ep_reward)

                batch_returns.append(ep_return)
                batch_len.append(ep_len)

                # This line adds one weight per step for this episode
                batch_weights += compute_weights(ep_reward, gamma, reward_to_go)

                # reset variables at the end of each episode
                obs, info = env.reset()
                done = False
                ep_reward = []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        if baseline:
            avg_weights = np.mean(batch_weights)
            for i in range(len(batch_weights)):
                batch_weights[i] = batch_weights[i] - avg_weights


        # next, we update our neural network (policy)
        #---------------------------------------------------------------------------------------------------------#
        optimizer.zero_grad()  # Clears the gradients stored from the previous training update

        # loss_function() uses these tensors to calculate how the policy's action probabilities should be encouraged or discouraged.
        # The result, batch_loss, is a single number representing the loss across the batch
        batch_loss = loss_function(torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
                                   torch.as_tensor(batch_act, dtype=torch.int64),
                                   torch.as_tensor(batch_weights, dtype=torch.float32))

        batch_loss.backward()  # Calculates the gradients of the loss with respect to the neural network's trainable parameters.
        optimizer.step()  # Adam uses those gradients to update the neural network's parameters.

        return batch_loss.item(), batch_returns, batch_len

    # training loop
    for i in range(epochs):

        (batch_loss, batch_rets, batch_lens) = train_one_epoch()

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
            (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

        mean_returns.append(np.mean(batch_rets))

    return mean_returns


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--whole_episode_return', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr, seed=args.seed, gamma=args.gamma,
          reward_to_go=not args.whole_episode_return, baseline=args.baseline)
