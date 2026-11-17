# rollout.py
from dataclasses import dataclass
import gymnasium as gym
import torch
from torch.distributions.categorical import Categorical
from actor_critic_gae.networks import ActorNet, CriticNet


def make_vec_env(env_id: str, num_envs: int,
                 max_episode_steps: int | None = None) -> gym.vector.VectorEnv:
    """Create num_envs copies of env_id that step in lockstep (sync vectorisation).

    The auto-reset mode must be set explicitly to SAME_STEP. The Gymnasium 1.4.0
    default is NEXT_STEP.
    max_episode_steps: None keeps the environment's own limit (500 for CartPole-v1).
      A small number such as 10 forces truncations, which acceptance check 3 needs.
    Returns the vector env. Does not reset it.
    """
    envs = gym.make_vec(env_id, num_envs, vectorization_mode="sync", max_episode_steps= max_episode_steps, vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    return envs


# Rollout is a container for all the information collected during a rollout.
@dataclass
class Rollout:
    """T steps from N environments. Time-major: [t, i] is step t of environment i.
    Every tensor is detached (collected without gradients).

    obs:         float32 [T, N, obs_dim]  state s_t the action was chosen in
    actions:     int64   [T, N]           a_t
    log_probs:   float32 [T, N]           log pi_old(a_t | s_t), from the policy that acted
    rewards:     float32 [T, N]           r_t
    values:      float32 [T, N]           V(s_t)
    next_values: float32 [T, N]           V(s_{t+1}) of the TRUE next state. If the
                                          episode ended at step t, that is the final
                                          observation, not the reset observation.
    terminated:  bool    [T, N]           real terminal state reached at step t
    truncated:   bool    [T, N]           time limit hit at step t
    episode_returns: list[float]          returns of episodes that finished during
                                          this rollout
    """
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_returns: list[float]


class RolloutCollector:
    def __init__(self, envs: gym.vector.VectorEnv, actor: ActorNet,
                 critic: CriticNet, num_steps: int, seed: int) -> None:
        """Keep the envs, the two networks and T = num_steps.

        Resets the envs once with seed, and keeps the current observations
        [N, obs_dim] and a running return per environment as state, because
        episodes continue across consecutive rollouts.
        """
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.num_steps = num_steps

        numpy_obs, numpy_info = envs.reset(seed=seed)
        self.obs = torch.tensor(numpy_obs, dtype=torch.float32) # convert observations into PyTorch tensor shape = [N, obs_dim]
        self.running_return = torch.zeros(envs.num_envs, dtype=torch.float32) # keep track of the cumulative reward for each environment's currently running episode


    def collect(self) -> Rollout:
        """The function's job is to:Run the actor, Sample actions, Run the critic, Step the environments,
        Calculate the next-state values, Save the data, Track completed episodes.
        This method is the actual data-collection engine.

        For each step t, without gradients:
          1. Actor on the current obs gives a distribution. Sample one action per
             environment and take its log-prob. Critic on the same obs gives V(s_t).
          2. Step all envs with the N actions (a numpy int array).
          3. Build the true next observation per env: the returned one, except
             where an episode just ended, where it is info["final_obs"][i].
          4. Critic on the true next observations gives next_values[t].
          5. Store all of it at index t.
          6. Add rewards to the running returns. When an env ends an episode,
             append its return to episode_returns and zero its counter.
          7. The returned observations (already reset where an episode ended)
             become the state for step t+1, and stay in the collector for the
             next call.
        Returns a Rollout with the shapes above. The next call to collect()
        continues where this one stopped.
        """
        obs = torch.empty(self.num_steps, self.envs.num_envs, self.obs.shape[1], dtype=torch.float32)
        actions = torch.empty(self.num_steps, self.envs.num_envs, dtype=torch.int64)
        log_probs = torch.empty(self.num_steps, self.envs.num_envs, dtype=torch.float32)
        rewards = torch.empty(self.num_steps, self.envs.num_envs, dtype=torch.float32)
        values = torch.empty(self.num_steps, self.envs.num_envs, dtype=torch.float32)
        next_values = torch.empty(self.num_steps, self.envs.num_envs, dtype=torch.float32)
        terminated = torch.empty(self.num_steps, self.envs.num_envs, dtype=torch.bool)
        truncated = torch.empty(self.num_steps, self.envs.num_envs, dtype=torch.bool)
        episode_returns = []  # Whenever an environment finishes an episode, the collector will append its total return to this list

        with torch.no_grad():
            for t in range(self.num_steps):

                # actor part
                logits = self.actor(self.obs)  # passes the current observations into the actor network
                probs = Categorical(logits=logits)  # turn logits to probabilities with softmax. The distribution now represents the actor's policy pi(a|s)
                act = probs.sample()  # sample an action. act will be a tensor size [N] of actions
                log_probs_t = probs.log_prob(act)  # Calculate the log probability of the selected actions. shape [N]

                # critic part
                st_value = self.critic(self.obs)  # st_value is a [N] tensor with state values

                # step the environments
                returned_obs, reward, terminated_np, truncated_np, info = self.envs.step(act.numpy())  # the env wants a numpy int array
                returned_obs = torch.tensor(returned_obs, dtype=torch.float32)  # convert observation from numpy to tensor
                next_obs_true = returned_obs.clone()

                """final_obs Contains final observations.   _final_obs Identifies which environments have final observations"""
                if "final_obs" in info:  # Check whether any episode finished. If yes, final_obs is the key in the dict info
                    mask = info["_final_obs"]

                    for i in range(self.envs.num_envs):
                        if mask[i]:
                            next_obs_true[i] = torch.from_numpy(info["final_obs"][i])

                obs[t] = self.obs
                actions[t] = act
                log_probs[t] = log_probs_t
                rewards[t] = torch.tensor(reward, dtype=torch.float32)
                values[t] = st_value
                next_values[t] = self.critic(next_obs_true) # critic on next_obs_true gives the value state estimate. V(s+1)
                truncated[t] = torch.tensor(truncated_np)
                terminated[t] = torch.tensor(terminated_np)
                self.running_return = self.running_return + rewards[t]

                self.obs = returned_obs # state in t+1 becomes the current state (one step forward)
                for i in range(self.envs.num_envs):
                    if terminated[t][i] or truncated[t][i]:
                        episode_returns.append((self.running_return[i]).item())
                        self.running_return[i] = 0
        return Rollout(obs, actions, log_probs, rewards, values, next_values, terminated, truncated, episode_returns)



