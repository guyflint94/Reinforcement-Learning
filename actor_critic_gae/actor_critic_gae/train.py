import csv
import gymnasium as gym
import random
import torch
import numpy as np
import os
import argparse
from actor_critic_gae.config import Config
from actor_critic_gae.networks import ActorNet
from torch.distributions.categorical import Categorical


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. Call before any env or network is created."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str) -> gym.Env:
    """Create one Gymnasium env. Does not reset it; the caller seeds the first reset."""
    env = gym.make(env_id)
    return env


def run_random_policy(cfg: Config) -> list[tuple[int, float, int]]:
    """
    Run cfg.num_episodes times with an untrained actor.
    Returns one (episode_index, return, length) tuple per episode.
    """
    set_seed(cfg.seed)
    num_episodes = cfg.num_episodes
    env = make_env(cfg.env_id)

    final_lst = []
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    actor_net = ActorNet(obs_dim=obs_dim, n_actions=act_dim, hidden_size=cfg.hidden_size)  # construct the actor neural net
    obs, info = env.reset(seed=cfg.seed)

    for itr in range(num_episodes):
        ep_rewards = []
        ep_length_count = 0

        while True:
            obs = torch.tensor(obs, dtype=torch.float)

            with torch.no_grad():
                logits = actor_net(obs)
                action_probs = Categorical(logits=logits)
                act = action_probs.sample().item()

            next_state, reward, terminated, truncated, info = env.step(act)
            obs = next_state

            ep_rewards.append(reward)
            ep_length_count += 1

            if truncated or terminated:
                final_lst.append((itr, sum(ep_rewards), ep_length_count))
                break
        obs, info = env.reset()

    return  final_lst


def write_episode_log(rows: list[tuple[int, float, int]], path: str) -> None:
    """Write a header row and one CSV row per episode."""

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length"])
        for r in rows:
            writer.writerow(r)  # data


def main() -> None:
    """Parse --seed, --env-id, --episodes and --log-path, build a Config, run, log."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=Config.seed)
    parser.add_argument('--env-id', type=str, default=Config.env_id)
    parser.add_argument('--episodes', type=int, default=Config.num_episodes)
    parser.add_argument('--log-path', type=str, default=Config.log_path)
    args = parser.parse_args()

    cfg = Config(env_id=args.env_id, seed=args.seed,
                 num_episodes=args.episodes, log_path=args.log_path)

    rows = run_random_policy(cfg)
    write_episode_log(rows, cfg.log_path)

    mean_return = sum(r[1] for r in rows) / len(rows)
    print(f"Episodes: {len(rows)}, mean return: {mean_return:.1f}")


if __name__ == "__main__":
    main()
