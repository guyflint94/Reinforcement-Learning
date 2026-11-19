import csv
import gymnasium as gym
import random
import torch
import numpy as np
import os
import argparse
from actor_critic_gae.config import Config
from actor_critic_gae.networks import ActorNet, CriticNet
from torch.distributions.categorical import Categorical
from actor_critic_gae.ppo_update import ppo_update
from actor_critic_gae.rollout import make_vec_env, RolloutCollector
from actor_critic_gae.update import a2c_update


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


def save_training_log(rows: list[tuple[int, float]], path: str) -> None:
    """Write (steps, return) for every finished training episode."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["steps", "return"])
        writer.writerows(rows)


def train(cfg):
    set_seed(seed=cfg.seed)
    envs = make_vec_env(cfg.env_id, cfg.num_envs)

    obs_dim = envs.single_observation_space.shape[0]
    num_of_actions = envs.single_action_space.n

    actor_net = ActorNet(obs_dim, num_of_actions, cfg.hidden_size)
    critic_net = CriticNet(obs_dim, cfg.hidden_size)

    collector = RolloutCollector(envs, actor_net, critic_net, cfg.rollout_steps, cfg.seed)
    optimizer = torch.optim.Adam(list(actor_net.parameters()) + list(critic_net.parameters()), lr=cfg.learning_rate)

    counter = 0
    iteration = 0
    tracking_lst = []
    while counter < cfg.total_steps:

        collect_experience = collector.collect()

        if cfg.algo == "ppo":
            stats = ppo_update(actor_net, critic_net, optimizer, collect_experience,
                               cfg.gamma, cfg.lam, cfg.clip_eps, cfg.num_epochs, cfg.minibatch_size,
                               cfg.value_coef, cfg.entropy_coef, cfg.max_grad_norm, cfg.normalise_advantages)
        elif cfg.algo == "a2c":
            stats = a2c_update(actor_net, critic_net, optimizer, collect_experience, cfg.gamma,
                               cfg.lam, cfg.value_coef, cfg.entropy_coef, cfg.max_grad_norm,
                               cfg.normalise_advantages)
        else:
            raise ValueError(f"Unknown algo: {cfg.algo!r} (use 'a2c' or 'ppo')")

        counter += cfg.rollout_steps * cfg.num_envs
        iteration += 1

        for s in collect_experience.episode_returns:
            tracking_lst.append((counter, s))


        if iteration % 50 == 0:
            recent = [r for _, r in tracking_lst[-20:]]
            mean_recent = sum(recent) / len(recent) if recent else float("nan")
            line = (f"iter {iteration:5d} | steps {counter:7d} | "
                    f"policy {stats['policy_loss']:8.4f} | value {stats['value_loss']:9.3f} | "
                    f"entropy {stats['entropy']:.3f} | return(last 20) {mean_recent:7.1f}")
            if "approx_kl" in stats:
                line += f" | approx_kl {stats['approx_kl']:.4f} | clip_fraction {stats['clip_fraction']:.3f}"
            print(line)

    return tracking_lst


def main() -> None:
    """Parse --seed, --env-id, --episodes and --log-path, build a Config, run, log."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=Config.seed)
    parser.add_argument('--env-id', type=str, default=Config.env_id)
    parser.add_argument('--episodes', type=int, default=Config.num_episodes)
    parser.add_argument('--log-path', type=str, default=Config.log_path)
    parser.add_argument('--num-envs', type=int, default=Config.num_envs)
    parser.add_argument('--total-steps', type=int, default=Config.total_steps)
    parser.add_argument('--algo', type=str, default=Config.algo)
    parser.add_argument('--lr', type=float, default=Config.learning_rate)
    parser.add_argument('--out-dir', type=str, default='results/runs')
    args = parser.parse_args()

    cfg = Config(env_id=args.env_id, seed=args.seed, num_episodes=args.episodes, log_path=args.log_path, num_envs=args.num_envs,
                 total_steps=args.total_steps, algo=args.algo, learning_rate=args.lr)


    # Train A2C
    training_rows = train(cfg)

    save_training_log(training_rows, f"{args.out_dir}/{args.algo}_seed{args.seed}.csv")

    # Each training row is (total_steps, episode_return)
    training_returns = [episode_return for _, episode_return in training_rows]

    num_episodes = len(training_returns)
    print(f"\nTraining episodes recorded: {num_episodes}")

    if num_episodes > 0:
        first_20 = training_returns[:20]
        last_20 = training_returns[-20:]

        print(f"Mean return of first {len(first_20)} episodes: "f"{np.mean(first_20):.2f}")

        print(f"Mean return of last {len(last_20)} episodes: "f"{np.mean(last_20):.2f}")
    else:
        print("No training episodes finished.")


if __name__ == "__main__":
    main()


