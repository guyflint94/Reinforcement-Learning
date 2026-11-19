# Reinforcement Learning

Exercises from learning reinforcement learning from the ground up, as preparation for multi-agent PPO (MAPPO) in a thesis on autonomous taxi fleets.

## Contents

| Folder | What it covers |
|---|---|
| [dynamic_programming/](dynamic_programming/) | Tabular methods on a GridWorld: policy evaluation and greedy policy improvement |
| [policy_gradient/](policy_gradient/) | REINFORCE on CartPole, with reward-to-go and a baseline |
| [a2c_gae_ppo/](a2c_gae_ppo/) | A2C with GAE and a PPO update, both from scratch in PyTorch, compared on CartPole over 5 seeds (A2C does not hold its peak, PPO reaches 475 on 5/5 seeds but ends there on 2/5) |

Each folder has its own README with the requirements and how to run its code.
