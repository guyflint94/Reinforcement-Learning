# Actor-critic with GAE (in progress)

Synchronous advantage actor-critic (A2C) with Generalised Advantage Estimation, written from scratch in PyTorch. It is the step between REINFORCE and PPO.

**Status: milestone 1 of 6 (skeleton). The agent does not learn yet.**

## Files

- `actor_critic_gae/config.py`: dataclass of hyperparameters.
- `actor_critic_gae/networks.py`: separate actor (logits) and critic (value) MLPs.
- `actor_critic_gae/train.py`: seeding, and a random-policy run on CartPole-v1 that writes one CSV row per episode (episode, return, length).
- `tests/test_networks.py`: output shapes, no shared parameters between actor and critic, and a check that the critic output is not squashed into 0 to 1.

## Requirements

Pinned in `requirements.txt`: `gymnasium`, `torch`, `numpy`, `pandas`, `matplotlib`, `pytest`.

## Running

Run both from this folder, because the code is imported as a package:

    python -m actor_critic_gae.train --seed 0 --episodes 20
    python -m pytest -v

Options: `--seed`, `--env-id`, `--episodes`, `--log-path`. The log goes to `logs/logs.csv`.

## Results

None yet. The only number so far is a sanity baseline for an untrained policy: 20 episodes on CartPole-v1 with seed 0 gave a mean return of 22.9. That is a random policy, not a learning result.

## Not done yet

Parallel rollouts with `gymnasium.vector`, the advantage estimators (TD(0) and GAE) with unit tests, the A2C update, a lambda sweep over several seeds, a truncation-versus-termination ablation, and Acrobot.
