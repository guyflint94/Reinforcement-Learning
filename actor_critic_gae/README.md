# Actor-critic with GAE (in progress)

Synchronous advantage actor-critic (A2C) with Generalised Advantage Estimation, written from scratch in PyTorch on Gymnasium's vector environments. It is the step between REINFORCE and PPO.

**Status:** the full A2C pipeline runs and learns on CartPole-v1, but it does not yet pass its acceptance check (see Results). PPO is the next step.

## What is implemented

- **Vectorised rollouts** (`rollout.py`): N environments stepped in lockstep with `AutoresetMode.SAME_STEP`. Each step stores the state, action, old log-prob, reward, V(s_t) and V(s_{t+1}), all as time-major `[T, N]` tensors collected without gradients. Where an episode ends, V(s_{t+1}) is computed from the true final observation (`info["final_obs"]`), not from the reset observation.
- **Termination versus truncation**: only `terminated` switches off the bootstrap. A time limit (`truncated`) or the end of the rollout still bootstraps from the critic. The GAE chain is cut at `terminated | truncated`, so advantages never leak across episodes.
- **Advantage estimators** (`advantages.py`): bootstrapped discounted returns, TD(0) advantages and GAE(λ).
- **A2C update** (`update.py`): policy-gradient loss with GAE advantages (optionally normalised per batch), squared-error value loss against the λ-return, an entropy bonus, gradient-norm clipping applied separately to actor and critic, and Adam over both networks.
- **Training loop** (`train.py`): collect, update, log the return of every finished episode against environment steps, and print losses, entropy and the mean of the last 20 returns every 50 updates.
- **Networks** (`networks.py`): separate actor (logits) and critic (value) MLPs with two hidden layers, no shared parameters.

## Checks

- `tests/test_networks.py`: output shapes, no shared parameters, critic output not squashed.
- `tests/test_advantages.py`, on a small rollout that contains a termination, a truncation and a step with both flags:
  - GAE with λ = 0 equals the TD(0) advantage;
  - GAE with λ = 1 equals the bootstrapped return minus V(s_t).
  - Two further tests (a hand-computed case and a direct truncation check) are written as stubs and marked as skipped.
- `scripts/check_rollout.py`: shapes and dtypes, no gradients in stored tensors, one recorded return per episode end, and, with a forced time limit of 10 steps, V(s_{t+1}) matches V of the next stored state on every non-final step and differs on all 96 of 96 final steps (so the true final observation is used).
- `scripts/check_update.py`: one update on a fresh rollout. Before the update, the recomputed log-probs match the stored ones to within 6e-8, which confirms that observations, actions and ordering line up after flattening.

## Running

From this folder:

    python -m actor_critic_gae.train --total-steps 500000 --seed 3
    python -m pytest -v

Other options: `--env-id`, `--num-envs`, `--log-path`, `--episodes` (for the random-policy baseline). Other hyperparameters are in `config.py`.

## Results so far (CartPole-v1, A2C, one seed)

All runs: seed 3, 8 environments, 500k environment steps, γ = 0.99, λ = 0.95, value coefficient 0.5, two hidden layers of 32 units. "Last 20" is the mean return of the last 20 finished training episodes, with actions sampled from the policy. A random policy scores 23.5.

| Run | Rollout length | Learning rate | Grad clipping | Adv. normalisation | Entropy coef. | Best last-20 (at steps) | Last 20 at end |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 3e-4 | joint | on | 0.01 | 470 (461k) | 88 |
| 2 | 32 | 3e-4 | joint | on | 0.01 | 275 (333k) | 177 |
| 3 | 32 | 7e-4 | joint | on | 0.01 | 399 (230k) | 171 |
| 4 | 32 | 7e-4 | joint | on | 0.001 | 381 (269k) | stopped at 410k |
| 5 | 32 | 7e-4 | separate | on | 0.01 | 403 (320k) | **329** |
| 6 | 32 | 7e-4 | separate | off | 0.01 | 453 (384k) | 279 |

The current defaults in `config.py` are run 5. Logs are in `results/a2c_cartpole/` (runs 1 and 6 are excerpts, with some printed lines not kept).

**What this shows, and what it does not:**

- The pipeline learns: every run climbs from 23 to between 275 and 470.
- No run holds its peak. Each one reaches 400 to 470, then drops sharply (for example from 380 to 67 within about 13k steps in run 4), and the drops come with large spikes in the value loss.
- The acceptance check set for this milestone is not met: a last-20 mean of at least 475 within 500k steps, with λ = 0.95, on at least 4 of 5 seeds.
- Changing the batch size, learning rate, entropy weight, clipping scheme and advantage normalisation changed when the swings happened, but none of them removed them. The policy entropy stayed at about 0.55 to 0.60 in every run (0.69 is uniform over two actions).
- Every row is a single seed. Single runs of the same setting vary a lot, so differences between rows are not reliable evidence. Separate clipping (run 5) gave the best final value, but that is one run.

Two things are not tried yet: a learning rate that decays over training, and a greedy evaluation (always taking the most likely action), which would separate what the policy has learned from the noise of sampling.

## Not done yet

- PPO (clipped ratio, several epochs, minibatches) on the same code, compared with A2C.
- Plots of the learning curves.
- A λ sweep over {0, 0.5, 0.9, 0.95, 1.0} with 5 seeds, the truncation-as-termination ablation, and Acrobot-v1.
