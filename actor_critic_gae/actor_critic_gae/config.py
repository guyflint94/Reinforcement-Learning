# config.py
from dataclasses import dataclass


@dataclass
class Config:
    """All hyperparameters for one run. Single source of truth.
       Choose the defaults yourself.
    """
    # set up hyperparameters on the form: name: type = default
    env_id: str = "CartPole-v1"
    seed: int = 3
    num_episodes: int = 100
    hidden_size: int = 32
    log_path: str = 'logs/logs.csv'
    num_envs: int = 8
    rollout_steps: int = 32
    learning_rate: float = 0.0007
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalise_advantages: bool = True
    total_steps: int = 500000
    algo: str = "a2c"
    clip_eps: float = 0.2
    num_epochs: int = 4
    minibatch_size: int = 4
