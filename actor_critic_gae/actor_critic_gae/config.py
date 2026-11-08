# config.py
from dataclasses import dataclass


@dataclass
class Config:
    """All hyperparameters for one run. Single source of truth.

    Add a field only when a milestone needs it.
    M1 needs: env_id, seed, num_episodes, hidden_size, log_path.
    Choose the defaults yourself.
    """

    # set up hyperparameters on the form: name: type = default
    env_id: str = "CartPole-v1"
    seed: int = 3
    num_episodes: int = 100
    hidden_size: int = 32
    log_path: str = 'logs/logs.csv'
