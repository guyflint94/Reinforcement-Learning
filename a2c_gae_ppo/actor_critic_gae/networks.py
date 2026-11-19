import torch
from torch import nn


class ActorNet(nn.Module):
    """Policy network: observation -> unnormalised action logits.

    Returns logits, not probabilities or log-probabilities.
    The caller builds the distribution.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int) -> None:
        super().__init__()

        layers = [
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, n_actions),
            nn.Identity()
            ]
        self.a_net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: float tensor [batch, obs_dim].
        Returns: float tensor [batch, n_actions].
        """
        logits = self.a_net(obs)
        return logits


class CriticNet(nn.Module):
    """State-value network: observation -> V(s). Shares no parameters with ActorNet."""

    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        """Two hidden layers of hidden_size units each."""
        super().__init__()
        layers = [
           nn.Linear(obs_dim, hidden_size),
           nn.ReLU(),

           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),

          nn.Linear(hidden_size, 1)
          ]
        self.c_net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: float tensor [batch, obs_dim].
        Returns: float tensor of shape [batch], not [batch, 1].
        """
        critics_value = self.c_net(obs).squeeze(dim=-1)
        return critics_value