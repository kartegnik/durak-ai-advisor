"""Recurrent actor-critic used by PPO and information-set search."""

from __future__ import annotations

import torch
from torch import nn

from state import OPTION_DIM
from .observation import OBSERVATION_DIM, PRIVILEGED_DIM


class RecurrentActorCritic(nn.Module):
    def __init__(self, hidden_size: int = 256, option_size: int = 96):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_encoder = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        )
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.option_encoder = nn.Sequential(
            nn.Linear(OPTION_DIM, option_size), nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_size + option_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.privileged_encoder = nn.Sequential(
            nn.Linear(PRIVILEGED_DIM, option_size), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size + option_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Tanh(),
        )

    def initial_hidden(self, device=None) -> torch.Tensor:
        return torch.zeros(self.hidden_size, device=device)

    def policy_context(
        self, recurrent_hidden: torch.Tensor,
        belief_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return recurrent_hidden

    def forward_step(
        self,
        observation: torch.Tensor,
        options: torch.Tensor,
        hidden: torch.Tensor | None = None,
        privileged: torch.Tensor | None = None,
        belief_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.initial_hidden(observation.device)
        encoded = self.observation_encoder(observation)
        next_hidden = self.memory(encoded, hidden)
        option_features = self.option_encoder(options)
        policy_hidden = self.policy_context(next_hidden, belief_hidden)
        context = policy_hidden.unsqueeze(0).expand(len(options), -1)
        logits = self.policy(torch.cat((context, option_features), dim=-1)).squeeze(-1)

        if privileged is None:
            privileged_features = torch.zeros(
                self.privileged_encoder[0].out_features, device=observation.device,
            )
        else:
            privileged_features = self.privileged_encoder(privileged)
        value = self.value(torch.cat((next_hidden, privileged_features), dim=-1)).squeeze(-1)
        return logits, value, next_hidden
