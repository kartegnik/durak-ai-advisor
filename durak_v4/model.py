"""Recurrent actor-critic with an auxiliary hidden-card prediction head."""

from __future__ import annotations

import torch
from torch import nn

from state import NUM_CARDS
from durak_v3.model import RecurrentActorCritic
from durak_v3.observation import OBSERVATION_DIM


BELIEF_CLASSES = 3
VISIBLE = 0
OPPONENT = 1
DECK = 2


class BeliefActorCritic(RecurrentActorCritic):
    """Keep the v3 policy intact while teaching its memory to infer hidden cards."""

    def __init__(self, hidden_size: int = 256, option_size: int = 96):
        super().__init__(hidden_size, option_size)
        self.belief_observation_encoder = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.belief_memory = nn.GRUCell(hidden_size, hidden_size)
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, NUM_CARDS * BELIEF_CLASSES),
        )
        self.belief_policy_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        nn.init.zeros_(self.belief_policy_adapter[-1].weight)

    def belief_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = (*hidden.shape[:-1], NUM_CARDS, BELIEF_CLASSES)
        return self.belief_head(hidden).reshape(shape)

    def initial_belief_hidden(self, device=None) -> torch.Tensor:
        return torch.zeros(self.hidden_size, device=device)

    def belief_step(
        self, observation: torch.Tensor, hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.initial_belief_hidden(observation.device)
        encoded = self.belief_observation_encoder(observation)
        next_hidden = self.belief_memory(encoded, hidden)
        return self.belief_logits(next_hidden), next_hidden

    def opponent_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        """Odds used to sample a fixed-size opponent hand without replacement."""
        logits = self.belief_logits(hidden)
        return (logits[..., OPPONENT] - logits[..., DECK]).clamp(-12.0, 12.0).exp()

    def policy_context(
        self, recurrent_hidden: torch.Tensor,
        belief_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if belief_hidden is None:
            return recurrent_hidden
        return recurrent_hidden + self.belief_policy_adapter(belief_hidden)

    def load_v3_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            name for name in self.state_dict() if name.startswith("belief_")
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"incompatible v3 checkpoint: missing={missing}, unexpected={unexpected}"
            )
