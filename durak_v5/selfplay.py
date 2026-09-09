"""Generate policy targets from information-set MCTS self-play."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from durak_v3.agents import BaseController
from durak_v3.observation import (
    encode_observation, encode_privileged, public_view_from_game,
)
from durak_v3.search import InformationSetMCTS, root_from_game
from state import encode_options_tensor


@dataclass
class SearchTargetStep:
    observation: torch.Tensor
    options: torch.Tensor
    privileged: torch.Tensor
    policy_target: torch.Tensor


class SearchSelfPlayController(BaseController):
    """Choose from MCTS visit counts and retain them as training labels."""

    def __init__(
        self,
        model,
        simulations: int = 32,
        time_limit_ms: int = 250,
        exploration_moves: int = 12,
        root_noise_fraction: float = 0.25,
        root_dirichlet_alpha: float = 0.3,
        record: bool = True,
    ):
        self.model = model
        self.search = InformationSetMCTS(
            model, simulations=simulations, time_limit_ms=time_limit_ms,
        )
        self.exploration_moves = exploration_moves
        self.root_noise_fraction = root_noise_fraction
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.record = record

    def reset(self, seat: int) -> None:
        super().reset(seat)
        self.hidden = self.model.initial_hidden()
        self.belief_hidden = self.model.initial_belief_hidden()
        self.steps: list[SearchTargetStep] = []
        self.decisions = 0

    def choose(self, game, seat: int, options) -> int:
        observation = encode_observation(public_view_from_game(game, seat, self.memory))
        encoded_options = encode_options_tensor(options)
        privileged = encode_privileged(game, seat)
        hidden_before = self.hidden
        with torch.no_grad():
            _, self.belief_hidden = self.model.belief_step(
                observation, self.belief_hidden,
            )
            _, _, self.hidden = self.model.forward_step(
                observation, encoded_options, self.hidden, privileged,
                self.belief_hidden,
            )
            opponent_weights = self.model.opponent_weights(self.belief_hidden)
            result = self.search.search(
                root_from_game(game, seat, self.memory), options, hidden_before,
                opponent_weights,
                root_noise_fraction=self.root_noise_fraction,
                root_dirichlet_alpha=self.root_dirichlet_alpha,
            )

        visits = torch.tensor(result.visits, dtype=torch.float32)
        if float(visits.sum()) == 0.0:
            policy_target = torch.full_like(visits, 1.0 / len(visits))
        else:
            policy_target = visits / visits.sum()
        if self.record:
            self.steps.append(SearchTargetStep(
                observation, encoded_options, privileged, policy_target,
            ))

        if self.decisions < self.exploration_moves:
            action = int(torch.multinomial(policy_target, 1).item())
        else:
            action = int(visits.argmax().item())
        self.decisions += 1
        return action
