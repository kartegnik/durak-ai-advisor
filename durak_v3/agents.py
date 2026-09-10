"""Controllers used for rollout collection, league play, and evaluation."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from durakgame import (
    AttackingMove, DefensiveMove, EndingMove, FinishingMove, ForfeitingMove,
    OpeningMove, TransferMove,
)
from model import DurakNet
from state import VALUE_ORDER, encode_combined, encode_options_tensor
from .model import RecurrentActorCritic
from .observation import (
    BeliefMemory, encode_observation, encode_privileged, public_view_from_game,
)


@dataclass
class RolloutStep:
    observation: torch.Tensor
    options: torch.Tensor
    privileged: torch.Tensor
    action: int
    old_log_prob: float
    old_value: float
    reward: float = 0.0
    done: bool = False
    tactical_rewards: torch.Tensor | None = None


class BaseController:
    def reset(self, seat: int) -> None:
        self.seat = seat
        self.memory = BeliefMemory()

    def observe(self, move, actor_seat: int, table_before) -> None:
        self.memory.observe(move, actor_seat == self.seat, table_before)


class RecurrentController(BaseController):
    def __init__(
        self,
        model: RecurrentActorCritic,
        deterministic: bool = True,
        temperature: float = 1.0,
        record: bool = False,
    ):
        self.model = model
        self.deterministic = deterministic
        self.temperature = temperature
        self.record = record
        self.trajectory: list[RolloutStep] = []

    def reset(self, seat: int) -> None:
        super().reset(seat)
        self.hidden = self.model.initial_hidden()
        self.belief_hidden = (
            self.model.initial_belief_hidden()
            if hasattr(self.model, "initial_belief_hidden") else None
        )
        self.trajectory = []

    def choose(self, game, seat: int, options) -> int:
        observation = encode_observation(public_view_from_game(game, seat, self.memory))
        encoded_options = encode_options_tensor(options)
        privileged = encode_privileged(game, seat)
        with torch.no_grad():
            if self.belief_hidden is not None:
                _, self.belief_hidden = self.model.belief_step(
                    observation, self.belief_hidden,
                )
            logits, value, self.hidden = self.model.forward_step(
                observation, encoded_options, self.hidden, privileged,
                self.belief_hidden,
            )
            distribution = Categorical(logits=logits / self.temperature)
            if self.deterministic:
                action = int(logits.argmax().item())
            else:
                action = int(distribution.sample().item())
            log_prob = float(distribution.log_prob(torch.tensor(action)).item())
        if self.record:
            option_rewards = torch.tensor([
                tactical_reward(game, option, options) for option in options
            ], dtype=torch.float32)
            self.trajectory.append(RolloutStep(
                observation, encoded_options, privileged, action,
                log_prob, float(value.item()),
                float(option_rewards[action]), False, option_rewards,
            ))
        return action


class SearchController(BaseController):
    """Deterministic recurrent player whose final choice is refined by IS-MCTS."""

    def __init__(
        self, model: RecurrentActorCritic, simulations: int = 128,
        time_limit_ms: int = 300,
        **search_kwargs,
    ):
        from .search import InformationSetMCTS
        self.model = model
        self.search = InformationSetMCTS(
            model, simulations=simulations, time_limit_ms=time_limit_ms,
            **search_kwargs,
        )

    def reset(self, seat: int) -> None:
        super().reset(seat)
        self.search.reset()
        self.search_results = []
        self.hidden = self.model.initial_hidden()
        self.belief_hidden = (
            self.model.initial_belief_hidden()
            if hasattr(self.model, "initial_belief_hidden") else None
        )

    def choose(self, game, seat: int, options) -> int:
        from .search import root_from_game
        observation = encode_observation(public_view_from_game(game, seat, self.memory))
        hidden_before = self.hidden
        with torch.no_grad():
            if self.belief_hidden is not None:
                _, self.belief_hidden = self.model.belief_step(
                    observation, self.belief_hidden,
                )
            _, _, self.hidden = self.model.forward_step(
                observation, encode_options_tensor(options), self.hidden,
                encode_privileged(game, seat), self.belief_hidden,
            )
            if self.belief_hidden is not None:
                opponent_weights = self.model.opponent_weights(self.belief_hidden)
            else:
                opponent_weights = None
        result = self.search.search(
            root_from_game(game, seat, self.memory), options, hidden_before,
            opponent_weights,
        )
        self.last_search_result = result
        self.search_results.append(result)
        return result.action_index


class RandomController(BaseController):
    def choose(self, game, seat: int, options) -> int:
        return random.randrange(len(options))


class FirstController(BaseController):
    def choose(self, game, seat: int, options) -> int:
        return 0


def _rank(card) -> int:
    return VALUE_ORDER.index(card.value)


def tactical_reward(game, move, options) -> float:
    """Provide small immediate credit for clear trump-management choices."""
    trump = game.trump.suit
    reward = 0.0
    if game.deck and isinstance(move, (OpeningMove, AttackingMove)):
        cards = move.cards if isinstance(move, AttackingMove) else [move.card]
        same_kind = type(move)
        nontrump_alternative = any(
            isinstance(option, same_kind)
            and any(card.suit != trump for card in (
                option.cards if isinstance(option, AttackingMove) else [option.card]
            ))
            for option in options
        )
        if nontrump_alternative:
            phase = min(len(game.deck), 24) / 24.0
            reward -= (0.03 + 0.03 * phase) * sum(card.suit == trump for card in cards)
        if isinstance(move, AttackingMove) and any(
            isinstance(option, EndingMove) for option in options
        ):
            reward -= 0.04 * sum(card.suit == trump for card in cards)

    if isinstance(move, DefensiveMove) and move.card.suit == trump:
        attack = game.table.attack[len(game.table.defense)]
        same_suit = [
            option.card for option in options
            if isinstance(option, DefensiveMove)
            and option.card.suit == attack.suit
            and option.card.suit != trump
        ]
        if same_suit:
            reward -= 0.08 if game.deck else 0.04
        trump_options = [
            option.card for option in options
            if isinstance(option, DefensiveMove) and option.card.suit == trump
        ]
        if trump_options:
            reward -= 0.02 * (_rank(move.card) - min(map(_rank, trump_options)))
    return reward


class HeuristicController(BaseController):
    """A stable tactical baseline that conserves trumps while the deck exists."""

    def choose(self, game, seat: int, options) -> int:
        trump = game.trump.suit
        deck_nonempty = bool(game.deck)

        def score(move) -> float:
            match move:
                case OpeningMove(card):
                    return -_rank(card) - (20.0 if deck_nonempty and card.suit == trump else 0.0)
                case AttackingMove(cards):
                    trump_cost = sum(card.suit == trump for card in cards)
                    return 5.0 * len(cards) - sum(_rank(card) for card in cards) - 20.0 * trump_cost
                case DefensiveMove(card):
                    attack = game.table.attack[len(game.table.defense)]
                    same_suit = card.suit == attack.suit
                    return 20.0 * same_suit - _rank(card) - 8.0 * (card.suit == trump and not same_suit)
                case TransferMove(card):
                    return 12.0 - _rank(card) - 4.0 * (card.suit == trump)
                case ForfeitingMove():
                    return -25.0
                case EndingMove():
                    return 10.0
                case FinishingMove(cards):
                    return 8.0 * len(cards) + sum(_rank(card) for card in cards) - 5.0 * sum(
                        card.suit == trump for card in cards
                    )
            return -100.0

        return max(range(len(options)), key=lambda index: score(options[index]))


class RecordingTeacherController(BaseController):
    """Record public observations while delegating decisions to a teacher."""

    def __init__(self, teacher: BaseController):
        self.teacher = teacher
        self.examples: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def reset(self, seat: int) -> None:
        super().reset(seat)
        self.teacher.reset(seat)
        self.examples = []

    def observe(self, move, actor_seat: int, table_before) -> None:
        super().observe(move, actor_seat, table_before)
        self.teacher.observe(move, actor_seat, table_before)

    def choose(self, game, seat: int, options) -> int:
        observation = encode_observation(public_view_from_game(game, seat, self.memory))
        encoded_options = encode_options_tensor(options)
        action = self.teacher.choose(game, seat, options)
        self.examples.append((observation, encoded_options, action))
        return action


class V2Controller(BaseController):
    def __init__(self, model: DurakNet):
        self.model = model

    @classmethod
    def from_checkpoint(cls, path) -> "V2Controller":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = DurakNet()
        model.load_compatible_state_dict(saved.get("model", saved))
        model.eval()
        return cls(model)

    def choose(self, game, seat: int, options) -> int:
        hand = game.player1.hand if seat == 1 else game.player2.hand
        with torch.no_grad():
            return int(self.model(encode_combined(hand, game.state, options)).argmax().item())


class GuardedV2Controller(V2Controller):
    """Strong v2 teacher with the same bounded tactical correction as live play."""

    def __init__(self, model: DurakNet, tactical_strength: float = 4.0):
        super().__init__(model)
        self.tactical_strength = tactical_strength

    def choose(self, game, seat: int, options) -> int:
        hand = game.player1.hand if seat == 1 else game.player2.hand
        with torch.no_grad():
            scores = self.model(encode_combined(hand, game.state, options))
        corrections = torch.tensor([
            tactical_reward(game, option, options) for option in options
        ]) * self.tactical_strength
        return int((scores + corrections).argmax().item())


def frozen_model(model: RecurrentActorCritic) -> RecurrentActorCritic:
    result = copy.deepcopy(model)
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result
