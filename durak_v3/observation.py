"""Honest public observations and persistent card beliefs for Durak v3."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from durakgame import (
    AttackingMove, Card, DefensiveMove, EndingMove, FinishingMove,
    ForfeitingMove, Move, OpeningMove, TransferMove,
)
from state import CARD_TO_INDEX, NUM_CARDS, cards_to_multi_hot


MOVE_KINDS = (
    OpeningMove, AttackingMove, DefensiveMove, ForfeitingMove,
    EndingMove, FinishingMove, TransferMove,
)
EVENT_DIM = len(MOVE_KINDS) + NUM_CARDS + 2
SCALAR_DIM = 9
OBSERVATION_DIM = NUM_CARDS * 6 + 4 + 9 + 1 + SCALAR_DIM + EVENT_DIM
PRIVILEGED_DIM = NUM_CARDS * 2


def move_cards(move: Move) -> list[Card]:
    match move:
        case OpeningMove(card) | DefensiveMove(card) | TransferMove(card):
            return [card]
        case AttackingMove(cards) | FinishingMove(cards):
            return list(cards)
        case _:
            return []


def move_kind(move: Move) -> int:
    for index, cls in enumerate(MOVE_KINDS):
        if isinstance(move, cls):
            return index
    raise TypeError(type(move).__name__)


@dataclass
class BeliefMemory:
    """Cards publicly known to remain in the opponent's hand plus last event."""

    known_opponent: set[Card] = field(default_factory=set)
    last_kind: int | None = None
    last_cards: tuple[Card, ...] = ()
    last_actor_self: bool | None = None

    def clone(self) -> "BeliefMemory":
        return BeliefMemory(
            set(self.known_opponent), self.last_kind,
            tuple(self.last_cards), self.last_actor_self,
        )

    def observe(self, move: Move, actor_is_self: bool, table_before) -> None:
        cards = move_cards(move)
        if not actor_is_self:
            self.known_opponent.difference_update(cards)

        # A finishing move completes a take. If we are the attacker, every
        # public table card is now known to be in the opponent's hand.
        if isinstance(move, FinishingMove) and actor_is_self:
            self.known_opponent.update(table_before.attack)
            self.known_opponent.update(table_before.defense)
            self.known_opponent.update(cards)

        self.last_kind = move_kind(move)
        self.last_cards = tuple(cards)
        self.last_actor_self = actor_is_self

    def event_vector(self) -> list[float]:
        result = [0.0] * EVENT_DIM
        if self.last_kind is None:
            return result
        result[self.last_kind] = 1.0
        offset = len(MOVE_KINDS)
        for card in self.last_cards:
            result[offset + CARD_TO_INDEX[card]] = 1.0
        actor_offset = offset + NUM_CARDS
        result[actor_offset + (0 if self.last_actor_self else 1)] = 1.0
        return result


@dataclass(frozen=True)
class PublicView:
    hand: tuple[Card, ...]
    attack: tuple[Card, ...]
    defense: tuple[Card, ...]
    discard: frozenset[Card]
    known_opponent: frozenset[Card]
    trump_suit: int
    trump_rank: int | None
    deck_count: int
    opponent_count: int
    is_attacker: bool
    table_forfeited: bool
    defender_has_defended: bool
    last_event: tuple[float, ...]


def public_view_from_game(game, seat: int, memory: BeliefMemory) -> PublicView:
    own = game.player1 if seat == 1 else game.player2
    opponent = game.player2 if seat == 1 else game.player1
    seat_switch = 1 if game.turn.value == 1 else 2
    return PublicView(
        hand=tuple(own.hand),
        attack=tuple(game.table.attack),
        defense=tuple(game.table.defense),
        discard=frozenset(game.discardPile),
        known_opponent=frozenset(memory.known_opponent),
        trump_suit=game.trump.suit.value,
        trump_rank=game.trump.value.value - 1,
        deck_count=len(game.deck),
        opponent_count=len(opponent.hand),
        is_attacker=seat_switch == seat,
        table_forfeited=game.table.isForfeited,
        defender_has_defended=game.table.defenderHasDefended,
        last_event=tuple(memory.event_vector()),
    )


def encode_observation(view: PublicView) -> torch.Tensor:
    visible = set(view.hand) | set(view.attack) | set(view.defense) | set(view.discard)
    visible |= set(view.known_opponent)
    unknown = set(CARD_TO_INDEX) - visible

    vector: list[float] = []
    vector.extend(cards_to_multi_hot(view.hand))
    vector.extend(cards_to_multi_hot(view.attack))
    vector.extend(cards_to_multi_hot(view.defense))
    vector.extend(cards_to_multi_hot(view.discard))
    vector.extend(cards_to_multi_hot(view.known_opponent))
    vector.extend(cards_to_multi_hot(unknown))

    trump_suit = [0.0] * 4
    trump_suit[view.trump_suit] = 1.0
    vector.extend(trump_suit)
    trump_rank = [0.0] * 9
    if view.trump_rank is not None:
        trump_rank[view.trump_rank] = 1.0
    vector.extend(trump_rank)
    vector.append(1.0 if view.trump_rank is not None else 0.0)

    vector.extend([
        min(view.deck_count, 24) / 24.0,
        len(view.hand) / 36.0,
        view.opponent_count / 36.0,
        len(view.attack) / 6.0,
        len(view.defense) / 6.0,
        1.0 if view.is_attacker else 0.0,
        0.0 if view.is_attacker else 1.0,
        1.0 if view.table_forfeited else 0.0,
        1.0 if view.defender_has_defended else 0.0,
    ])
    vector.extend(view.last_event)
    assert len(vector) == OBSERVATION_DIM
    return torch.tensor(vector, dtype=torch.float32)


def encode_privileged(game, seat: int) -> torch.Tensor:
    opponent = game.player2 if seat == 1 else game.player1
    vector = cards_to_multi_hot(opponent.hand) + cards_to_multi_hot(game.deck)
    assert len(vector) == PRIVILEGED_DIM
    return torch.tensor(vector, dtype=torch.float32)
