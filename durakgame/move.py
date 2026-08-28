import dataclasses as dc

from .primitives import Card


@dc.dataclass
class Move:
    """Base class for all kinds of moves you can do in Durak."""


@dc.dataclass
class AttackingMove(Move):
    """Continue an attack with one or more cards of the same rank."""
    cards: list[Card]


@dc.dataclass
class DefensiveMove(Move):
    card: Card


@dc.dataclass
class ForfeitingMove(Move):
    """Forfeit the table and take all its cards."""


@dc.dataclass
class OpeningMove(Move):
    card: Card


@dc.dataclass
class EndingMove(Move):
    """End the bout and move the table to the discard pile."""


@dc.dataclass
class FinishingMove(Move):
    """Give optional final cards after the defender forfeits."""
    cards: list[Card]


@dc.dataclass
class TransferMove(Move):
    """Transfer an attack by playing another card of the same rank."""
    card: Card
