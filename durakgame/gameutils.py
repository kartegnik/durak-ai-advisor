import dataclasses as dc

from .primitives import Card, CardValue


@dc.dataclass
class Table:
    """A representation of a card battle happening on the table."""
    attack: list[Card] = dc.field(default_factory=list)
    defense: list[Card] = dc.field(default_factory=list)
    isForfeited: bool = False
    defenderHasDefended: bool = False

    @property
    def isEmpty(self) -> bool:
        return not self.attack

    def getTossableValues(self) -> set[CardValue]:
        return {card.value for card in self.attack + self.defense}

    def clear(self):
        self.attack.clear()
        self.defense.clear()
        self.isForfeited = False
        self.defenderHasDefended = False


@dc.dataclass(frozen=True)
class GameConfiguration:
    InitialDeckCC: int = 36
    HandCC: int = 6
    MaxTossBD: int = HandCC - 1
    MaxTossAD: int = HandCC
    allowTransfer: bool = False


@dc.dataclass(frozen=True)
class GameState:
    deckCC: int
    trumpCard: Card
    table: Table
    discardPile: set[Card]
