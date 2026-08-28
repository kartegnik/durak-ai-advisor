from dataclasses import dataclass
from enum import Enum, auto
from functools import total_ordering


@total_ordering
class Suit(Enum):
    Heart = 0
    Cross = 1
    Spade = 2
    Diamond = 3

    def __lt__(self, other: "Suit") -> bool:
        return self.value < other.value


@total_ordering
class CardValue(Enum):
    Six = auto()
    Seven = auto()
    Eight = auto()
    Nine = auto()
    Ten = auto()
    Jack = auto()
    Queen = auto()
    King = auto()
    Ace = auto()

    def __lt__(self, other: "CardValue") -> bool:
        return self.value < other.value


@dataclass(order=True, eq=True, frozen=True)
class Card:
    value: CardValue
    suit: Suit

    def __repr__(self):
        return f"{self.value.name}({self.suit.name})"

    def __str__(self) -> str:
        return f"{self.value.name}({self.suit.name})"


class Switch(Enum):
    First = 1
    Second = 2

    def toggled(self) -> "Switch":
        match self:
            case Switch.First:
                return Switch.Second
            case Switch.Second:
                return Switch.First
        assert False


class GameResult(Enum):
    Win = 1
    Draw = 0
    Loss = -1
    Unfinished = -2
