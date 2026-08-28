import dataclasses as dc
import random

from .gameutils import GameState
from .move import Move
from .primitives import Card


@dc.dataclass
class Player:
    hand: list[Card] = dc.field(default_factory=list, init=False)

    def nextMove(self, state: GameState, options: list[Move]) -> int:
        raise NotImplementedError("Subclasses should implement that")


@dc.dataclass
class MrFirst(Player):
    def nextMove(self, state: GameState, options: list[Move]) -> int:
        return 0


@dc.dataclass
class MrRandom(Player):
    def nextMove(self, state: GameState, options: list[Move]) -> int:
        return random.randint(0, len(options) - 1)
