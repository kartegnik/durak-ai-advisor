import dataclasses as dc
import itertools
from collections import defaultdict

from .gameutils import GameConfiguration, GameState, Table
from .history import GameHistory
from .move import (
    AttackingMove, DefensiveMove, EndingMove, FinishingMove, ForfeitingMove,
    Move, OpeningMove, TransferMove,
)
from .player import Player
from .primitives import Card, GameResult, Switch


@dc.dataclass
class Game:
    """Manage a 36-card game of Durak."""

    deck: list[Card]
    player1: Player
    player2: Player
    turn: Switch = dc.field(default=Switch.First, init=False)
    move: Switch = dc.field(default=Switch.First, init=False)
    trump: Card = dc.field(init=False)
    table: Table = dc.field(default_factory=Table, init=False)
    discardPile: set[Card] = dc.field(default_factory=set, init=False)
    history: GameHistory = dc.field(init=False)
    config: GameConfiguration = dc.field(default_factory=GameConfiguration)

    @property
    def isFinished(self) -> bool:
        if not self.table.isEmpty:
            return False
        return not self.player1.hand or not self.player2.hand

    @property
    def state(self) -> GameState:
        return GameState(
            deckCC=len(self.deck),
            trumpCard=self.trump,
            table=self.table,
            discardPile=self.discardPile,
        )

    @property
    def currentPlayer(self) -> Player:
        return self.player1 if self.move == Switch.First else self.player2

    @property
    def currentOpponent(self) -> Player:
        return self.player2 if self.move == Switch.First else self.player1

    @property
    def result(self) -> GameResult:
        match not self.player1.hand, not self.player2.hand:
            case True, True:
                return GameResult.Draw
            case True, False:
                return GameResult.Win
            case False, True:
                return GameResult.Loss
            case False, False:
                return GameResult.Unfinished
        raise AssertionError

    def __post_init__(self):
        assert len(set(self.deck)) == self.config.InitialDeckCC
        index = 0
        for _ in range(self.config.HandCC):
            self.player1.hand.append(self.deck[index])
            self.player2.hand.append(self.deck[index + 1])
            index += 2
        self.trump = self.deck[index]
        self.deck = self.deck[index + 1:] + [self.trump]
        self.history = GameHistory(
            list(self.player1.hand), list(self.player2.hand), list(self.deck),
        )

    def kickoff(self, numberOfMoves: int | None = None):
        moves = 0
        while not self.isFinished:
            if numberOfMoves is not None and moves >= 2 * numberOfMoves:
                break
            options = self.generateOptions()
            choice = self.currentPlayer.nextMove(self.state, options)
            self.updateHistory(options[choice])
            self.process(options[choice])
            moves += 1
        self.history.result = self.result

    def updateHistory(self, move: Move):
        if self.move == Switch.First:
            self.history.p1Moves.append(move)
        else:
            self.history.p2Moves.append(move)

    def generateOptions(self) -> list[Move]:
        if self.table.isEmpty:
            assert self.turn == self.move
            return [OpeningMove(card) for card in self.currentPlayer.hand]
        if self.turn == self.move:
            return self.generateAttackOptions()
        return self.generateDefenseOptions()

    def generateAttackOptions(self) -> list[Move]:
        cards = self.currentPlayer.hand
        tossable_values = self.table.getTossableValues()
        if not self.currentOpponent.hand:
            return [EndingMove()]

        maximum = self.config.MaxTossAD if self.discardPile else self.config.MaxTossBD
        defender_had = len(self.currentOpponent.hand) + len(self.table.defense)
        limit = min(maximum, defender_had)
        attack_cards = [card for card in cards if card.value in tossable_values]

        if self.table.isForfeited:
            opponent_count = len(self.table.defense) + len(self.currentOpponent.hand)
            cards_to_toss = min(opponent_count, self.config.HandCC) - len(self.table.attack)
            result: list[Move] = []
            count = max(0, min(len(attack_cards), cards_to_toss))
            for size in range(count + 1):
                result.extend(
                    FinishingMove(list(variant))
                    for variant in itertools.combinations(attack_cards, size)
                )
            return result

        can_toss = limit - len(self.table.attack)
        if can_toss <= 0:
            return [EndingMove()] if len(self.table.defense) == len(self.table.attack) else []

        by_value: dict[object, list[Card]] = defaultdict(list)
        for card in attack_cards:
            by_value[card.value].append(card)
        result = []
        for cards_of_value in by_value.values():
            count = min(len(cards_of_value), can_toss)
            for size in range(1, count + 1):
                result.extend(
                    AttackingMove(list(variant))
                    for variant in itertools.combinations(cards_of_value, size)
                )
        if len(self.table.defense) == len(self.table.attack):
            result.append(EndingMove())
        return result

    def generateDefenseOptions(self) -> list[Move]:
        assert len(self.table.attack) >= len(self.table.defense) + 1
        undefended_index = len(self.table.defense)
        card_to_beat = self.table.attack[undefended_index]
        result: list[Move] = [
            DefensiveMove(card) for card in self.currentPlayer.hand
            if self.canBeat(card, card_to_beat)
        ]
        if self.config.allowTransfer and not self.table.defenderHasDefended:
            attack_after_transfer = len(self.table.attack) + 1
            if attack_after_transfer <= len(self.currentOpponent.hand):
                result.extend(
                    TransferMove(card) for card in self.currentPlayer.hand
                    if card.value == card_to_beat.value
                )
        return result + [ForfeitingMove()]

    def process(self, move: Move):
        match move:
            case OpeningMove(card):
                assert self.turn == self.move
                self.currentPlayer.hand.remove(card)
                self.table.attack.append(card)
                self.toggleMove()
            case AttackingMove(cards):
                assert self.turn == self.move
                for card in cards:
                    self.currentPlayer.hand.remove(card)
                    self.table.attack.append(card)
                self.toggleMove()
            case DefensiveMove(card):
                assert self.turn != self.move
                self.currentPlayer.hand.remove(card)
                self.table.defense.append(card)
                self.table.defenderHasDefended = True
                if len(self.table.defense) == len(self.table.attack):
                    self.toggleMove()
            case ForfeitingMove():
                assert self.turn != self.move
                self.table.isForfeited = True
                self.toggleMove()
            case EndingMove():
                assert self.turn == self.move
                self.discardPile = self.discardPile.union(
                    self.table.attack, self.table.defense,
                )
                self.table.clear()
                self.takeCards(self.currentPlayer)
                self.takeCards(self.currentOpponent)
                self.toggleTurn()
            case FinishingMove(cards):
                assert self.turn == self.move
                assert self.table.isForfeited
                for card in cards:
                    self.currentPlayer.hand.remove(card)
                self.currentOpponent.hand += self.table.defense
                self.currentOpponent.hand += self.table.attack
                self.currentOpponent.hand += cards
                self.table.clear()
                self.takeCards(self.currentPlayer)
            case TransferMove(card):
                assert self.turn != self.move
                self.currentPlayer.hand.remove(card)
                self.table.attack.append(card)
                self.move = self.move.toggled()
                self.turn = self.turn.toggled()
            case _:
                raise AssertionError

    def toggleMove(self):
        self.move = self.move.toggled()

    def toggleTurn(self):
        assert self.table.isEmpty
        self.toggleMove()
        self.turn = self.turn.toggled()

    def takeCards(self, player: Player):
        needed = max(self.config.HandCC - len(player.hand), 0)
        count = min(needed, len(self.deck))
        player.hand.extend(self.deck[:count])
        self.deck = self.deck[count:]

    def canBeat(self, card: Card, another: Card) -> bool:
        if self.isTrump(card):
            return not self.isTrump(another) or card.value > another.value
        return card.suit == another.suit and card.value > another.value

    def isTrump(self, card: Card) -> bool:
        return card.suit == self.trump.suit
