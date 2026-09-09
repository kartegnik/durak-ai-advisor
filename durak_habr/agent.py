"""A transparent Habr-style hand evaluator extended with public card memory.

The article deliberately evaluates only the current hand.  This controller
keeps that evaluator as its base, but also uses cards that have been discarded
or publicly picked up by the opponent.  It never reads the hidden contents of
the opponent hand or deck.
"""

from __future__ import annotations

import math
from collections import Counter

from durakgame import (
    AttackingMove, DefensiveMove, EndingMove, FinishingMove, ForfeitingMove,
    OpeningMove, TransferMove,
)
from durak_v3.agents import BaseController
from durak_v3.observation import move_cards
from state import CARD_TO_INDEX, VALUE_ORDER


RANK_MULTIPLIER = 100
TRUMP_BONUS = 1300
UNBALANCED_HAND_PENALTY = 200
MANY_CARDS_PENALTY = 600
SAME_RANK_BONUSES = (0.0, 0.0, 0.5, 0.75, 1.25)
OUT_OF_PLAY = 1_000_000


def rank_value(card) -> int:
    """Article scale restricted to the 36-card ranks: six=-200, ace=600."""
    return (VALUE_ORDER.index(card.value) - 2) * RANK_MULTIPLIER


def hand_value(hand, trump_suit, deck_count: int, opponent_count: int) -> float:
    """Evaluate a hand using the four components described in the article."""
    cards = list(hand)
    if not cards and deck_count == 0:
        return float(OUT_OF_PLAY)

    rank_counts = Counter(card.value for card in cards)
    suit_counts = Counter(card.suit for card in cards)
    result = sum(
        rank_value(card) + (TRUMP_BONUS if card.suit == trump_suit else 0)
        for card in cards
    )

    # Equal ranks cooperate during both throwing and defense.  The article's
    # coefficients are retained, while the base is kept on the same 100-point
    # scale as the rest of the evaluator.
    for card_value, count in rank_counts.items():
        sample = next(card for card in cards if card.value == card_value)
        result += max(rank_value(sample), RANK_MULTIPLIER) * SAME_RANK_BONUSES[count]

    non_trump_count = sum(card.suit != trump_suit for card in cards)
    average = non_trump_count / 3.0
    if average > 0:
        for suit in type(trump_suit):
            if suit != trump_suit:
                deviation = abs(suit_counts[suit] - average) / average
                result -= UNBALANCED_HAND_PENALTY * deviation

    other_cards = deck_count + opponent_count
    ratio = len(cards) / other_cards if other_cards else (10.0 if cards else 0.0)
    result += (0.25 - ratio) * MANY_CARDS_PENALTY
    return float(result)


def can_beat(card, attack, trump_suit) -> bool:
    if card.suit == trump_suit:
        return attack.suit != trump_suit or card.value > attack.value
    return card.suit == attack.suit and card.value > attack.value


def _probability_at_least_one(population: int, successes: int, draws: int) -> float:
    if successes <= 0 or draws <= 0 or population <= 0:
        return 0.0
    successes = min(successes, population)
    draws = min(draws, population)
    if population - successes < draws:
        return 1.0
    return 1.0 - math.comb(population - successes, draws) / math.comb(population, draws)


class HabrMemoryController(BaseController):
    """One-ply Habr evaluator with explicit public-memory risk estimates."""

    def _own_hand(self, game, seat: int):
        return game.player1.hand if seat == 1 else game.player2.hand

    def _opponent_count(self, game, seat: int) -> int:
        opponent = game.player2 if seat == 1 else game.player1
        return len(opponent.hand)

    def unseen_cards(self, game, seat: int) -> set:
        visible = set(self._own_hand(game, seat))
        visible.update(game.table.attack)
        visible.update(game.table.defense)
        visible.update(game.discardPile)
        visible.update(self.memory.known_opponent)
        return set(CARD_TO_INDEX) - visible

    def opponent_can_beat_probability(self, game, seat: int, attack) -> float:
        known = self.memory.known_opponent
        if any(can_beat(card, attack, game.trump.suit) for card in known):
            return 1.0
        unseen = self.unseen_cards(game, seat)
        beating = sum(can_beat(card, attack, game.trump.suit) for card in unseen)
        unknown_slots = max(0, self._opponent_count(game, seat) - len(known))
        return _probability_at_least_one(len(unseen), beating, unknown_slots)

    def projected_hand(self, game, seat: int, move) -> list:
        hand = list(self._own_hand(game, seat))
        for card in move_cards(move):
            if card in hand:
                hand.remove(card)
        if isinstance(move, ForfeitingMove):
            hand.extend(game.table.attack)
            hand.extend(game.table.defense)
        return hand

    def score_move(self, game, seat: int, move) -> float:
        hand = self.projected_hand(game, seat, move)
        score = hand_value(
            hand, game.trump.suit, len(game.deck), self._opponent_count(game, seat),
        )

        # Public-memory terms.  These estimate whether an attack is likely to
        # survive without ever inspecting the real hidden hand.
        if isinstance(move, (OpeningMove, AttackingMove)):
            attacks = [move.card] if isinstance(move, OpeningMove) else move.cards
            resistance = sum(
                self.opponent_can_beat_probability(game, seat, card)
                for card in attacks
            ) / max(1, len(attacks))
            score += 180.0 * (1.0 - resistance) + 35.0 * len(attacks)
        elif isinstance(move, DefensiveMove):
            # A raw hand-strength comparison otherwise prefers taking cards to
            # spending a valuable defender.  Completing a legal defense must
            # therefore carry the positional value of avoiding the pickup.
            score += 1900.0
            if move.card.suit == game.trump.suit:
                score -= 180.0 if game.deck else 60.0
        elif isinstance(move, TransferMove):
            score += 1300.0
            if move.card.suit == game.trump.suit:
                score -= 160.0
        elif isinstance(move, ForfeitingMove):
            score -= 1700.0 + 80.0 * (len(game.table.attack) + len(game.table.defense))
        elif isinstance(move, EndingMove):
            score += 260.0 + 30.0 * (len(game.table.attack) + len(game.table.defense))
        elif isinstance(move, FinishingMove):
            score += 80.0 * len(move.cards)

        # In the endgame, exploit cards publicly known to be in the opponent's
        # hand.  This is the main deliberate extension beyond the article.
        if not game.deck and self.memory.known_opponent:
            coverage = sum(
                any(can_beat(ours, theirs, game.trump.suit) for ours in hand)
                for theirs in self.memory.known_opponent
            )
            score += 55.0 * coverage
        return score

    def choose(self, game, seat: int, options) -> int:
        return max(
            range(len(options)),
            key=lambda index: (self.score_move(game, seat, options[index]), -index),
        )
