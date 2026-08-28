#!/usr/bin/env python3
"""Score the CV advisor's legal actions with the trained Durak network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from durakgame import (
    AttackingMove, Card, CardValue, DefensiveMove, EndingMove, ForfeitingMove,
    GameState, OpeningMove, Suit, Table, TransferMove,
)

from legal_move_advisor import Advice
from model import DurakNet
from state import encode_combined


SUITS = {"H": Suit.Heart, "D": Suit.Diamond, "C": Suit.Cross, "S": Suit.Spade}
VALUES = {
    "6": CardValue.Six, "7": CardValue.Seven, "8": CardValue.Eight,
    "9": CardValue.Nine, "10": CardValue.Ten, "J": CardValue.Jack,
    "Q": CardValue.Queen, "K": CardValue.King, "A": CardValue.Ace,
}
VALUE_INDEX = {name: index for index, name in enumerate(VALUES)}


def to_card(name: str) -> Card:
    return Card(VALUES[name[:-1]], SUITS[name[-1]])


def action_to_move(action: str):
    if action.startswith("ход "):
        return OpeningMove(to_card(action.split()[-1]))
    if action.startswith("подкинуть "):
        return AttackingMove([to_card(action.split()[-1])])
    if action.startswith("отбить "):
        return DefensiveMove(to_card(action.split()[-1]))
    if action.startswith("перевести картой "):
        return TransferMove(to_card(action.split()[-1]))
    if action == "взять":
        return ForfeitingMove()
    if action == "бито / пас":
        return EndingMove()
    raise ValueError(f"unsupported action: {action}")


@dataclass(frozen=True)
class NeuralRecommendation:
    action: str
    scores: tuple[tuple[str, float], ...]


def _played_card(action: str) -> str | None:
    """Return the card spent by an action, if the action spends one."""
    if action.startswith(("ход ", "подкинуть ", "отбить ", "перевести картой ")):
        return action.split()[-1]
    return None


def strategic_adjustments(
    actions: tuple[str, ...], trump: str, deck_count: int
) -> tuple[float, ...]:
    """Small, interpretable corrections for weaknesses seen in self-play.

    The network was trained from terminal game rewards.  In practice both old
    and new checkpoints tend to spend trumps too freely while cards remain in
    the deck.  These corrections preserve the network's ranking except for
    clear resource-management mistakes.
    """
    corrections = [0.0] * len(actions)
    played = [_played_card(action) for action in actions]

    # Avoid opening or throwing in with a trump while a non-trump (or pass) is
    # available.  Once the deck is empty, shedding a trump can be exactly right.
    if deck_count > 0:
        deck_phase = min(deck_count, 24) / 24.0
        trump_attack_penalty = 0.12 + 0.12 * deck_phase
        for index, (action, card) in enumerate(zip(actions, played)):
            if card is None or card[-1] != trump:
                continue
            if action.startswith("ход ") and any(
                other is not None and other[-1] != trump
                for other_action, other in zip(actions, played)
                if other_action.startswith("ход ")
            ):
                corrections[index] -= trump_attack_penalty
            elif action.startswith("подкинуть ") and "бито / пас" in actions:
                corrections[index] -= trump_attack_penalty

    # On defense prefer the lowest adequate trump.  If a same-suit defense is
    # available, also account for its rank: spending a six of trumps instead of
    # a king can still be sensible, but spending a nine instead of a jack is not.
    defensive = [
        (index, action, card)
        for index, (action, card) in enumerate(zip(actions, played))
        if action.startswith("отбить ") and card is not None
    ]
    trump_defenses = [(index, card) for index, _, card in defensive if card[-1] == trump]
    if trump_defenses:
        for index, action, card in defensive:
            attack = action.split()[1]
            if card[-1] == attack[-1] and card[-1] != trump:
                rank_margin = VALUE_INDEX[card[:-1]] - VALUE_INDEX[attack[:-1]]
                corrections[index] += max(0.0, 0.08 - 0.015 * rank_margin)

        cheapest_trump = min(VALUE_INDEX[card[:-1]] for _, card in trump_defenses)
        for index, card in trump_defenses:
            corrections[index] -= 0.05 * (VALUE_INDEX[card[:-1]] - cheapest_trump)

        for index, action, card in defensive:
            if card[-1] != trump:
                continue
            attack = action.split()[1]
            same_suit_ranks = [
                VALUE_INDEX[other[:-1]]
                for _, other_action, other in defensive
                if other[-1] == attack[-1] and other[-1] != trump
            ]
            if same_suit_ranks:
                rank_tradeoff = VALUE_INDEX[card[:-1]] - min(same_suit_ranks)
                corrections[index] -= max(0.0, 0.19 + 0.025 * rank_tradeoff)

    return tuple(corrections)


class NeuralMoveAdvisor:
    def __init__(self, checkpoint: Path):
        self.model = DurakNet()
        saved = torch.load(checkpoint, weights_only=True)
        self.model.load_compatible_state_dict(saved.get("model", saved))
        self.model.eval()

    def recommend(self, tracker, advice: Advice, trump: str, deck_count: int) -> NeuralRecommendation:
        moves = [action_to_move(action) for action in advice.actions]
        hand = [to_card(card) for card in tracker.hand]
        table = Table(
            attack=[to_card(card) for card in tracker.table_attack],
            defense=[to_card(card) for card in tracker.table_defense],
            defenderHasDefended=bool(tracker.table_defense),
        )
        # State encoding uses only the trump suit, so its rank is immaterial.
        state = GameState(
            deckCC=deck_count,
            trumpCard=Card(CardValue.Six, SUITS[trump]),
            table=table,
            discardPile={to_card(card) for card in tracker.discard},
        )
        with torch.no_grad():
            values = self.model(encode_combined(hand, state, moves)).tolist()
        corrections = strategic_adjustments(advice.actions, trump, deck_count)
        values = [value + correction for value, correction in zip(values, corrections)]
        ranked = tuple(sorted(zip(advice.actions, values), key=lambda item: item[1], reverse=True))
        return NeuralRecommendation(ranked[0][0], ranked)
