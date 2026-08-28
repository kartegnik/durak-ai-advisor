#!/usr/bin/env python3
"""Generate legal player actions from the CV tracker's remembered state."""

from __future__ import annotations

from dataclasses import dataclass


RANKS = {rank: index for index, rank in enumerate(("6", "7", "8", "9", "10", "J", "Q", "K", "A"))}
SUITS = {"H", "D", "C", "S"}


def split_card(card: str) -> tuple[str, str]:
    rank, suit = card[:-1], card[-1]
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"unknown card: {card}")
    return rank, suit


def beats(card: str, attack: str, trump: str) -> bool:
    rank, suit = split_card(card)
    attack_rank, attack_suit = split_card(attack)
    if suit == attack_suit:
        return RANKS[rank] > RANKS[attack_rank]
    return suit == trump and attack_suit != trump


@dataclass(frozen=True)
class Advice:
    actions: tuple[str, ...]
    note: str = ""


def legal_actions(tracker, trump: str) -> Advice:
    """Return rule-correct actions; strategy scoring is intentionally separate."""
    trump = trump.upper()
    if trump not in SUITS:
        raise ValueError("trump must be one of H, D, C, S")
    hand = sorted(tracker.hand, key=lambda card: (RANKS[split_card(card)[0]], card[-1]))
    attacks = tracker.table_attack
    defenses = tracker.table_defense

    if not attacks:
        if tracker.player_attacker is None:
            return Advice(
                tuple(f"ход {card}" for card in hand),
                "если первый ход ваш",
            )
        if tracker.player_attacker is False:
            return Advice((), "ожидаем ход соперника")
        return Advice(tuple(f"ход {card}" for card in hand), "выберите карту для атаки")

    ranks_on_table = {split_card(card)[0] for card in attacks + defenses}
    if tracker.player_attacker:
        throws = [card for card in hand if split_card(card)[0] in ranks_on_table]
        actions = [*(f"подкинуть {card}" for card in throws)]
        if len(defenses) >= len(attacks):
            actions.append("бито / пас")
        elif not actions:
            return Advice((), "ожидаем, пока соперник отобьёт или возьмёт")
        return Advice(tuple(actions))

    if len(attacks) <= len(defenses):
        return Advice((), "ожидаем следующую карту соперника")
    attack = attacks[len(defenses)]
    actions = [f"отбить {attack} картой {card}" for card in hand if beats(card, attack, trump)]
    # In transfer Durak a rank can be transferred only before any card was beaten.
    if not defenses:
        attack_rank = split_card(attack)[0]
        actions.extend(f"перевести картой {card}" for card in hand if split_card(card)[0] == attack_rank)
    actions.append("взять")
    return Advice(tuple(actions))
