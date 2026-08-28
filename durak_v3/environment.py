"""Game runner that exposes public events while keeping actor observations honest."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Protocol

from durakgame import GameConfiguration, GameResult, Player, standardDeck
from durakgame.game import Game
from durakgame.primitives import Switch


GAME_CONFIG = GameConfiguration(allowTransfer=True)
MAX_ACTIONS = 300


class Controller(Protocol):
    def reset(self, seat: int) -> None: ...
    def choose(self, game: Game, seat: int, options) -> int: ...
    def observe(self, move, actor_seat: int, table_before) -> None: ...


@dataclass(frozen=True)
class EpisodeResult:
    result: GameResult
    actions: int
    trump_plays_with_deck: tuple[int, int]
    avoidable_trump_defenses: tuple[int, int]


def shuffled_deck(seed: int | None = None):
    deck = standardDeck()
    (random if seed is None else random.Random(seed)).shuffle(deck)
    return deck


def seat_to_switch(seat: int) -> Switch:
    return Switch.First if seat == 1 else Switch.Second


def result_for_seat(result: GameResult, seat: int) -> GameResult:
    if seat == 1:
        return result
    if result == GameResult.Win:
        return GameResult.Loss
    if result == GameResult.Loss:
        return GameResult.Win
    return result


def play_episode(
    controller1: Controller,
    controller2: Controller,
    deck=None,
    max_actions: int = MAX_ACTIONS,
) -> EpisodeResult:
    game = Game(
        list(deck) if deck is not None else shuffled_deck(),
        Player(), Player(), config=GAME_CONFIG,
    )
    controllers = {1: controller1, 2: controller2}
    for seat, controller in controllers.items():
        controller.reset(seat)

    trump_plays = [0, 0]
    avoidable_trumps = [0, 0]
    action_count = 0
    while not game.isFinished and action_count < max_actions:
        seat = 1 if game.move == Switch.First else 2
        options = game.generateOptions()
        choice = controllers[seat].choose(game, seat, options)
        if not 0 <= choice < len(options):
            raise IndexError(f"controller returned {choice} for {len(options)} options")
        move = options[choice]

        from durakgame import AttackingMove, DefensiveMove, OpeningMove
        from durak_v3.observation import move_cards
        spent = move_cards(move)
        if game.deck and isinstance(move, (OpeningMove, AttackingMove)):
            from durakgame import EndingMove
            same_kind = type(move)
            has_safe_alternative = any(
                isinstance(option, same_kind)
                and any(card.suit != game.trump.suit for card in move_cards(option))
                for option in options
            ) or (
                isinstance(move, AttackingMove)
                and any(isinstance(option, EndingMove) for option in options)
            )
            if has_safe_alternative:
                trump_plays[seat - 1] += sum(
                    card.suit == game.trump.suit for card in spent
                )
        if isinstance(move, DefensiveMove) and move.card.suit == game.trump.suit:
            attack = game.table.attack[len(game.table.defense)]
            if attack.suit != game.trump.suit:
                same_suit_exists = any(
                    isinstance(option, DefensiveMove)
                    and option.card.suit == attack.suit
                    for option in options
                )
                avoidable_trumps[seat - 1] += int(same_suit_exists)

        table_before = copy.deepcopy(game.table)
        game.updateHistory(move)
        game.process(move)
        for observer_seat, controller in controllers.items():
            controller.observe(move, seat, table_before)
        action_count += 1

    result = game.result if game.isFinished else GameResult.Draw
    game.history.result = result
    return EpisodeResult(
        result, action_count, tuple(trump_plays), tuple(avoidable_trumps),
    )
