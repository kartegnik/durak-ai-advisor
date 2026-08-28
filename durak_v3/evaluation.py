"""Paired fixed-deck evaluation with tactical diagnostics."""

from __future__ import annotations

import random
from collections.abc import Callable

from durakgame import GameResult
from .environment import play_episode, result_for_seat, shuffled_deck


def evaluate_factories(
    learner_factory: Callable[[], object],
    opponent_factory: Callable[[], object],
    games_per_seat: int,
    seed: int,
) -> dict[str, float]:
    counts = {"wins": 0, "draws": 0, "losses": 0}
    early_trumps = 0
    avoidable_trumps = 0
    actions = 0
    saved_state = random.getstate()
    try:
        for index in range(games_per_seat):
            game_seed = seed + index
            deck = shuffled_deck(game_seed)
            for seat in (1, 2):
                random.seed(game_seed * 2 + seat)
                learner = learner_factory()
                opponent = opponent_factory()
                episode = play_episode(
                    learner if seat == 1 else opponent,
                    opponent if seat == 1 else learner,
                    deck,
                )
                result = result_for_seat(episode.result, seat)
                if result == GameResult.Win:
                    counts["wins"] += 1
                elif result == GameResult.Loss:
                    counts["losses"] += 1
                else:
                    counts["draws"] += 1
                early_trumps += episode.trump_plays_with_deck[seat - 1]
                avoidable_trumps += episode.avoidable_trump_defenses[seat - 1]
                actions += episode.actions
    finally:
        random.setstate(saved_state)
    total = 2 * games_per_seat
    return {
        "games": float(total),
        "wins": float(counts["wins"]),
        "draws": float(counts["draws"]),
        "losses": float(counts["losses"]),
        "win_rate": counts["wins"] / total,
        "points": (counts["wins"] + 0.5 * counts["draws"]) / total,
        "early_trumps_per_game": early_trumps / total,
        "avoidable_trump_defenses_per_game": avoidable_trumps / total,
        "actions_per_game": actions / total,
    }
