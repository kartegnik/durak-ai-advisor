#!/usr/bin/env python3
"""Compare stronger information-set search budgets against raw v4."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from durakgame import GameResult
from durak_v3.agents import RecurrentController, SearchController
from durak_v3.environment import play_episode, result_for_seat, shuffled_deck
from train_v4 import load_v4


def evaluate(model, simulations: int, games_per_seat: int, seed: int, search_ms: int):
    counts = {"wins": 0, "draws": 0, "losses": 0}
    searches = simulations_done = determinizations = exact = 0
    reused = leaf_depth = quiet_steps = 0.0
    for index in range(games_per_seat):
        deck = shuffled_deck(seed + index)
        for seat in (1, 2):
            random.seed((seed + index) * 2 + seat)
            search = SearchController(
                model, simulations=simulations, time_limit_ms=search_ms,
                determinizations=max(4, round(simulations ** 0.5)),
                quiescence=True, habr_leaf_weight=0.10, reuse_tree=True,
            )
            raw = RecurrentController(model, deterministic=True)
            outcome = play_episode(
                search if seat == 1 else raw,
                raw if seat == 1 else search,
                deck,
            )
            result = result_for_seat(outcome.result, seat)
            if result == GameResult.Win:
                counts["wins"] += 1
            elif result == GameResult.Loss:
                counts["losses"] += 1
            else:
                counts["draws"] += 1
            for item in search.search_results:
                searches += 1
                simulations_done += item.simulations
                determinizations += item.unique_determinizations
                exact += int(item.solved_exactly)
                reused += item.reused_root_visits
                leaf_depth += item.mean_leaf_depth
                quiet_steps += item.mean_quiescence_steps
        if (index + 1) % 10 == 0:
            print(
                f"simulations={simulations} paired_decks={index + 1}/{games_per_seat}",
                flush=True,
            )
    total = 2 * games_per_seat
    return {
        "games": total,
        **counts,
        "points": (counts["wins"] + 0.5 * counts["draws"]) / total,
        "search_calls": searches,
        "completed_simulations_per_call": simulations_done / max(1, searches),
        "determinizations_per_call": determinizations / max(1, searches),
        "exact_endgames": exact,
        "reused_root_visits_per_call": reused / max(1, searches),
        "mean_leaf_depth": leaf_depth / max(1, searches),
        "mean_quiescence_steps": quiet_steps / max(1, searches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("durak_model_v4.pt"))
    parser.add_argument("--simulations", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--games-per-seat", type=int, default=100)
    parser.add_argument("--search-ms", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/durak_v5_search/search_strength.json"),
    )
    args = parser.parse_args()
    model, _ = load_v4(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": str(args.model),
        "games_per_seat": args.games_per_seat,
        "search_ms": args.search_ms,
        "results": {},
    }
    for simulations in args.simulations:
        started = time.monotonic()
        result["results"][str(simulations)] = evaluate(
            model, simulations, args.games_per_seat, args.seed, args.search_ms,
        )
        result["results"][str(simulations)]["elapsed_seconds"] = (
            time.monotonic() - started
        )
        # Each budget is valuable and can take hours, so make progress durable.
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(
            json.dumps({str(simulations): result["results"][str(simulations)]}, indent=2),
            flush=True,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
