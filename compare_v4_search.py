#!/usr/bin/env python3
"""Compare uniform and learned-belief information-set search on fixed decks."""

import argparse
import json
from pathlib import Path

from durak_v3.agents import RecurrentController, SearchController
from durak_v3.evaluation import evaluate_factories
from train_v3 import load_v3
from train_v4 import load_v4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-model", type=Path, default=Path("runs/durak_v4_belief/best.pt"))
    parser.add_argument("--v3-model", type=Path, default=Path("durak_model_v3.pt"))
    parser.add_argument("--games-per-seat", type=int, default=20)
    parser.add_argument("--search-ms", type=int, default=75)
    parser.add_argument("--search-simulations", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("runs/durak_v4_belief/search_comparison.json"))
    args = parser.parse_args()

    v4, _ = load_v4(args.v4_model)
    v3, _ = load_v3(args.v3_model)
    opponent = lambda: RecurrentController(v3, deterministic=True)
    contenders = {
        "raw_v4": lambda: RecurrentController(v4, deterministic=True),
        "uniform_search_v3": lambda: SearchController(
            v3, args.search_simulations, args.search_ms,
        ),
        "belief_search_v4": lambda: SearchController(
            v4, args.search_simulations, args.search_ms,
        ),
    }
    result = {
        "games_per_seat": args.games_per_seat,
        "search_ms": args.search_ms,
        "search_simulations": args.search_simulations,
        "against": "v3",
        "results": {
            name: evaluate_factories(factory, opponent, args.games_per_seat, 121_000)
            for name, factory in contenders.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
