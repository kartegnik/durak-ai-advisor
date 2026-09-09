#!/usr/bin/env python3
"""Paired evaluation of the stateful Habr-derived agent."""

import argparse
import json

from durak_habr import HabrMemoryController
from durak_v3.agents import HeuristicController, RandomController, RecurrentController, V2Controller
from durak_v3.evaluation import evaluate_factories
from train_v3 import load_v3
from train_v4 import load_v4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seat", type=int, default=1000)
    args = parser.parse_args()
    v1 = V2Controller.from_checkpoint("durak_model.pt").model
    v2 = V2Controller.from_checkpoint("durak_model_v2.pt").model
    v3, _ = load_v3("durak_model_v3.pt")
    v4, _ = load_v4("durak_model_v4.pt")
    opponents = {
        "random": RandomController,
        "heuristic": HeuristicController,
        "v1": lambda: V2Controller(v1),
        "v2": lambda: V2Controller(v2),
        "v3": lambda: RecurrentController(v3, deterministic=True),
        "v4": lambda: RecurrentController(v4, deterministic=True),
    }
    result = {
        name: evaluate_factories(
            HabrMemoryController, factory, args.games_per_seat, 310_000 + index * 10_000,
        )
        for index, (name, factory) in enumerate(opponents.items())
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
