#!/usr/bin/env python3
"""Large paired evaluation for a v3 checkpoint."""

import argparse
import json
from pathlib import Path

from durak_v3.agents import HeuristicController, RandomController, RecurrentController, V2Controller
from durak_v3.evaluation import evaluate_factories
from train_v3 import load_v3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=Path("runs/durak_v3/best.pt"))
    parser.add_argument("--v2-model", type=Path, default=Path("durak_model_v2.pt"))
    parser.add_argument("--games-per-seat", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("runs/durak_v3/independent_evaluation.json"))
    args = parser.parse_args()
    model, checkpoint = load_v3(args.checkpoint)
    v2 = V2Controller.from_checkpoint(args.v2_model).model
    learner = lambda: RecurrentController(model, deterministic=True)
    opponents = {
        "random": (RandomController, 91_000),
        "heuristic": (HeuristicController, 92_000),
        "v2": (lambda: V2Controller(v2), 93_000),
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "episode": checkpoint.get("episode", 0),
        "paired_decks_per_seat": args.games_per_seat,
        "opponents": {
            name: evaluate_factories(learner, factory, args.games_per_seat, seed)
            for name, (factory, seed) in opponents.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
