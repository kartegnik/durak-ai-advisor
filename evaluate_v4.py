#!/usr/bin/env python3
"""Paired evaluation of belief-aware v4 against the deployed v3."""

import argparse
import json
from pathlib import Path

from durak_v3.agents import HeuristicController, RandomController, RecurrentController, V2Controller
from durak_v3.evaluation import evaluate_factories
from train_v3 import load_v3
from train_v4 import load_v4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=Path("runs/durak_v4_belief/best.pt"))
    parser.add_argument("--v3-model", type=Path, default=Path("durak_model_v3.pt"))
    parser.add_argument("--v1-model", type=Path, default=Path("durak_model.pt"))
    parser.add_argument("--v2-model", type=Path, default=Path("durak_model_v2.pt"))
    parser.add_argument("--games-per-seat", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("runs/durak_v4_belief/independent_evaluation.json"))
    args = parser.parse_args()

    model, checkpoint = load_v4(args.checkpoint)
    v3, _ = load_v3(args.v3_model)
    v1 = V2Controller.from_checkpoint(args.v1_model).model
    v2 = V2Controller.from_checkpoint(args.v2_model).model
    learner = lambda: RecurrentController(model, deterministic=True)
    opponents = {
        "random": (RandomController, 111_000),
        "heuristic": (HeuristicController, 112_000),
        "v1": (lambda: V2Controller(v1), 112_500),
        "v2": (lambda: V2Controller(v2), 113_000),
        "v3": (lambda: RecurrentController(v3, deterministic=True), 114_000),
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
