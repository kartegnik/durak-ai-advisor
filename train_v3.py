#!/usr/bin/env python3
"""Train a separate recurrent PPO Durak policy with a population league."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from durakgame import GameResult
from durak_v3.agents import (
    FirstController, HeuristicController, RandomController,
    GuardedV2Controller, RecordingTeacherController, RecurrentController,
    V2Controller, frozen_model,
)
from durak_v3.environment import play_episode, result_for_seat, shuffled_deck
from durak_v3.evaluation import evaluate_factories
from durak_v3.model import RecurrentActorCritic
from durak_v3.ppo import behavior_clone, ppo_update, prepare_trajectory


TRAINING_VERSION = "recurrent-ppo-league-v3"


def reward_for(result: GameResult) -> float:
    return 1.0 if result == GameResult.Win else -1.0 if result == GameResult.Loss else 0.0


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def model_payload(model, optimizer, episode, update, best_metrics, args) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "update": update,
        "best_metrics": best_metrics,
        "training_version": TRAINING_VERSION,
        "model_config": {"hidden_size": args.hidden_size, "option_size": args.option_size},
    }


def load_v3(path: Path) -> tuple[RecurrentActorCritic, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    model = RecurrentActorCritic(**config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@dataclass
class LeagueEntry:
    name: str
    model: RecurrentActorCritic
    games: int = 0
    points: float = 0.0

    @property
    def learner_rate(self) -> float:
        return self.points / self.games if self.games else 0.5


class League:
    def __init__(self, limit: int = 8):
        self.limit = limit
        self.entries: list[LeagueEntry] = []

    def add(self, name: str, model: RecurrentActorCritic) -> None:
        self.entries.append(LeagueEntry(name, frozen_model(model)))
        self.entries = self.entries[-self.limit:]

    def sample(self) -> LeagueEntry:
        # Prioritized fictitious self-play: opponents close to 50% are most useful.
        weights = [1.1 - min(1.0, abs(entry.learner_rate - 0.5) * 2.0) for entry in self.entries]
        return random.choices(self.entries, weights=weights, k=1)[0]


def collect_teacher_sequences(
    games: int, v2_path: Path, seed: int,
) -> list[list[tuple[torch.Tensor, torch.Tensor, int]]]:
    sequences = []
    v2_model = V2Controller.from_checkpoint(v2_path).model
    for index in range(games):
        teacher = GuardedV2Controller(v2_model)
        recorder = RecordingTeacherController(teacher)
        opponent = (
            RandomController() if index % 3 == 0
            else HeuristicController() if index % 3 == 1
            else V2Controller(v2_model)
        )
        seat = 1 if index % 2 else 2
        play_episode(
            recorder if seat == 1 else opponent,
            opponent if seat == 1 else recorder,
            shuffled_deck(seed + index),
        )
        if recorder.examples:
            sequences.append(recorder.examples)
    return sequences


def evaluate_suite(model, v2_model, games_per_seat: int) -> dict[str, object]:
    learner = lambda: RecurrentController(model, deterministic=True)
    opponents = {
        "random": (RandomController, 81_000),
        "heuristic": (HeuristicController, 82_000),
        "first": (FirstController, 83_000),
        "v2": (lambda: V2Controller(v2_model), 84_000),
    }
    results = {
        name: evaluate_factories(learner, factory, games_per_seat, seed)
        for name, (factory, seed) in opponents.items()
    }
    composite = (
        0.10 * results["random"]["points"]
        + 0.20 * results["heuristic"]["points"]
        + 0.10 * results["first"]["points"]
        + 0.60 * results["v2"]["points"]
    )
    return {"composite": composite, **results}


def short_metrics(metrics: dict[str, object]) -> str:
    return " ".join(
        f"{name}={metrics[name]['points']:.1%}"
        for name in ("random", "heuristic", "first", "v2")
    ) + f" composite={metrics['composite']:.1%}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/durak_v3"))
    parser.add_argument("--v2-model", type=Path, default=Path("durak_model_v2.pt"))
    parser.add_argument("--episodes", type=int, default=120_000)
    parser.add_argument("--batch-games", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--tactical-coefficient", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--option-size", type=int, default=64)
    parser.add_argument("--imitation-games", type=int, default=2_000)
    parser.add_argument("--imitation-epochs", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=25,
                        help="PPO updates between evaluations")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="paired decks per seat/opponent")
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.run_dir / "latest.pt"
    best_path = args.run_dir / "best.pt"
    metrics_path = args.run_dir / "metrics.jsonl"
    log_path = args.run_dir / "train_events.jsonl"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    v2_controller = V2Controller.from_checkpoint(args.v2_model)
    v2_model = v2_controller.model
    model = RecurrentActorCritic(args.hidden_size, args.option_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    episode = 0
    update = 0
    best_metrics: dict[str, object] = {}

    if latest_path.exists() and not args.fresh:
        loaded, checkpoint = load_v3(latest_path)
        model = loaded
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint["optimizer"])
        episode = int(checkpoint.get("episode", 0))
        update = int(checkpoint.get("update", 0))
        best_metrics = checkpoint.get("best_metrics", {})
        print(f"Resumed {latest_path} episode={episode} update={update}", flush=True)
    else:
        print(f"Collecting {args.imitation_games} honest teacher games...", flush=True)
        sequences = collect_teacher_sequences(args.imitation_games, args.v2_model, args.seed + 10_000)
        imitation_loss = behavior_clone(
            model, optimizer, sequences, epochs=args.imitation_epochs,
        )
        print(f"Imitation complete: sequences={len(sequences)} loss={imitation_loss:.4f}", flush=True)

    league = League()
    league.add(f"snapshot_{update}", model)
    model.train()
    stopped = [False]

    def stop_handler(signum, _frame):
        print(f"Signal {signum}: stopping safely after the current batch", flush=True)
        stopped[0] = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    started = time.time()

    if not best_metrics:
        model.eval()
        best_metrics = evaluate_suite(model, v2_model, args.eval_games)
        print("Initial: " + short_metrics(best_metrics), flush=True)
        atomic_save(model_payload(model, optimizer, episode, update, best_metrics, args), best_path)
        model.train()

    while episode < args.episodes and not stopped[0]:
        trajectories = []
        batch_points = 0.0
        league_records: list[tuple[LeagueEntry, float]] = []
        for _ in range(min(args.batch_games, args.episodes - episode)):
            seat = 1 if episode % 2 == 0 else 2
            learner = RecurrentController(model, deterministic=False, record=True)
            draw = random.random()
            entry = None
            if draw < 0.10:
                opponent = RandomController()
                opponent_name = "random"
            elif draw < 0.30:
                opponent = HeuristicController()
                opponent_name = "heuristic"
            elif draw < 0.60:
                opponent = V2Controller(v2_model)
                opponent_name = "v2"
            else:
                entry = league.sample()
                opponent = RecurrentController(entry.model, deterministic=False, temperature=0.85)
                opponent_name = entry.name

            game_seed = args.seed + 1_000_000 + episode
            outcome = play_episode(
                learner if seat == 1 else opponent,
                opponent if seat == 1 else learner,
                shuffled_deck(game_seed),
            )
            result = result_for_seat(outcome.result, seat)
            reward = reward_for(result)
            if learner.trajectory:
                trajectories.append(prepare_trajectory(learner.trajectory, reward))
            points = (reward + 1.0) / 2.0
            batch_points += points
            if entry is not None:
                league_records.append((entry, points))
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "episode": episode + 1, "opponent": opponent_name,
                    "seat": seat, "result": result.name,
                    "early_trumps": outcome.trump_plays_with_deck[seat - 1],
                    "avoidable_trump_defenses": outcome.avoidable_trump_defenses[seat - 1],
                }) + "\n")
            episode += 1

        for entry, points in league_records:
            entry.games += 1
            entry.points += points
        losses = ppo_update(
            model, optimizer, trajectories, epochs=args.ppo_epochs,
            tactical_coefficient=args.tactical_coefficient,
        )
        update += 1
        print(
            f"update={update:>5d} episode={episode:>7d} "
            f"points={batch_points / len(trajectories):.1%} "
            f"loss={losses['loss']:+.4f} value={losses['value']:.4f} "
            f"tactical={losses['tactical']:.4f} entropy={losses['entropy']:.3f} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

        if update % args.snapshot_every == 0:
            league.add(f"snapshot_{update}", model)
        if update % args.eval_every == 0:
            model.eval()
            metrics = evaluate_suite(model, v2_model, args.eval_games)
            record = {"episode": episode, "update": update, "elapsed": time.time() - started, **metrics}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print("EVAL " + short_metrics(metrics), flush=True)
            atomic_save(
                model_payload(model, optimizer, episode, update, metrics, args),
                args.run_dir / "checkpoints" / f"update_{update:05d}.pt",
            )
            improves = metrics["composite"] > best_metrics["composite"] + 0.002
            random_safe = metrics["random"]["points"] >= max(0.88, best_metrics["random"]["points"] - 0.04)
            strategy_safe = (
                metrics["random"]["early_trumps_per_game"]
                <= best_metrics["random"]["early_trumps_per_game"] + 0.05
                and metrics["random"]["avoidable_trump_defenses_per_game"]
                <= best_metrics["random"]["avoidable_trump_defenses_per_game"] + 0.05
            )
            if improves and random_safe and strategy_safe:
                best_metrics = metrics
                atomic_save(model_payload(model, optimizer, episode, update, best_metrics, args), best_path)
                league.add(f"champion_{update}", model)
                print("PROMOTED best.pt", flush=True)
            model.train()

        atomic_save(model_payload(model, optimizer, episode, update, best_metrics, args), latest_path)

    atomic_save(model_payload(model, optimizer, episode, update, best_metrics, args), latest_path)
    print(f"Stopped episode={episode} update={update}; best {short_metrics(best_metrics)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
