#!/usr/bin/env python3
"""Fine-tune a separate belief-aware v4 policy from the deployed v3 model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import torch

from durakgame import GameResult
from durak_v3.agents import (
    HeuristicController, RandomController, RecurrentController, V2Controller,
    frozen_model,
)
from durak_v3.environment import play_episode, result_for_seat, shuffled_deck
from durak_v3.evaluation import evaluate_factories
from durak_v3.ppo import ppo_update, prepare_trajectory, pretrain_belief
from durak_v4.belief import belief_cross_entropy, hidden_hand_recall
from durak_v4.model import BeliefActorCritic
from train_v3 import load_v3


TRAINING_VERSION = "belief-recurrent-ppo-league-v4"


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
        "base_v3": str(args.base_v3),
        "model_config": {"hidden_size": args.hidden_size, "option_size": args.option_size},
    }


def load_v4(path: Path) -> tuple[BeliefActorCritic, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BeliefActorCritic(**checkpoint.get("model_config", {}))
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        name for name in model.state_dict()
        if name.startswith("belief_policy_adapter.")
    }
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"incompatible v4 checkpoint: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, checkpoint


class LeagueEntry:
    def __init__(self, name: str, model: BeliefActorCritic):
        self.name = name
        self.model = frozen_model(model)
        self.games = 0
        self.points = 0.0

    @property
    def learner_rate(self) -> float:
        return self.points / self.games if self.games else 0.5


class League:
    def __init__(self, limit: int = 8):
        self.limit = limit
        self.entries: list[LeagueEntry] = []

    def add(self, name: str, model: BeliefActorCritic) -> None:
        self.entries.append(LeagueEntry(name, model))
        self.entries = self.entries[-self.limit:]

    def sample(self) -> LeagueEntry:
        weights = [
            1.1 - min(1.0, abs(entry.learner_rate - 0.5) * 2.0)
            for entry in self.entries
        ]
        return random.choices(self.entries, weights=weights, k=1)[0]


def collect_belief_sequences(
    model: BeliefActorCritic,
    games: int,
    v1_model,
    v2_model,
    v3_model,
    seed: int,
) -> list[list]:
    sequences = []
    opponents = (
        RandomController,
        HeuristicController,
        lambda: V2Controller(v1_model),
        lambda: V2Controller(v2_model),
        lambda: RecurrentController(v3_model, deterministic=True),
    )
    model.eval()
    for index in range(games):
        learner = RecurrentController(model, deterministic=True, record=True)
        opponent = opponents[index % len(opponents)]()
        seat = 1 if index % 2 == 0 else 2
        play_episode(
            learner if seat == 1 else opponent,
            opponent if seat == 1 else learner,
            shuffled_deck(seed + index),
        )
        if learner.trajectory:
            sequences.append(learner.trajectory)
    return sequences


def evaluate_belief(model: BeliefActorCritic, sequences: list[list]) -> dict[str, float]:
    loss_sum = 0.0
    loss_positions = 0
    recall_sum = 0.0
    baseline_sum = 0.0
    recall_positions = 0
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            hidden = model.initial_belief_hidden()
            for step in sequence:
                logits, hidden = model.belief_step(step.observation, hidden)
                loss_sum += float(belief_cross_entropy(
                    logits, step.observation, step.privileged,
                ).item())
                loss_positions += 1
                recall, baseline, positions = hidden_hand_recall(
                    logits, step.observation, step.privileged,
                )
                recall_sum += recall
                baseline_sum += baseline
                recall_positions += positions
    return {
        "cross_entropy": loss_sum / max(1, loss_positions),
        "opponent_topk_recall": recall_sum / max(1, recall_positions),
        "uniform_topk_recall": baseline_sum / max(1, recall_positions),
        "positions": float(loss_positions),
    }


def evaluate_suite(
    model, v1_model, v2_model, v3_model, games_per_seat: int,
) -> dict[str, object]:
    learner = lambda: RecurrentController(model, deterministic=True)
    opponents = {
        "random": (RandomController, 101_000),
        "heuristic": (HeuristicController, 102_000),
        "v1": (lambda: V2Controller(v1_model), 102_500),
        "v2": (lambda: V2Controller(v2_model), 103_000),
        "v3": (lambda: RecurrentController(v3_model, deterministic=True), 104_000),
    }
    results = {
        name: evaluate_factories(learner, factory, games_per_seat, seed)
        for name, (factory, seed) in opponents.items()
    }
    composite = (
        0.05 * results["random"]["points"]
        + 0.35 * results["heuristic"]["points"]
        + 0.20 * results["v1"]["points"]
        + 0.20 * results["v2"]["points"]
        + 0.20 * results["v3"]["points"]
    )
    return {"composite": composite, **results}


def short_metrics(metrics: dict[str, object]) -> str:
    return " ".join(
        f"{name}={metrics[name]['points']:.1%}"
        for name in ("random", "heuristic", "v1", "v2", "v3")
        if name in metrics
    ) + f" composite={metrics['composite']:.1%}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/durak_v4_belief"))
    parser.add_argument("--base-v3", type=Path, default=Path("durak_model_v3.pt"))
    parser.add_argument("--base-belief", type=Path,
                        help="optional pretrained v4 belief checkpoint")
    parser.add_argument("--v1-model", type=Path, default=Path("durak_model.pt"))
    parser.add_argument("--v2-model", type=Path, default=Path("durak_model_v2.pt"))
    parser.add_argument("--episodes", type=int, default=60_000)
    parser.add_argument("--batch-games", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--belief-coefficient", type=float, default=0.20)
    parser.add_argument("--tactical-coefficient", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--option-size", type=int, default=64)
    parser.add_argument("--belief-games", type=int, default=3_000)
    parser.add_argument("--belief-epochs", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--belief-eval-games", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260828)
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

    v1_model = V2Controller.from_checkpoint(args.v1_model).model
    v2_model = V2Controller.from_checkpoint(args.v2_model).model
    v3_model, _ = load_v3(args.base_v3)
    episode = 0
    update = 0
    best_metrics: dict[str, object] = {}

    if latest_path.exists() and not args.fresh:
        model, checkpoint = load_v4(latest_path)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint["optimizer"])
        episode = int(checkpoint.get("episode", 0))
        update = int(checkpoint.get("update", 0))
        best_metrics = checkpoint.get("best_metrics", {})
        print(f"Resumed {latest_path} episode={episode} update={update}", flush=True)
    else:
        if args.base_belief:
            model, _ = load_v4(args.base_belief)
            print(f"Loaded pretrained belief memory from {args.base_belief}", flush=True)
        else:
            model = BeliefActorCritic(args.hidden_size, args.option_size)
            model.load_v3_state_dict(v3_model.state_dict())
        if args.belief_games > 0:
            sequences = collect_belief_sequences(
                model, args.belief_games, v1_model, v2_model, v3_model,
                args.seed + 10_000,
            )
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("belief_"))
            head_optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=2e-4, weight_decay=1e-5,
            )
            print(f"Pretraining belief head on {len(sequences)} generated games...", flush=True)
            belief_loss = pretrain_belief(
                model, head_optimizer, sequences, epochs=args.belief_epochs,
            )
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            print(f"Belief pretraining complete: loss={belief_loss:.4f}", flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    validation_sequences = collect_belief_sequences(
        model, args.belief_eval_games, v1_model, v2_model, v3_model,
        args.seed + 900_000,
    )
    belief_metrics = evaluate_belief(model, validation_sequences)
    print(
        "BELIEF "
        f"loss={belief_metrics['cross_entropy']:.4f} "
        f"top-k={belief_metrics['opponent_topk_recall']:.1%} "
        f"uniform={belief_metrics['uniform_topk_recall']:.1%}",
        flush=True,
    )

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
        best_metrics = evaluate_suite(
            model, v1_model, v2_model, v3_model, args.eval_games,
        )
        best_metrics["belief"] = belief_metrics
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
            if draw < 0.08:
                opponent = RandomController()
                opponent_name = "random"
            elif draw < 0.36:
                opponent = HeuristicController()
                opponent_name = "heuristic"
            elif draw < 0.51:
                opponent = V2Controller(v1_model)
                opponent_name = "v1"
            elif draw < 0.65:
                opponent = V2Controller(v2_model)
                opponent_name = "v2"
            elif draw < 0.85:
                opponent = RecurrentController(v3_model, deterministic=False, temperature=0.85)
                opponent_name = "v3"
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
            belief_coefficient=args.belief_coefficient,
        )
        update += 1
        print(
            f"update={update:>5d} episode={episode:>7d} "
            f"points={batch_points / len(trajectories):.1%} "
            f"loss={losses['loss']:+.4f} value={losses['value']:.4f} "
            f"belief={losses['belief']:.4f} entropy={losses['entropy']:.3f} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

        if update % args.snapshot_every == 0:
            league.add(f"snapshot_{update}", model)
        if update % args.eval_every == 0:
            model.eval()
            metrics = evaluate_suite(
                model, v1_model, v2_model, v3_model, args.eval_games,
            )
            validation_sequences = collect_belief_sequences(
                model, args.belief_eval_games, v1_model, v2_model, v3_model,
                args.seed + 900_000,
            )
            metrics["belief"] = evaluate_belief(model, validation_sequences)
            record = {
                "episode": episode, "update": update,
                "elapsed": time.time() - started, **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                "EVAL " + short_metrics(metrics)
                + f" belief_top-k={metrics['belief']['opponent_topk_recall']:.1%}",
                flush=True,
            )
            atomic_save(
                model_payload(model, optimizer, episode, update, metrics, args),
                args.run_dir / "checkpoints" / f"update_{update:05d}.pt",
            )
            improves = metrics["composite"] > best_metrics["composite"] + 0.002
            random_safe = metrics["random"]["points"] >= max(
                0.88, best_metrics["random"]["points"] - 0.04,
            )
            if improves and random_safe:
                best_metrics = copy.deepcopy(metrics)
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
