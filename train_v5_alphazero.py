#!/usr/bin/env python3
"""Train v5 from information-set MCTS self-play targets.

This is AlphaZero-style rather than literal AlphaZero: Durak has hidden cards,
so every search samples hands and deck orders consistent with public memory.
Released v1-v4 checkpoints are read-only inputs and are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

import torch

from durak_habr import HabrMemoryController
from durak_v3.agents import HeuristicController, RecurrentController, frozen_model
from durak_v3.environment import play_episode, shuffled_deck
from durak_v3.evaluation import evaluate_factories
from durak_v5 import SearchSelfPlayController
from durak_v5.training import alphazero_loss
from train_v4 import load_v4


TRAINING_VERSION = "information-set-alphazero-v5"


def reward(result) -> float:
    return 1.0 if result.value == 1 else -1.0 if result.value == -1 else 0.0


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def payload(model, optimizer, episode: int, update: int, metrics: dict, args) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "update": update,
        "best_metrics": metrics,
        "training_version": TRAINING_VERSION,
        "base_v4": str(args.base_v4),
        "model_config": {
            "hidden_size": model.hidden_size,
            "option_size": model.option_encoder[0].out_features,
        },
        "search_config": {
            "simulations": args.simulations,
            "time_limit_ms": args.search_ms,
            "exploration_moves": args.exploration_moves,
            "root_noise_fraction": args.root_noise,
            "dirichlet_alpha": args.dirichlet_alpha,
            "search_value_coefficient": args.search_value_coefficient,
            "quiescence": True,
            "habr_leaf_weight": 0.10,
            "reuse_tree": True,
            "exact_endgame_cards": 10,
        },
    }


def load_v5(path: Path):
    model, checkpoint = load_v4(path)
    version = checkpoint.get("training_version", "")
    if version != TRAINING_VERSION:
        raise RuntimeError(f"not a v5 checkpoint: {version!r}")
    return model, checkpoint


def evaluate_candidate(model, champion, baseline_v4, games_per_seat: int, seed: int) -> dict:
    candidate = lambda: RecurrentController(model, deterministic=True)
    return {
        "champion": evaluate_factories(
            candidate,
            lambda: RecurrentController(champion, deterministic=True),
            games_per_seat, seed,
        ),
        "v4_baseline": evaluate_factories(
            candidate,
            lambda: RecurrentController(baseline_v4, deterministic=True),
            games_per_seat, seed + 5_000,
        ),
        "habr_memory": evaluate_factories(
            candidate, HabrMemoryController, games_per_seat, seed + 10_000,
        ),
        "heuristic": evaluate_factories(
            candidate, HeuristicController, games_per_seat, seed + 20_000,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/durak_v5_alphazero"))
    parser.add_argument("--base-v4", type=Path, default=Path("durak_model_v4.pt"))
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--batch-games", type=int, default=8)
    parser.add_argument("--replay-games", type=int, default=256)
    parser.add_argument("--train-trajectories", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--search-ms", type=int, default=1500)
    parser.add_argument("--exploration-moves", type=int, default=8)
    parser.add_argument("--root-noise", type=float, default=0.08)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--belief-coefficient", type=float, default=0.1)
    parser.add_argument("--search-value-coefficient", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--promotion-score", type=float, default=0.53)
    parser.add_argument("--v4-safety-score", type=float, default=0.51)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.run_dir / "latest.pt"
    best_path = args.run_dir / "best.pt"
    metrics_path = args.run_dir / "metrics.jsonl"
    training_path = args.run_dir / "train_metrics.jsonl"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode = 0
    update = 0
    best_metrics: dict = {}
    if latest_path.exists() and not args.fresh:
        model, checkpoint = load_v5(latest_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint["optimizer"])
        episode = int(checkpoint.get("episode", 0))
        update = int(checkpoint.get("update", 0))
        best_metrics = checkpoint.get("best_metrics", {})
        if best_path.exists():
            champion, _ = load_v5(best_path)
        else:
            champion = copy.deepcopy(model)
        print(f"Resumed {latest_path} episode={episode} update={update}", flush=True)
    else:
        model, _ = load_v4(args.base_v4)
        champion = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        atomic_save(payload(champion, optimizer, 0, 0, {}, args), best_path)
        print(f"Bootstrapped v5 from {args.base_v4}; v4 remains unchanged", flush=True)

    champion = frozen_model(champion)
    baseline_v4, _ = load_v4(args.base_v4)
    baseline_v4 = frozen_model(baseline_v4)
    replay: deque[tuple[list, float]] = deque(maxlen=args.replay_games * 2)
    stopped = [False]

    def stop_handler(signum, _frame):
        print(f"Signal {signum}: stopping safely after the current batch", flush=True)
        stopped[0] = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    started = time.time()

    while episode < args.episodes and not stopped[0]:
        games_now = min(args.batch_games, args.episodes - episode)
        selfplay_points = 0.0
        search_results = []
        champion.eval()
        for _ in range(games_now):
            player1 = SearchSelfPlayController(
                champion, args.simulations, args.search_ms,
                args.exploration_moves, args.root_noise,
                args.dirichlet_alpha,
            )
            player2 = SearchSelfPlayController(
                champion, args.simulations, args.search_ms,
                args.exploration_moves, args.root_noise,
                args.dirichlet_alpha,
            )
            outcome = play_episode(
                player1, player2, shuffled_deck(args.seed + 1_000_000 + episode),
            )
            first_reward = reward(outcome.result)
            replay.append((player1.steps, first_reward))
            replay.append((player2.steps, -first_reward))
            search_results.extend(player1.search_results)
            search_results.extend(player2.search_results)
            selfplay_points += (first_reward + 1.0) / 2.0
            episode += 1

        sample_size = min(args.train_trajectories, len(replay))
        sample = random.sample(list(replay), sample_size)
        model.train()
        totals = {"policy": 0.0, "value": 0.0, "belief": 0.0, "entropy": 0.0}
        for _ in range(args.train_epochs):
            optimizer.zero_grad()
            loss, loss_metrics = alphazero_loss(
                model,
                [item[0] for item in sample],
                [item[1] for item in sample],
                args.belief_coefficient,
                args.search_value_coefficient,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name in totals:
                totals[name] += loss_metrics[name] / args.train_epochs
        update += 1
        search_calls = max(1, len(search_results))
        search_metrics = {
            "calls": len(search_results),
            "simulations_per_call": sum(item.simulations for item in search_results) / search_calls,
            "determinizations_per_call": sum(item.unique_determinizations for item in search_results) / search_calls,
            "reused_root_visits_per_call": sum(item.reused_root_visits for item in search_results) / search_calls,
            "mean_leaf_depth": sum(item.mean_leaf_depth for item in search_results) / search_calls,
            "mean_quiescence_steps": sum(item.mean_quiescence_steps for item in search_results) / search_calls,
            "exact_endgames": sum(item.solved_exactly for item in search_results),
        }
        train_record = {
            "episode": episode, "update": update,
            "selfplay_p1": selfplay_points / games_now,
            **totals, "search": search_metrics,
            "elapsed": time.time() - started,
        }
        with training_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(train_record) + "\n")
        print(
            f"update={update:>5d} episode={episode:>7d} "
            f"selfplay_p1={selfplay_points / games_now:.1%} "
            f"policy={totals['policy']:.4f} value={totals['value']:.4f} "
            f"belief={totals['belief']:.4f} elapsed={time.time() - started:.0f}s",
            flush=True,
        )

        should_evaluate = args.eval_every > 0 and update % args.eval_every == 0
        if should_evaluate:
            model.eval()
            metrics = evaluate_candidate(
                model, champion, baseline_v4, args.eval_games,
                args.seed + 2_000_000 + update * 1000,
            )
            promoted = (
                metrics["champion"]["points"] >= args.promotion_score
                and metrics["v4_baseline"]["points"] >= args.v4_safety_score
            )
            record = {
                "episode": episode, "update": update,
                "elapsed": time.time() - started,
                "promoted": promoted, **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                "EVAL " + " ".join(
                    f"{name}={value['points']:.1%}" for name, value in metrics.items()
                ) + (" PROMOTED" if promoted else ""),
                flush=True,
            )
            atomic_save(
                payload(model, optimizer, episode, update, metrics, args),
                args.run_dir / "checkpoints" / f"update_{update:05d}.pt",
            )
            if promoted:
                best_metrics = copy.deepcopy(metrics)
                champion = frozen_model(model)
                atomic_save(
                    payload(model, optimizer, episode, update, best_metrics, args), best_path,
                )

        atomic_save(payload(model, optimizer, episode, update, best_metrics, args), latest_path)

    atomic_save(payload(model, optimizer, episode, update, best_metrics, args), latest_path)
    print(f"Stopped episode={episode} update={update}; checkpoints are in {args.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
