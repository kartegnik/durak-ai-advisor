#!/usr/bin/env python3
"""Safe league-style policy training for the Durak move advisor.

The existing ``durak_model.pt`` is treated as a read-only parent.  All new
checkpoints and logs live in a separate run directory, and ``best.pt`` is only
updated after a fixed evaluation suite improves.
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
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from durakgame import (
    DeckShuffle,
    GameConfiguration,
    GameResult,
    MrFirst,
    MrRandom,
    play,
    standardDeck,
)
from model import DurakNet
from player import NNPlayer
from state import encode_combined


MOVE_LIMIT = 150
GAME_CONFIG = GameConfiguration(allowTransfer=True)
TRAINING_VERSION = "league-policy-v2"


class SamplingNNPlayer(NNPlayer):
    """Learner that samples actions and retains its decision inputs."""

    def __init__(self, model: DurakNet, temperature: float, epsilon: float):
        super().__init__(model)
        self.temperature = temperature
        self.epsilon = epsilon
        self.trajectory: list[tuple[torch.Tensor, int]] = []

    def nextMove(self, state, options):
        combined = encode_combined(self.hand, state, options)
        with torch.no_grad():
            scores = self.model(combined)
            if random.random() < self.epsilon:
                chosen = random.randrange(len(options))
            else:
                # The parent was trained as a greedy action scorer, not as
                # calibrated softmax logits.  Epsilon-greedy therefore keeps
                # its strong existing behaviour while still exploring.
                chosen = int(scores.argmax().item())
        self.trajectory.append((combined, chosen))
        return chosen


@dataclass
class RunningBaseline:
    values: dict[str, float]
    momentum: float = 0.97

    def advantage(self, key: str, reward: float) -> float:
        old = self.values.get(key, 0.0)
        self.values[key] = self.momentum * old + (1.0 - self.momentum) * reward
        return reward - old


def load_model(path: Path) -> tuple[DurakNet, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
    model = DurakNet()
    model.load_compatible_state_dict(state_dict)
    model.eval()
    episode = int(checkpoint.get("episode", 0)) if isinstance(checkpoint, dict) else 0
    return model, episode


def frozen_copy(model: DurakNet) -> DurakNet:
    result = copy.deepcopy(model)
    result.eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(
    path: Path,
    model: DurakNet,
    optimizer: torch.optim.Optimizer,
    episode: int,
    parent_path: Path,
    parent_episode: int,
    best_metrics: dict[str, float],
    baselines: RunningBaseline,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": episode,
            "parent_checkpoint": str(parent_path),
            "parent_episode": parent_episode,
            "training_version": TRAINING_VERSION,
            "best_metrics": best_metrics,
            "reward_baselines": baselines.values,
        },
        path,
    )


def result_for_seat(result: GameResult, seat: int) -> GameResult:
    if seat == 1:
        return result
    if result == GameResult.Win:
        return GameResult.Loss
    if result == GameResult.Loss:
        return GameResult.Win
    return result


def reward_for_result(result: GameResult) -> float:
    if result == GameResult.Win:
        return 1.0
    if result == GameResult.Loss:
        return -1.0
    return 0.0


def make_opponent(kind: str, parent: DurakNet, league: DurakNet):
    if kind == "random":
        return MrRandom()
    if kind == "first":
        return MrFirst()
    if kind == "parent":
        return NNPlayer(parent)
    if kind == "league":
        return NNPlayer(league)
    raise ValueError(f"Unknown opponent kind: {kind}")


def choose_opponent() -> str:
    # Fixed baselines keep basic play from being forgotten; the frozen parent
    # and best league snapshot make the curriculum progressively harder.
    value = random.random()
    if value < 0.30:
        return "random"
    if value < 0.50:
        return "first"
    if value < 0.75:
        return "parent"
    return "league"


def train_game(
    model: DurakNet,
    optimizer: torch.optim.Optimizer,
    parent: DurakNet,
    league: DurakNet,
    baselines: RunningBaseline,
    episode: int,
    total_episodes: int,
    anchor_strength: float,
) -> tuple[GameResult, float, str, int]:
    progress = episode / max(total_episodes, 1)
    temperature = max(0.012, 0.025 - 0.013 * progress)
    epsilon = max(0.02, 0.10 * (1.0 - progress))
    opponent_kind = choose_opponent()
    seat = 1 if episode % 2 else 2

    learner = SamplingNNPlayer(model, temperature=temperature, epsilon=epsilon)
    opponent = make_opponent(opponent_kind, parent, league)
    if seat == 1:
        history = play(learner, opponent, moveLimit=MOVE_LIMIT, gameConfig=GAME_CONFIG)
    else:
        history = play(opponent, learner, moveLimit=MOVE_LIMIT, gameConfig=GAME_CONFIG)

    result = result_for_seat(history.result, seat)
    reward = reward_for_result(result)
    key = f"{opponent_kind}:seat{seat}"
    advantage = baselines.advantage(key, reward)

    if not learner.trajectory:
        return result, 0.0, opponent_kind, seat

    policy_terms = []
    entropy_terms = []
    anchor_terms = []
    for combined, chosen in learner.trajectory:
        scores = model(combined)
        log_probs = F.log_softmax(scores / temperature, dim=0)
        probs = log_probs.exp()
        policy_terms.append(-advantage * log_probs[chosen])
        entropy_terms.append(-(probs * log_probs).sum())
        with torch.no_grad():
            parent_scores = parent(combined)
        anchor_terms.append(F.smooth_l1_loss(scores, parent_scores))

    policy_loss = torch.stack(policy_terms).mean()
    entropy = torch.stack(entropy_terms).mean()
    anchor_loss = torch.stack(anchor_terms).mean()
    loss = policy_loss - 0.003 * entropy + anchor_strength * anchor_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return result, float(loss.detach()), opponent_kind, seat


def _new_eval_opponent(kind: str, opponent_model: DurakNet | None):
    if kind == "random":
        return MrRandom()
    if kind == "first":
        return MrFirst()
    if kind == "model" and opponent_model is not None:
        return NNPlayer(opponent_model)
    raise ValueError(kind)


def evaluate_match(
    model: DurakNet,
    kind: str,
    opponent_model: DurakNet | None,
    games_per_seat: int,
    seed: int,
) -> dict[str, float]:
    counts = {"wins": 0, "draws": 0, "losses": 0}
    saved_random_state = random.getstate()
    saved_torch_state = torch.random.get_rng_state()
    try:
        with torch.no_grad():
            for index in range(games_per_seat):
                game_seed = seed + index
                deck = standardDeck()
                random.Random(game_seed).shuffle(deck)
                for seat in (1, 2):
                    random.seed(game_seed)
                    torch.manual_seed(game_seed)
                    learner = NNPlayer(model)
                    opponent = _new_eval_opponent(kind, opponent_model)
                    if seat == 1:
                        history = play(
                            learner, opponent, shuffleType=DeckShuffle.custom,
                            customDeck=list(deck), moveLimit=MOVE_LIMIT,
                            gameConfig=GAME_CONFIG,
                        )
                    else:
                        history = play(
                            opponent, learner, shuffleType=DeckShuffle.custom,
                            customDeck=list(deck), moveLimit=MOVE_LIMIT,
                            gameConfig=GAME_CONFIG,
                        )
                    result = result_for_seat(history.result, seat)
                    if result == GameResult.Win:
                        counts["wins"] += 1
                    elif result == GameResult.Loss:
                        counts["losses"] += 1
                    else:
                        counts["draws"] += 1
    finally:
        random.setstate(saved_random_state)
        torch.random.set_rng_state(saved_torch_state)

    total = 2 * games_per_seat
    return {
        "win_rate": counts["wins"] / total,
        "points": (counts["wins"] + 0.5 * counts["draws"]) / total,
        **{key: float(value) for key, value in counts.items()},
    }


def evaluate_suite(
    model: DurakNet,
    parent: DurakNet,
    games_per_seat: int,
) -> dict[str, float]:
    model.eval()
    random_result = evaluate_match(model, "random", None, games_per_seat, 11_000)
    first_result = evaluate_match(model, "first", None, games_per_seat, 22_000)
    parent_result = evaluate_match(model, "model", parent, games_per_seat, 33_000)
    composite = (
        0.25 * random_result["points"]
        + 0.25 * first_result["points"]
        + 0.50 * parent_result["points"]
    )
    return {
        "random": random_result["points"],
        "first": first_result["points"],
        "parent": parent_result["points"],
        "composite": composite,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(
        f"{name}={metrics[name]:.1%}"
        for name in ("random", "first", "parent", "composite")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("durak_model.pt"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/durak_v2"))
    parser.add_argument("--episodes", type=int, default=30_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--anchor-strength", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=40,
                        help="games per seat and per opponent during evaluation")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore latest.pt and restart from the parent")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.parent.exists():
        raise FileNotFoundError(args.parent)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.run_dir / "latest.pt"
    best_path = args.run_dir / "best.pt"
    parent_copy_path = args.run_dir / "parent.pt"
    metrics_path = args.run_dir / "metrics.jsonl"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    parent_model, parent_episode = load_model(args.parent)
    parent_model = frozen_copy(parent_model)

    model = copy.deepcopy(parent_model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5,
    )
    baselines = RunningBaseline({})
    start_episode = 0
    best_metrics: dict[str, float]

    if latest_path.exists() and not args.fresh:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        model.load_compatible_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = int(checkpoint.get("episode", 0))
        best_metrics = dict(checkpoint.get("best_metrics", {}))
        baselines.values.update(checkpoint.get("reward_baselines", {}))
        print(f"Resumed {latest_path} at episode {start_episode}", flush=True)
    else:
        parent_payload = {
            "model": parent_model.state_dict(),
            "episode": parent_episode,
            "source": str(args.parent.resolve()),
            "training_version": "frozen-parent",
        }
        atomic_torch_save(parent_payload, parent_copy_path)
        best_metrics = evaluate_suite(model, parent_model, args.eval_games)
        print(f"Parent baseline: {format_metrics(best_metrics)}", flush=True)
        initial_payload = {
            "model": model.state_dict(),
            "episode": 0,
            "parent_checkpoint": str(args.parent.resolve()),
            "parent_episode": parent_episode,
            "training_version": TRAINING_VERSION,
            "metrics": best_metrics,
        }
        atomic_torch_save(initial_payload, best_path)

    if not best_metrics:
        best_metrics = evaluate_suite(model, parent_model, args.eval_games)
    league_model, _ = load_model(best_path)
    league_model = frozen_copy(league_model)

    stopped = [False]

    def handle_signal(signum, _frame):
        print(f"Signal {signum}: will save after the current game", flush=True)
        stopped[0] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    total_episode = start_episode + args.episodes
    started = time.time()
    recent_results: list[float] = []
    current_episode = start_episode
    print(
        f"Training {TRAINING_VERSION}: episodes {start_episode + 1}..{total_episode}; "
        f"parent={args.parent}; output={args.run_dir}",
        flush=True,
    )

    for current_episode in range(start_episode + 1, total_episode + 1):
        result, loss, opponent_kind, seat = train_game(
            model=model,
            optimizer=optimizer,
            parent=parent_model,
            league=league_model,
            baselines=baselines,
            episode=current_episode,
            total_episodes=total_episode,
            anchor_strength=args.anchor_strength,
        )
        recent_results.append(reward_for_result(result))
        if len(recent_results) > 100:
            recent_results.pop(0)

        if current_episode % 100 == 0:
            recent_points = sum((value + 1.0) / 2.0 for value in recent_results) / len(recent_results)
            print(
                f"ep={current_episode:>6d} result={result.name:<4s} "
                f"op={opponent_kind:<6s} seat={seat} loss={loss:+.4f} "
                f"recent_points={recent_points:.1%} elapsed={time.time() - started:.0f}s",
                flush=True,
            )

        if current_episode % args.eval_every == 0:
            metrics = evaluate_suite(model, parent_model, args.eval_games)
            record = {"episode": current_episode, "elapsed": time.time() - started, **metrics}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"EVAL ep={current_episode}: {format_metrics(metrics)}", flush=True)

            # Require a real composite gain and guard the two simple baselines
            # against a large regression before promoting the league snapshot.
            improves = metrics["composite"] > best_metrics["composite"] + 0.002
            safe_baselines = (
                metrics["random"] >= best_metrics["random"] - 0.075
                and metrics["first"] >= best_metrics["first"] - 0.075
            )
            if improves and safe_baselines:
                best_metrics = metrics
                best_payload = {
                    "model": model.state_dict(),
                    "episode": current_episode,
                    "parent_checkpoint": str(args.parent.resolve()),
                    "parent_episode": parent_episode,
                    "training_version": TRAINING_VERSION,
                    "metrics": best_metrics,
                }
                atomic_torch_save(best_payload, best_path)
                league_model = frozen_copy(model)
                print(f"PROMOTED best.pt at episode {current_episode}", flush=True)

        if current_episode % args.save_every == 0 or stopped[0]:
            save_training_checkpoint(
                latest_path, model, optimizer, current_episode,
                args.parent.resolve(), parent_episode, best_metrics, baselines,
            )
        if stopped[0]:
            print(f"Stopped safely at episode {current_episode}", flush=True)
            return 0

    save_training_checkpoint(
        latest_path, model, optimizer, current_episode,
        args.parent.resolve(), parent_episode, best_metrics, baselines,
    )
    print(f"Done at episode {current_episode}; best: {format_metrics(best_metrics)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
