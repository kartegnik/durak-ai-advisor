#!/usr/bin/env python3
"""Live, read-only Durak assistant: capture/watch frames and show move advice."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from capture import capture_screenshot, get_next_number
from game_state_tracker import GameStateTracker
from legal_move_advisor import legal_actions
from neural_move_advisor import NeuralMoveAdvisor
from neural_v3_advisor import NeuralV3Advisor
from neural_v4_advisor import NeuralV4Advisor
from track_screenshots import FrameRecognizer, numeric_stem


TRUMP_NAMES = {
    "H": "черви ♥",
    "D": "бубны ♦",
    "C": "трефы ♣",
    "S": "пики ♠",
}

TRUMP_ALIASES = {
    "h": "H", "ч": "H", "черви": "H", "червы": "H", "♥": "H",
    "d": "D", "б": "D", "бубны": "D", "буби": "D", "♦": "D",
    "c": "C", "т": "C", "трефы": "C", "крести": "C", "♣": "C",
    "s": "S", "п": "S", "пики": "S", "♠": "S",
}


def parse_trump(value: str) -> str:
    """Accept both compact and human-readable Russian suit names."""
    normalized = value.strip().lower()
    suit = TRUMP_ALIASES.get(normalized)
    if suit is None:
        raise argparse.ArgumentTypeError(
            "масть должна быть: черви/H, бубны/D, трефы/C или пики/S"
        )
    return suit


def prompt_for_trump() -> str | None:
    """Ask for a reliable manual trump; fall back to CV without a terminal."""
    if not sys.stdin.isatty():
        return None
    while True:
        try:
            value = input("Введите козырь (черви/бубны/трефы/пики): ")
        except (EOFError, KeyboardInterrupt):
            print("\nРучной ввод отменён — включено автоматическое распознавание козыря.")
            return None
        try:
            return parse_trump(value)
        except argparse.ArgumentTypeError as error:
            print(f"Неверная масть: {error}")


@dataclass(frozen=True)
class LiveResult:
    frame: str
    state: str
    trump: str | None
    deck_count: int | None
    actions: tuple[str, ...]
    recommendation: str | None
    ranking: tuple[tuple[str, float], ...]
    phase_note: str
    conditional: bool
    warnings: tuple[str, ...]
    changes: tuple[str, ...]
    reset: bool = False

    def display_text(self) -> str:
        header = f"Козырь: {self.trump or '?'}    Колода: {self.deck_count if self.deck_count is not None else '?'}"
        if self.recommendation and self.conditional:
            move = "ЕСЛИ АТАКУЕМ МЫ: " + self.recommendation
        elif self.recommendation:
            move = "РЕКОМЕНДАЦИЯ: " + self.recommendation
        elif self.actions:
            move = "МОЖНО: " + "; ".join(self.actions)
        elif self.phase_note:
            move = self.phase_note[0].upper() + self.phase_note[1:]
        else:
            move = "Ожидание изменения игры"
        warning = "\n⚠ " + "; ".join(self.warnings) if self.warnings else ""
        return f"{header}\n{move}\n{self.state}{warning}"


class LiveAdvisorCore:
    def __init__(
        self, recognizer: FrameRecognizer, neural: NeuralMoveAdvisor,
        log_path: Path, confidence: float = 0.25,
        trump_override: str | None = None, deck_override: int | None = None,
        role_override: str | None = None,
        state_path: Path | None = None, resume: bool = True,
        trump_prompt: Callable[[], str | None] | None = None,
    ):
        self.recognizer = recognizer
        self.neural = neural
        self.log_path = log_path
        self.confidence = confidence
        self.trump_override = trump_override
        self.deck_override = deck_override
        self.role_override = role_override
        self.trump_prompt = trump_prompt
        self.state_path = state_path
        self.resumed_state = False
        self.resume_gap_seconds = 0.0
        self.tracker = GameStateTracker(minimum_initial_hand=5)
        if role_override:
            self.tracker.player_attacker = role_override == "attack"
        if resume and state_path and state_path.exists():
            self._load_state()

    def _load_state(self) -> None:
        saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.resumed_state = True
        updated = datetime.fromisoformat(saved["updated"])
        self.resume_gap_seconds = max(
            0.0, (datetime.now().astimezone() - updated).total_seconds()
        )
        tracker = saved["tracker"]
        self.tracker.hand = set(tracker["hand"])
        self.tracker.table = set(tracker["table"])
        self.tracker.table_attack = list(tracker["table_attack"])
        self.tracker.table_defense = list(tracker["table_defense"])
        self.tracker.discard = set(tracker["discard"])
        self.tracker.opponent_known = set(tracker.get("opponent_known", ()))
        self.tracker.player_attacker = tracker["player_attacker"]
        self.tracker.initialized = tracker["initialized"]
        self.tracker.active_events = set(tracker.get("active_events", ()))
        self.tracker.draw_candidates = dict(tracker.get("draw_candidates", {}))
        self.tracker.table_candidates = dict(tracker.get("table_candidates", {}))
        self.tracker.hand_reconcile_pending = tracker.get("hand_reconcile_pending", False)
        self.tracker.hand_reconcile_age = tracker.get("hand_reconcile_age", 0)
        self.tracker.hand_snapshot_candidate = frozenset(
            tracker.get("hand_snapshot_candidate", ())
        )
        self.tracker.hand_snapshot_count = tracker.get("hand_snapshot_count", 0)
        recognizer = saved["recognizer"]
        self.recognizer.last_trump = recognizer["trump"]
        self.recognizer.trump_candidate = recognizer["trump"]
        self.recognizer.trump_candidate_count = 2 if recognizer["trump"] else 0
        self.recognizer.deck_count = recognizer["deck_count"]
        self.recognizer.deck_candidate = recognizer["deck_count"]
        self.recognizer.deck_candidate_count = 2 if recognizer["deck_count"] is not None else 0
        self.tracker.last_deck_count = tracker.get(
            "last_deck_count", recognizer["deck_count"]
        )
        self.tracker.empty_deck_grace = tracker.get("empty_deck_grace", 0)
        self.tracker.last_event_kind = tracker.get("last_event_kind")
        self.tracker.last_event_cards = tuple(tracker.get("last_event_cards", ()))
        self.tracker.last_event_actor_self = tracker.get("last_event_actor_self")

    def _save_state(self, frame: str) -> None:
        if not self.state_path:
            return
        saved = {
            "version": 1,
            "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "last_frame": frame,
            "tracker": {
                "hand": sorted(self.tracker.hand),
                "table": sorted(self.tracker.table),
                "table_attack": self.tracker.table_attack,
                "table_defense": self.tracker.table_defense,
                "discard": sorted(self.tracker.discard),
                "opponent_known": sorted(self.tracker.opponent_known),
                "player_attacker": self.tracker.player_attacker,
                "initialized": self.tracker.initialized,
                "active_events": sorted(self.tracker.active_events),
                "draw_candidates": self.tracker.draw_candidates,
                "table_candidates": self.tracker.table_candidates,
                "hand_reconcile_pending": self.tracker.hand_reconcile_pending,
                "hand_reconcile_age": self.tracker.hand_reconcile_age,
                "hand_snapshot_candidate": sorted(self.tracker.hand_snapshot_candidate),
                "hand_snapshot_count": self.tracker.hand_snapshot_count,
                "last_deck_count": self.tracker.last_deck_count,
                "empty_deck_grace": self.tracker.empty_deck_grace,
                "last_event_kind": self.tracker.last_event_kind,
                "last_event_cards": list(self.tracker.last_event_cards),
                "last_event_actor_self": self.tracker.last_event_actor_self,
            },
            "recognizer": {
                "trump": self.recognizer.last_trump,
                "deck_count": self.recognizer.deck_count,
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _new_game_detected(self) -> bool:
        return (
            self.tracker.initialized
            and self.recognizer.deck_count not in {None, 24}
            and self.recognizer.deck_candidate == 24
            and self.recognizer.deck_candidate_count >= 2
        )

    def process(self, path: Path) -> LiveResult | None:
        image = cv2.imread(str(path))
        if image is None:
            return None
        observation = self.recognizer.recognize(image, self.confidence)
        reset = self._new_game_detected()
        if reset:
            self.recognizer.reset_game()
            self.tracker = GameStateTracker(minimum_initial_hand=5)
            if hasattr(self.neural, "reset"):
                self.neural.reset()
            if self.role_override:
                self.tracker.player_attacker = self.role_override == "attack"
            if self.trump_prompt:
                self.trump_override = self.trump_prompt()
        update = self.tracker.observe(observation)

        trump = self.trump_override or self.recognizer.last_trump
        deck_count = self.deck_override if self.deck_override is not None else self.recognizer.deck_count
        advice = legal_actions(self.tracker, trump) if trump and self.tracker.initialized else None
        conditional = (
            self.tracker.player_attacker is None
            and not self.tracker.table_attack
        )
        recommendation = None
        ranking: tuple[tuple[str, float], ...] = ()
        if advice and advice.actions and deck_count is not None:
            neural_result = self.neural.recommend(
                self.tracker, advice, trump, deck_count
            )
            recommendation = neural_result.action
            ranking = neural_result.scores

        warnings = []
        uncertain = [
            f"{card.card} ({card.class_confidence:.2f}/{card.box_confidence:.2f})"
            for card in observation.cards if not card.reliable
        ]
        if uncertain:
            warnings.append("неуверенные карты: " + ", ".join(uncertain))
        if self.tracker.initialized and trump is None:
            warnings.append("козырь ещё не подтверждён")
        if self.tracker.initialized and deck_count is None:
            warnings.append("счётчик колоды ещё не подтверждён")
        if reset:
            warnings.append("обнаружена новая партия; память сброшена")
        if self.resumed_state:
            warnings.append("состояние восстановлено после перезапуска")
            if self.resume_gap_seconds > 10:
                warnings.append(
                    f"пауза {round(self.resume_gap_seconds)} с: если игра продолжалась, память неполная"
                )

        result = LiveResult(
            frame=path.name,
            state=self.tracker.summary(),
            trump=trump,
            deck_count=deck_count,
            actions=advice.actions if advice else (),
            recommendation=recommendation,
            ranking=ranking,
            phase_note=advice.note if advice else "",
            conditional=conditional,
            warnings=tuple(warnings),
            changes=tuple(update.changes),
            reset=reset,
        )
        self._log(result, observation.events)
        self._save_state(path.name)
        self.resumed_state = False
        return result

    def _log(self, result: LiveResult, events: tuple[str, ...]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "frame": result.frame,
            "state": result.state,
            "trump": result.trump,
            "deck_count": result.deck_count,
            "events": list(events),
            "changes": list(result.changes),
            "actions": list(result.actions),
            "recommendation": result.recommendation,
            "phase_note": result.phase_note,
            "conditional": result.conditional,
            "scores": [{"action": action, "score": score} for action, score in result.ranking],
            "warnings": list(result.warnings),
            "reset": result.reset,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class Overlay:
    def __init__(self):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("Durak — подсказка")
        self.root.attributes("-topmost", True)
        self.root.geometry("570x235+20+120")
        self.root.configure(bg="#15202b")
        self.text = tk.StringVar(value="Ожидаю новый скриншот…")
        tk.Label(
            self.root, textvariable=self.text, justify="left", anchor="nw",
            wraplength=540, bg="#15202b", fg="#f5f5f5",
            font=("DejaVu Sans", 12), padx=14, pady=12,
        ).pack(fill="both", expand=True)

    def update(self, result: LiveResult):
        self.text.set(f"{result.frame}\n{result.display_text()}")


def build_core(args) -> LiveAdvisorCore:
    project = Path(__file__).resolve().parents[2]
    recognizer = FrameRecognizer(
        project, args.localizer, args.classifier, args.zone_label
    )
    if args.use_v2:
        neural = NeuralMoveAdvisor(args.nn_model)
    elif args.use_v4:
        neural = NeuralV4Advisor(
            args.v4_model, args.search_ms, args.search_simulations,
        )
    else:
        neural = NeuralV3Advisor(
            args.v3_model, args.search_ms, args.search_simulations,
        )
    return LiveAdvisorCore(
        recognizer, neural, args.log,
        args.confidence, args.trump, args.deck_count,
        args.role,
        args.state, not args.new_game,
        args.trump_prompt,
    )


def print_result(result: LiveResult) -> None:
    print(f"\n{result.frame}")
    for change in result.changes:
        print("  " + change)
    print("  " + result.display_text().replace("\n", "\n  "))
    if result.ranking:
        print("  оценки: " + "; ".join(
            f"{action}={score:.3f}" for action, score in result.ranking[:3]
        ))


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=project / "cv/data/raw_screenshots")
    parser.add_argument("--capture", action="store_true", help="take screenshots itself")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--start", type=int, help="also process existing frames from this number")
    parser.add_argument("--end", type=int, help="last existing frame to replay")
    parser.add_argument("--once", action="store_true", help="exit after existing frames are processed")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--trump", type=parse_trump, metavar="МАСТЬ",
        help="manual trump: черви/H, бубны/D, трефы/C or пики/S",
    )
    parser.add_argument(
        "--auto-trump", action="store_true",
        help="do not ask; recognize the trump from screenshots",
    )
    parser.add_argument("--deck-count", type=int, choices=range(25), metavar="0..24")
    parser.add_argument("--role", choices=("attack", "defense"),
                        help="initial role override when starting mid-game")
    parser.add_argument("--log", type=Path, default=project / "cv/logs/live_game.jsonl")
    parser.add_argument("--state", type=Path, default=project / "cv/logs/live_state.json")
    parser.add_argument("--new-game", action="store_true",
                        help="ignore saved state and start a fresh tracker")
    parser.add_argument("--localizer", type=Path, default=project / "cv/models/card_localizer.pt")
    parser.add_argument("--classifier", type=Path, default=project / "cv/models/card_crop_classifier_v10.pt")
    parser.add_argument("--zone-label", type=Path, default=project / "cv/assets/game_zone.txt")
    parser.add_argument("--nn-model", type=Path, default=project / "durak_model_v2.pt")
    parser.add_argument("--v3-model", type=Path, default=project / "durak_model_v3.pt",
                        help="recurrent v3 checkpoint")
    parser.add_argument("--v4-model", type=Path, default=project / "durak_model_v4.pt",
                        help="belief-aware recurrent v4 checkpoint")
    parser.add_argument("--use-v2", action="store_true",
                        help="rollback to the unchanged --nn-model checkpoint")
    parser.add_argument("--use-v4", action="store_true",
                        help="use the experimental belief-aware v4 checkpoint")
    parser.add_argument("--search-ms", type=int, default=0,
                        help="information-set search budget per new position (v3/v4)")
    parser.add_argument("--search-simulations", type=int, default=128)
    args = parser.parse_args()
    if args.use_v2 and args.use_v4:
        parser.error("--use-v2 and --use-v4 are mutually exclusive")

    args.trump_prompt = None
    if args.trump is None and not args.auto_trump:
        args.trump_prompt = prompt_for_trump
        args.trump = prompt_for_trump()
        if args.trump:
            print(f"Козырь зафиксирован вручную: {TRUMP_NAMES[args.trump]}")
        else:
            args.trump_prompt = None

    core = build_core(args)
    overlay = None if args.no_overlay else Overlay()
    last_number = args.start - 1 if args.start is not None else max(
        (numeric_stem(path) for path in args.images.glob("game_*.png")), default=-1
    )
    capture_number = get_next_number()
    last_signature = None

    result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    stop_event = threading.Event()

    def process_path(path: Path):
        nonlocal last_number, last_signature
        result = core.process(path)
        if result is None:
            return
        last_number = numeric_stem(path)
        signature = (result.state, result.recommendation, result.actions, result.warnings)
        if signature != last_signature:
            print_result(result)
            last_signature = signature
        if overlay:
            result_queue.put(("result", result))

    def tick():
        nonlocal capture_number
        if args.capture:
            path = Path(capture_screenshot(f"game_{capture_number:04d}.png"))
            capture_number += 1
            process_path(path)
        else:
            pending = sorted(
                (
                    path for path in args.images.glob("game_*.png")
                    if numeric_stem(path) > last_number
                    and (args.end is None or numeric_stem(path) <= args.end)
                ),
                key=numeric_stem,
            )
            for path in pending:
                process_path(path)
            if args.once and not pending:
                return False
        return True

    if overlay:
        def worker():
            try:
                while not stop_event.is_set() and tick():
                    stop_event.wait(args.interval if args.capture else 0.35)
            except Exception as error:
                result_queue.put(("error", error))
            finally:
                result_queue.put(("done", None))

        def poll_results():
            try:
                while True:
                    kind, value = result_queue.get_nowait()
                    if kind == "result":
                        overlay.update(value)
                    elif kind == "error":
                        overlay.text.set(f"Ошибка: {value}")
                    elif kind == "done" and args.once:
                        overlay.root.destroy()
                        return
            except queue.Empty:
                pass
            overlay.root.after(100, poll_results)

        def close_overlay():
            stop_event.set()
            overlay.root.destroy()

        overlay.root.protocol("WM_DELETE_WINDOW", close_overlay)
        threading.Thread(target=worker, name="durak-recognizer", daemon=True).start()
        overlay.root.after(100, poll_results)
        overlay.root.mainloop()
    else:
        try:
            while tick():
                if args.once and not args.capture:
                    break
                time.sleep(args.interval if args.capture else 0.35)
        except KeyboardInterrupt:
            print("\nОстановлено")


if __name__ == "__main__":
    main()
