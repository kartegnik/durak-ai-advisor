#!/usr/bin/env python3
"""Replay screenshots through the recognizer while preserving Durak state."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from card_crop_classifier import CLASS_NAMES, extract_crop, load_card_classifier
from deck_count_detector import DeckCountDetector, resolve_hidden_counter
from game_state_tracker import FrameObservation, GameStateTracker, ObservedCard
from legal_move_advisor import legal_actions
from neural_move_advisor import NeuralMoveAdvisor
from neural_v3_advisor import NeuralV3Advisor
from neural_v4_advisor import NeuralV4Advisor
from status_detector import detect_statuses
from test_hybrid_recognizer import viewport_bounds
from trump_detector import detect_trump, TrumpSuitDetector


class FrameRecognizer:
    def __init__(self, project: Path, localizer: Path, classifier_path: Path, zone_label: Path):
        self.project = project
        self.localizer = YOLO(localizer)
        self.classifier = load_card_classifier(classifier_path)
        self.zone_label = zone_label
        self.last_trump = None
        self.last_trump_confidence = 0.0
        self.trump_candidate = None
        self.trump_candidate_count = 0
        deck_asset = project / "cv/assets/deck_digit_templates.npz"
        if not deck_asset.exists():
            raise FileNotFoundError(
                f"missing runtime template {deck_asset}; run export_runtime_assets.py"
            )
        self.deck_detector = DeckCountDetector.from_asset(deck_asset)
        self.deck_count = None
        self.deck_candidate = None
        self.deck_candidate_count = 0
        self.current_trump_detection = None
        trump_asset = project / "cv/assets/trump_suit_templates.npz"
        if not trump_asset.exists():
            raise FileNotFoundError(
                f"missing runtime template {trump_asset}; run export_runtime_assets.py"
            )
        self.trump_suit_detector = TrumpSuitDetector.from_asset(trump_asset)

    def reset_game(self) -> None:
        """Clear values that are fixed or monotonic only within one game."""
        current_trump = self.current_trump_detection
        self.last_trump = None
        self.last_trump_confidence = 0.0
        self.trump_candidate = current_trump.suit if current_trump else None
        self.trump_candidate_count = 1 if current_trump and current_trump.suit else 0
        self.deck_count = 24
        self.deck_candidate = 24
        self.deck_candidate_count = 2

    @staticmethod
    def zone(center_y: float, viewport_height: int) -> str:
        ratio = center_y / viewport_height
        if ratio < 0.22:
            return "opponent"
        if ratio >= 0.68:
            return "hand"
        return "table"

    def recognize(self, image: cv2.typing.MatLike, confidence: float) -> FrameObservation:
        vx1, vy1, vx2, vy2 = viewport_bounds(image, self.zone_label, 20)
        viewport = image[vy1:vy2, vx1:vx2]
        trump = detect_trump(viewport, self.classifier, self.trump_suit_detector)
        self.current_trump_detection = trump
        if self.last_trump is None and trump.suit:
            if trump.suit == self.trump_candidate:
                self.trump_candidate_count += 1
            else:
                self.trump_candidate = trump.suit
                self.trump_candidate_count = 1
            if self.trump_candidate_count >= 2:
                self.last_trump = trump.suit
                self.last_trump_confidence = trump.confidence
        deck = self.deck_detector.detect(viewport)
        result = self.localizer.predict(
            viewport, imgsz=416, conf=confidence, iou=0.5,
            agnostic_nms=True, verbose=False
        )[0]
        candidates = []
        height, width = viewport.shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
            normalized = (
                ((x1 + x2) / 2) / width,
                ((y1 + y2) / 2) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
            crop = extract_crop(viewport, normalized)
            tensor = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
            candidates.append((box, normalized, tensor))

        observed = []
        if candidates:
            with torch.no_grad():
                probabilities = self.classifier(torch.stack([item[2] for item in candidates])).softmax(dim=1)
            for (box, normalized, _), scores in zip(candidates, probabilities):
                class_score, class_id = scores.max(dim=0)
                observed.append(ObservedCard(
                    CLASS_NAMES[class_id],
                    self.zone(normalized[1] * height, height),
                    class_score.item(),
                    box.conf[0].item(),
                    normalized[0],
                    normalized[1],
                ))

        # One physical deck contains one card of each class. Keep its strongest
        # observation if YOLO drew shifted duplicate boxes.
        unique = {}
        for item in observed:
            score = item.class_confidence * item.box_confidence
            previous = unique.get(item.card)
            if previous is None or score > previous.class_confidence * previous.box_confidence:
                unique[item.card] = item
        observed_deck_count = resolve_hidden_counter(
            deck,
            trump_card_present=trump.card_present,
            gameplay_card_visible=bool(unique),
        )
        if observed_deck_count is not None:
            if observed_deck_count == self.deck_candidate:
                self.deck_candidate_count += 1
            else:
                self.deck_candidate = observed_deck_count
                self.deck_candidate_count = 1
            if self.deck_candidate_count >= 2:
                plausible = self.deck_count is None or observed_deck_count <= self.deck_count
                if plausible:
                    self.deck_count = observed_deck_count
        statuses = detect_statuses(image, self.project)
        return FrameObservation(tuple(unique.values()), statuses.events, self.deck_count)


def numeric_stem(path: Path) -> int:
    return int(path.stem.split("_")[1])


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=project / "cv/data/raw_screenshots")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--trump", choices=("H", "D", "C", "S"),
                        help="override automatic trump detection")
    parser.add_argument(
        "--nn-model", type=Path, default=project / "durak_model_v2.pt",
        help="trained move-scoring checkpoint",
    )
    parser.add_argument("--v3-model", type=Path, default=project / "durak_model_v3.pt")
    parser.add_argument("--v4-model", type=Path, default=project / "durak_model_v4.pt")
    parser.add_argument("--use-v2", action="store_true")
    parser.add_argument("--use-v4", action="store_true")
    parser.add_argument("--search-ms", type=int, default=0)
    parser.add_argument("--search-simulations", type=int, default=128)
    parser.add_argument(
        "--deck-count", type=int, choices=range(0, 25), metavar="0..24",
        help="known cards remaining in deck; enables neural recommendation",
    )
    parser.add_argument(
        "--localizer", type=Path,
        default=project / "cv/models/card_localizer.pt",
    )
    parser.add_argument(
        "--classifier", type=Path,
        default=project / "cv/models/card_crop_classifier_v10.pt",
    )
    parser.add_argument(
        "--zone-label", type=Path,
        default=project / "cv/assets/game_zone.txt",
    )
    args = parser.parse_args()
    if args.use_v2 and args.use_v4:
        parser.error("--use-v2 and --use-v4 are mutually exclusive")

    paths = [
        path for path in sorted(args.images.glob("game_*.png"), key=numeric_stem)
        if numeric_stem(path) >= args.start
        and (args.end is None or numeric_stem(path) <= args.end)
    ]
    recognizer = FrameRecognizer(project, args.localizer, args.classifier, args.zone_label)
    tracker = GameStateTracker(minimum_initial_hand=5)
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
    announced_trump = None
    announced_deck_count = None
    for path in paths:
        image = cv2.imread(str(path))
        observation = recognizer.recognize(image, args.confidence)
        update = tracker.observe(observation)
        trump = args.trump or recognizer.last_trump
        deck_count = args.deck_count if args.deck_count is not None else recognizer.deck_count
        if trump and trump != announced_trump:
            print(f"trump: {trump} (automatic)" if not args.trump else f"trump: {trump} (manual)")
            announced_trump = trump
        if deck_count is not None and deck_count != announced_deck_count:
            source = "manual" if args.deck_count is not None else "automatic"
            print(f"deck count: {deck_count} ({source})")
            announced_deck_count = deck_count
        if update.changes:
            print(path.name)
            for change in update.changes:
                print(f"  {change}")
            print(f"  {tracker.summary()}")
            if trump:
                advice = legal_actions(tracker, trump)
                if advice.actions:
                    print("  можно: " + "; ".join(advice.actions))
                    if deck_count is not None:
                        recommendation = neural.recommend(
                            tracker, advice, trump, deck_count
                        )
                        print(f"  нейросеть: {recommendation.action}")
                        ranking = "; ".join(
                            f"{action}={score:.3f}"
                            for action, score in recommendation.scores[:3]
                        )
                        print(f"  оценки: {ranking}")
                elif advice.note:
                    print("  " + advice.note)


if __name__ == "__main__":
    main()
