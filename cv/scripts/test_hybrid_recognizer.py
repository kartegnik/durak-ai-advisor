#!/usr/bin/env python3
"""Run the hybrid Durak card recognizer on one screenshot and save a preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from card_crop_classifier import CLASS_NAMES, extract_crop, load_card_classifier
from status_detector import detect_statuses


def viewport_bounds(image, zone_label: Path, padding: int) -> tuple[int, int, int, int]:
    fields = zone_label.read_text().splitlines()[0].split()
    cx, cy, bw, bh = map(float, fields[1:])
    height, width = image.shape[:2]
    return (
        max(0, round((cx - bw / 2) * width) - padding),
        max(0, round((cy - bh / 2) * height) - padding),
        min(width, round((cx + bw / 2) * width) + padding),
        min(height, round((cy + bh / 2) * height) + padding),
    )


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=project / "cv/hybrid_result.jpg")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument(
        "--card-padding", type=float, default=0.0,
        help="Relative padding around detected boxes before classification",
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

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    vx1, vy1, vx2, vy2 = viewport_bounds(image, args.zone_label, args.padding)
    statuses = detect_statuses(image, project)
    viewport = image[vy1:vy2, vx1:vx2]

    detector = YOLO(args.localizer)
    result = detector.predict(
        viewport, imgsz=416, conf=args.confidence, iou=0.5,
        agnostic_nms=True, verbose=False,
    )[0]
    classifier = load_card_classifier(args.classifier)

    predictions = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
        height, width = viewport.shape[:2]
        normalized = (
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        )
        padded = (
            normalized[0], normalized[1],
            normalized[2] * (1 + args.card_padding * 2),
            normalized[3] * (1 + args.card_padding * 2),
        )
        crop = extract_crop(viewport, padded)
        tensor = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad():
            probabilities = classifier(tensor).softmax(dim=1)[0]
        score, class_id = probabilities.max(dim=0)
        top_scores, top_ids = probabilities.topk(3)
        alternatives = tuple(
            (CLASS_NAMES[class_index], class_score.item())
            for class_score, class_index in zip(top_scores, top_ids)
        )
        predictions.append((
            x1, y1, x2, y2, CLASS_NAMES[class_id], score.item(), box.conf[0].item(), alternatives
        ))

    # A 36-card deck cannot contain the same physical card twice. Detectors can
    # nevertheless draw two partially shifted boxes around one card, especially
    # in a fanned hand. Keep the strongest localization for each predicted card.
    unique_predictions = {}
    for prediction in predictions:
        card = prediction[4]
        score = prediction[5] * prediction[6]
        previous = unique_predictions.get(card)
        if previous is None or score > previous[5] * previous[6]:
            unique_predictions[card] = prediction
    suppressed = len(predictions) - len(unique_predictions)
    predictions = list(unique_predictions.values())

    preview = viewport.copy()
    predictions.sort(key=lambda item: (item[1], item[0]))
    for x1, y1, x2, y2, card, class_conf, box_conf, _alternatives in predictions:
        color = (0, 200, 0) if class_conf >= 0.65 else (0, 180, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        label = f"{card} card={class_conf:.2f} box={box_conf:.2f}"
        cv2.putText(preview, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), preview)
    print(f"detected: {len(predictions)}")
    if suppressed:
        print(f"suppressed duplicate boxes: {suppressed}")
    for item in predictions:
        alternatives = ", ".join(f"{card}={score:.3f}" for card, score in item[7])
        status = "UNCERTAIN  " if item[5] < 0.65 else ""
        print(f"{status}{item[4]}  class={item[5]:.3f}  box={item[6]:.3f}  top3=[{alternatives}]")
    print(f"status top: {statuses.top or '-'} ({statuses.top_score:.3f})")
    print(f"status bottom: {statuses.bottom or '-'} ({statuses.bottom_score:.3f})")
    print("events: " + (", ".join(statuses.events) if statuses.events else "-"))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
