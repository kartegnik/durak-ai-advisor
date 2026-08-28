#!/usr/bin/env python3
"""Recognize Durak Online status bubbles near both player avatars."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StatusResult:
    top: str | None
    bottom: str | None
    top_score: float
    bottom_score: float

    @property
    def events(self) -> tuple[str, ...]:
        events = []
        mapping = {
            ("top", "Беру"): "OPPONENT_TAKES",
            ("bottom", "Беру"): "PLAYER_TAKES",
            ("top", "Бито"): "PLAYER_DEFENDED",
            ("bottom", "Бито"): "OPPONENT_DEFENDED",
            ("top", "Пас"): "OPPONENT_STOPPED_THROWING",
            ("bottom", "Пас"): "PLAYER_STOPPED_THROWING",
        }
        for position, status in (("top", self.top), ("bottom", self.bottom)):
            if status:
                events.append(mapping[(position, status)])
        return tuple(events)


def status_crop(image: np.ndarray, position: str) -> np.ndarray:
    """Crop a resolution-independent word area around an avatar."""
    height, width = image.shape[:2]
    # Keep this tight around the word itself. A wider crop includes changing
    # avatar/card pixels and makes identical text look different.
    x1, x2 = round(width * 0.500), round(width * 0.536)
    if position == "top":
        y1, y2 = round(height * 0.137), round(height * 0.166)
    else:
        y1, y2 = round(height * 0.824), round(height * 0.851)
    return image[y1:y2, x1:x2]


def glyph_signature(crop: np.ndarray) -> np.ndarray:
    """Normalize dark Cyrillic glyphs while mostly ignoring bubble colour."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 40), interpolation=cv2.INTER_AREA)
    # Text is dark on either a white or yellow notification background.
    dark = (gray < 125).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return dark


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_f = left.astype(np.float32).ravel() / 255
    right_f = right.astype(np.float32).ravel() / 255
    denominator = np.linalg.norm(left_f) * np.linalg.norm(right_f)
    return float(left_f @ right_f / denominator) if denominator else 0.0


def load_templates(project: Path) -> dict[str, list[np.ndarray]]:
    asset_path = project / "cv/assets/status_templates.npz"
    if asset_path.exists():
        saved = np.load(asset_path)
        return {
            "Беру": [saved["take_bottom"], saved["take_top"]],
            "Бито": [saved["defended_top"], saved["defended_bottom"]],
            "Пас": [saved["pass_bottom"]],
        }
    examples = {
        "Беру": (("game_0005", "bottom"), ("game_0289", "top")),
        "Бито": (("game_0015", "top"), ("game_0024", "bottom")),
        "Пас": (("game_0289", "bottom"),),
    }
    images = project / "cv/data/raw_screenshots"
    templates = {}
    for label, sources in examples.items():
        templates[label] = []
        for stem, position in sources:
            image = cv2.imread(str(images / f"{stem}.png"))
            templates[label].append(glyph_signature(status_crop(image, position)))
    return templates


def classify(crop: np.ndarray, templates: dict[str, list[np.ndarray]], threshold: float) -> tuple[str | None, float]:
    signature = glyph_signature(crop)
    scores = {
        label: max(similarity(signature, template) for template in variants)
        for label, variants in templates.items()
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    return (label if score >= threshold else None), score


def detect_statuses(image: np.ndarray, project: Path, threshold: float = 0.85) -> StatusResult:
    templates = load_templates(project)
    top, top_score = classify(status_crop(image, "top"), templates, threshold)
    bottom, bottom_score = classify(status_crop(image, "bottom"), templates, threshold)
    return StatusResult(top, bottom, top_score, bottom_score)


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    result = detect_statuses(image, project, args.threshold)
    print(f"top: {result.top or '-'} ({result.top_score:.3f})")
    print(f"bottom: {result.bottom or '-'} ({result.bottom_score:.3f})")
    print("events: " + (", ".join(result.events) if result.events else "-"))


if __name__ == "__main__":
    main()
