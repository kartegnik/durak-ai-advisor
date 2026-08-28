#!/usr/bin/env python3
"""Read the fixed deck counter shown to the left of the trump card."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# The UI shows the number of cards, not the number of complete deals.  It may
# therefore be odd when the players draw different amounts after a bout.
# Values 0 and 1 are rendered as a blank counter and resolved by the caller.
VALID_COUNTS = set(range(2, 25))


def _white_mask(region: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 175), (179, 75, 255))


def _glyphs(viewport: np.ndarray, box=(0.0, 0.315, 0.18, 0.405)) -> list[np.ndarray]:
    height, width = viewport.shape[:2]
    x1, y1, x2, y2 = (
        round(box[0] * width), round(box[1] * height),
        round(box[2] * width), round(box[3] * height),
    )
    mask = _white_mask(viewport[y1:y2, x1:x2])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= 16 and w >= 4:
            candidates.append((x, mask[y:y + h, x:x + w]))
    candidates.sort(key=lambda item: item[0])
    return [
        cv2.resize(glyph, (24, 40), interpolation=cv2.INTER_AREA)
        for _, glyph in candidates[:2]
    ]


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


@dataclass(frozen=True)
class DeckCountDetection:
    count: int | None
    confidence: float
    counter_visible: bool = True


def resolve_hidden_counter(
    detection: DeckCountDetection,
    trump_card_present: bool,
    gameplay_card_visible: bool,
) -> int | None:
    """Resolve the blank 0/1 counter without mistaking loading screens for 0."""
    if detection.counter_visible:
        return detection.count
    if trump_card_present:
        return 1
    if gameplay_card_visible:
        return 0
    return None


class DeckCountDetector:
    def __init__(self, templates: dict[str, list[np.ndarray]]):
        self.templates = templates

    @classmethod
    def from_viewports(cls, examples: list[tuple[np.ndarray, str]]):
        templates: dict[str, list[np.ndarray]] = {}
        for viewport, text in examples:
            glyphs = _glyphs(viewport)
            if len(glyphs) != len(text):
                raise ValueError(f"could not segment deck template {text}")
            for digit, glyph in zip(text, glyphs):
                templates.setdefault(digit, []).append(glyph)
        return cls(templates)

    @classmethod
    def from_asset(cls, path: Path) -> "DeckCountDetector":
        saved = np.load(path)
        return cls({digit: list(saved[digit]) for digit in saved.files})

    def detect(self, viewport: np.ndarray) -> DeckCountDetection:
        glyphs = _glyphs(viewport)
        if not glyphs:
            # Both 0 and 1 are displayed without a number. The remaining
            # face-up trump card distinguishes them in the caller.
            return DeckCountDetection(None, 1.0, False)
        digits = []
        distances = []
        for glyph in glyphs:
            choices = [
                (digit, min(_distance(glyph, template) for template in variants))
                for digit, variants in self.templates.items()
            ]
            digit, distance = min(choices, key=lambda item: item[1])
            digits.append(digit)
            distances.append(distance)
        value = int("".join(digits))
        confidence = max(0.0, 1.0 - float(np.mean(distances)) / 90.0)
        if value not in VALID_COUNTS or confidence < 0.55:
            return DeckCountDetection(None, confidence)
        return DeckCountDetection(value, confidence)
