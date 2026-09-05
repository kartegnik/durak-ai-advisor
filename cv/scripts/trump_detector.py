#!/usr/bin/env python3
"""Recognize the face-up trump card at its fixed place beside the deck."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from card_crop_classifier import CLASS_NAMES


@dataclass(frozen=True)
class TrumpDetection:
    suit: str | None
    confidence: float
    card: str | None
    card_present: bool


def _suit_glyph(crop: np.ndarray) -> np.ndarray:
    """Normalize only the large suit mark; ignore rank and court artwork."""
    # In the deskewed crop the large suit mark lives below the rank. Keeping
    # the ROI narrow is important: otherwise 6S and JS look more like each
    # other than two cards of the same suit do.
    suit_roi = crop[14:32, :18]
    hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (saturation > 95) & ((hue < 15) | (hue > 165))
    black = value < 105
    ink = (red | black).astype(np.uint8) * 255
    return cv2.resize(ink, (32, 44), interpolation=cv2.INTER_NEAREST)


class TrumpSuitDetector:
    def __init__(self, templates: dict[str, list[np.ndarray]]):
        self.templates = templates

    @classmethod
    def from_examples(cls, examples: list[tuple[np.ndarray, str]]):
        templates: dict[str, list[np.ndarray]] = {}
        for viewport, suit in examples:
            templates.setdefault(suit, []).append(_suit_glyph(extract_trump(viewport)))
        return cls(templates)

    @classmethod
    def from_asset(cls, path: Path) -> "TrumpSuitDetector":
        saved = np.load(path)
        return cls({suit: list(saved[suit]) for suit in saved.files})

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        glyph = _suit_glyph(crop)
        distances = {
            suit: min(float(np.mean(cv2.absdiff(glyph, template))) for template in variants)
            for suit, variants in self.templates.items()
        }
        ordered = sorted(distances.items(), key=lambda item: item[1])
        suit, best = ordered[0]
        margin = ordered[1][1] - best
        confidence = max(0.0, min(1.0, 0.55 + margin / 100.0 - best / 500.0))
        return suit, confidence


def extract_trump(viewport: np.ndarray) -> np.ndarray:
    """Deskew the fixed, slightly rotated face-up card into a 64x96 crop."""
    height, width = viewport.shape[:2]
    # Coordinates are normalized to the game viewport and remain stable across
    # desktop resolutions. The left edge may be covered slightly by the deck.
    source = np.float32([
        [0.090 * width, 0.430 * height],
        [0.235 * width, 0.455 * height],
        [0.195 * width, 0.600 * height],
        [0.025 * width, 0.565 * height],
    ])
    target = np.float32([[0, 0], [63, 0], [63, 95], [0, 95]])
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(viewport, matrix, (64, 96), borderMode=cv2.BORDER_REPLICATE)


def detect_trump(viewport: np.ndarray, classifier, suit_detector: TrumpSuitDetector | None = None) -> TrumpDetection:
    crop = extract_trump(viewport)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white_fraction = float(np.mean((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 155)))
    white_card_candidate = white_fraction >= 0.18
    # The exposed card corner changes orientation depending on how much of the
    # deck covers it. Score several rotations and keep each suit's best view.
    crops = (
        crop,
        cv2.resize(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), (64, 96)),
        cv2.resize(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), (64, 96)),
        cv2.resize(cv2.rotate(crop, cv2.ROTATE_180), (64, 96)),
    )
    tensor = torch.stack([
        torch.from_numpy(cv2.cvtColor(item, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        for item in crops
    ])
    with torch.no_grad():
        probabilities = classifier(tensor).softmax(dim=1)
    # Sum all nine ranks for each suit. Rank can be obscured by the deck, while
    # the suit symbol remains enough for legal-move generation.
    suit_scores = {
        suit: max(
            sum(row[index].item() for index, name in enumerate(CLASS_NAMES) if name.endswith(suit))
            for row in probabilities
        )
        for suit in ("H", "D", "C", "S")
    }
    suit, confidence = max(suit_scores.items(), key=lambda item: item[1])
    if suit_detector is not None:
        suit, confidence = suit_detector.classify(crop)
        suit_ink_fraction = float(np.mean(_suit_glyph(crop) > 0))
        # A white popup or another window can overlap this fixed area. Treat
        # the card as present only when it also contains a credible suit mark.
        card_present = (
            white_card_candidate
            and suit_ink_fraction >= 0.03
            and confidence >= 0.55
        )
    else:
        card_present = white_card_candidate
    flat_id = probabilities.argmax().item()
    class_id = flat_id % len(CLASS_NAMES)
    return TrumpDetection(
        suit if card_present and confidence >= 0.55 else None,
        confidence,
        CLASS_NAMES[class_id],
        card_present,
    )
