#!/usr/bin/env python3
"""Export small runtime templates so private screenshots need not be shipped."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from deck_count_detector import _glyphs
from status_detector import glyph_signature, status_crop
from test_hybrid_recognizer import viewport_bounds
from trump_detector import _suit_glyph, extract_trump


def require_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    screenshots = project / "cv/data/raw_screenshots"
    assets = project / "cv/assets"
    assets.mkdir(parents=True, exist_ok=True)
    zone = project / "cv/data/for-mimo/game_0001.txt"

    status_sources = {
        "take_bottom": (5, "bottom"),
        "take_top": (289, "top"),
        "defended_top": (15, "top"),
        "defended_bottom": (24, "bottom"),
        "pass_bottom": (289, "bottom"),
    }
    status = {
        name: glyph_signature(status_crop(
            require_image(screenshots / f"game_{frame:04d}.png"), position,
        ))
        for name, (frame, position) in status_sources.items()
    }
    np.savez_compressed(assets / "status_templates.npz", **status)

    digit_sources = (
        (1, "24"), (7, "22"), (16, "18"), (30, "8"), (40, "6"),
        (2453, "24"), (2496, "22"), (2540, "20"), (2559, "18"),
        (2604, "16"), (2650, "15"), (2700, "13"), (2850, "8"),
        (2900, "2"),
    )
    digits: dict[str, list[np.ndarray]] = {}
    for frame, text in digit_sources:
        image = require_image(screenshots / f"game_{frame:04d}.png")
        bounds = viewport_bounds(image, zone, 20)
        glyphs = _glyphs(image[bounds[1]:bounds[3], bounds[0]:bounds[2]])
        if len(glyphs) != len(text):
            raise ValueError(f"could not segment deck counter in game_{frame:04d}")
        for digit, glyph in zip(text, glyphs):
            digits.setdefault(digit, []).append(glyph)
    np.savez_compressed(assets / "deck_digit_templates.npz", **{
        digit: np.stack(variants) for digit, variants in digits.items()
    })

    suit_sources = ((1, "S"), (350, "D"), (479, "H"), (582, "C"), (766, "C"))
    suits: dict[str, list[np.ndarray]] = {}
    for frame, suit in suit_sources:
        image = require_image(screenshots / f"game_{frame:04d}.png")
        bounds = viewport_bounds(image, zone, 20)
        viewport = image[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        suits.setdefault(suit, []).append(_suit_glyph(extract_trump(viewport)))
    np.savez_compressed(assets / "trump_suit_templates.npz", **{
        suit: np.stack(variants) for suit, variants in suits.items()
    })

    shutil.copy2(
        project / "runs/detect/durak_training/card_localizer_expanded_416/weights/best.pt",
        project / "cv/models/card_localizer.pt",
    )
    print(f"Exported runtime assets to {assets}")


if __name__ == "__main__":
    main()
