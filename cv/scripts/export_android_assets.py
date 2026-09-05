#!/usr/bin/env python3
"""Convert published runtime assets to dependency-free Android resources."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_trump_templates(source: Path, target: Path) -> None:
    saved = np.load(source)
    templates = {suit: list(saved[suit]) for suit in saved.files}
    payload = bytearray(b"DTS1")
    payload.append(len(templates))
    for suit in sorted(templates):
        variants = templates[suit]
        payload.extend((ord(suit), len(variants)))
        for glyph in variants:
            if glyph.shape != (44, 32):
                raise ValueError(f"unexpected {suit} glyph shape: {glyph.shape}")
            payload.extend(np.ascontiguousarray(glyph, dtype=np.uint8).tobytes())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    source = project / "cv/assets/trump_suit_templates.npz"
    target = project / "android/app/src/main/assets/trump_suit_templates.bin"
    write_trump_templates(source, target)
    print(f"Exported {target.relative_to(project)}")


if __name__ == "__main__":
    main()
