import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "cv/scripts"))

from trump_detector import TrumpSuitDetector, _suit_glyph  # noqa: E402


class TrumpSuitAssetTest(unittest.TestCase):
    def test_glyph_ignores_rank_and_card_artwork(self):
        first = np.full((96, 64, 3), 255, dtype=np.uint8)
        second = first.copy()

        # Identical black suit mark in the dedicated ROI.
        cv2.circle(first, (8, 22), 5, (0, 0, 0), -1)
        cv2.circle(second, (8, 22), 5, (0, 0, 0), -1)

        # Deliberately different rank and artwork outside that ROI.
        cv2.putText(first, "6", (1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(second, "K", (1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.rectangle(first, (20, 35), (60, 90), (255, 0, 0), -1)
        cv2.rectangle(second, (20, 35), (60, 90), (0, 0, 255), -1)

        np.testing.assert_array_equal(_suit_glyph(first), _suit_glyph(second))

    def test_runtime_asset_contains_four_distinct_suits(self):
        detector = TrumpSuitDetector.from_asset(
            PROJECT / "cv/assets/trump_suit_templates.npz"
        )
        self.assertEqual(set(detector.templates), {"H", "D", "C", "S"})

        for variants in detector.templates.values():
            for glyph in variants:
                self.assertEqual(glyph.shape, (44, 32))

        for first_index, first_suit in enumerate(sorted(detector.templates)):
            for second_suit in sorted(detector.templates)[first_index + 1:]:
                distance = min(
                    float(np.mean(cv2.absdiff(first, second)))
                    for first in detector.templates[first_suit]
                    for second in detector.templates[second_suit]
                )
                self.assertGreater(distance, 20.0)


if __name__ == "__main__":
    unittest.main()
