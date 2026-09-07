import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort


PROJECT = Path(__file__).resolve().parent
MODELS = PROJECT / "android/app/src/main/assets/models"


class AndroidCardModelsTest(unittest.TestCase):
    def test_localizer_asset_has_expected_interface(self):
        session = ort.InferenceSession(
            MODELS / "card_localizer.onnx", providers=("CPUExecutionProvider",),
        )
        self.assertEqual(session.get_inputs()[0].name, "images")
        self.assertEqual(session.get_inputs()[0].shape, [1, 3, 416, 416])
        output = session.run(None, {
            "images": np.zeros((1, 3, 416, 416), dtype=np.float32),
        })[0]
        self.assertEqual(output.shape, (1, 5, 3549))
        self.assertTrue(np.isfinite(output).all())

    def test_classifier_asset_accepts_dynamic_batch(self):
        session = ort.InferenceSession(
            MODELS / "card_classifier.onnx", providers=("CPUExecutionProvider",),
        )
        self.assertEqual(session.get_inputs()[0].name, "cards")
        output = session.run(None, {
            "cards": np.zeros((2, 3, 96, 64), dtype=np.float32),
        })[0]
        self.assertEqual(output.shape, (2, 36))
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
