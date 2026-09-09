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

    def test_v1_policy_asset_accepts_dynamic_options(self):
        session = ort.InferenceSession(
            MODELS / "durak_policy_v1.onnx", providers=("CPUExecutionProvider",),
        )
        self.assertEqual(session.get_inputs()[0].name, "combined")
        scores = session.run(None, {
            "combined": np.zeros((3, 193), dtype=np.float32),
        })[0]
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())

    def test_v3_policy_asset_accepts_dynamic_options(self):
        session = ort.InferenceSession(
            MODELS / "durak_policy_v3.onnx", providers=("CPUExecutionProvider",),
        )
        self.assertEqual(
            [item.name for item in session.get_inputs()],
            ["observation", "options", "hidden"],
        )
        logits, hidden = session.run(None, {
            "observation": np.zeros(284, dtype=np.float32),
            "options": np.zeros((3, 43), dtype=np.float32),
            "hidden": np.zeros(192, dtype=np.float32),
        })
        self.assertEqual(logits.shape, (3,))
        self.assertEqual(hidden.shape, (192,))
        self.assertTrue(np.isfinite(logits).all())
        self.assertTrue(np.isfinite(hidden).all())

    def test_v4_policy_asset_accepts_dynamic_options_and_two_memories(self):
        session = ort.InferenceSession(
            MODELS / "durak_policy_v4.onnx", providers=("CPUExecutionProvider",),
        )
        self.assertEqual(
            [item.name for item in session.get_inputs()],
            ["observation", "options", "policy_hidden", "belief_hidden"],
        )
        logits, policy_hidden, belief_hidden = session.run(None, {
            "observation": np.zeros(284, dtype=np.float32),
            "options": np.zeros((3, 43), dtype=np.float32),
            "policy_hidden": np.zeros(192, dtype=np.float32),
            "belief_hidden": np.zeros(192, dtype=np.float32),
        })
        self.assertEqual(logits.shape, (3,))
        self.assertEqual(policy_hidden.shape, (192,))
        self.assertEqual(belief_hidden.shape, (192,))
        self.assertTrue(np.isfinite(logits).all())
        self.assertTrue(np.isfinite(policy_hidden).all())
        self.assertTrue(np.isfinite(belief_hidden).all())


if __name__ == "__main__":
    unittest.main()
