import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "cv/scripts"))

from durak_v3.model import RecurrentActorCritic  # noqa: E402
from game_state_tracker import GameStateTracker  # noqa: E402
from legal_move_advisor import Advice  # noqa: E402
from neural_v3_advisor import NeuralV3Advisor  # noqa: E402


class NeuralV3AdvisorTest(unittest.TestCase):
    def test_repeated_frame_does_not_advance_recurrent_memory(self):
        model = RecurrentActorCritic(hidden_size=32, option_size=16)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v3.pt"
            torch.save({
                "model": model.state_dict(),
                "model_config": {"hidden_size": 32, "option_size": 16},
            }, checkpoint)
            advisor = NeuralV3Advisor(checkpoint)
            tracker = GameStateTracker()
            tracker.initialized = True
            tracker.hand = {"6H", "7S"}
            tracker.player_attacker = True
            advice = Advice(("ход 6H", "ход 7S"))
            first = advisor.recommend(tracker, advice, "S", 24)
            hidden = advisor.hidden.clone()
            second = advisor.recommend(tracker, advice, "S", 24)
            self.assertEqual(first, second)
            self.assertTrue(torch.equal(hidden, advisor.hidden))


if __name__ == "__main__":
    unittest.main()
