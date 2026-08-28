import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "cv/scripts"))
from game_state_tracker import GameStateTracker  # noqa: E402
from legal_move_advisor import Advice  # noqa: E402
from neural_move_advisor import (  # noqa: E402
    NeuralMoveAdvisor, action_to_move, strategic_adjustments, to_card,
)
from durakgame import DefensiveMove, TransferMove  # noqa: E402


class NeuralMoveAdvisorTest(unittest.TestCase):
    def test_action_conversion(self):
        self.assertIsInstance(action_to_move("отбить 8H картой 9H"), DefensiveMove)
        self.assertIsInstance(action_to_move("перевести картой 8D"), TransferMove)
        self.assertEqual(str(to_card("10S")), "Ten(Spade)")

    def test_checkpoint_scores_legal_actions(self):
        tracker = GameStateTracker()
        tracker.hand = {"9H", "6S"}
        tracker.table = {"8H"}
        tracker.table_attack = ["8H"]
        tracker.player_attacker = False
        advisor = NeuralMoveAdvisor(PROJECT / "durak_model_v2.pt")
        result = advisor.recommend(
            tracker, Advice(("отбить 8H картой 9H", "отбить 8H картой 6S", "взять")),
            "S", 24,
        )
        self.assertEqual(len(result.scores), 3)
        self.assertIn(result.action, {item[0] for item in result.scores})

    def test_early_trump_attack_is_penalized_but_endgame_is_not(self):
        actions = ("ход 7S", "ход QD")
        early = strategic_adjustments(actions, "S", 24)
        endgame = strategic_adjustments(actions, "S", 0)
        self.assertLess(early[0], -0.2)
        self.assertEqual(early[1], 0.0)
        self.assertEqual(endgame, (0.0, 0.0))

    def test_logged_early_defense_now_preserves_trumps(self):
        tracker = GameStateTracker()
        tracker.hand = {"10H", "9C", "AD", "JD", "KS", "QS"}
        tracker.table = {"6H"}
        tracker.table_attack = ["6H"]
        tracker.player_attacker = False
        advisor = NeuralMoveAdvisor(PROJECT / "durak_model_v2.pt")
        result = advisor.recommend(
            tracker,
            Advice((
                "отбить 6H картой 10H", "отбить 6H картой JD",
                "отбить 6H картой AD", "взять",
            )),
            "D", 24,
        )
        self.assertEqual(result.action, "отбить 6H картой 10H")

    def test_defense_prefers_lower_trump_when_both_cards_beat(self):
        actions = ("отбить 8S картой KS", "отбить 8S картой AS", "взять")
        corrections = strategic_adjustments(actions, "S", 0)
        self.assertEqual(corrections[0], 0.0)
        self.assertLess(corrections[1], corrections[0])


if __name__ == "__main__":
    unittest.main()
