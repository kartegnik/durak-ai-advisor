import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "cv/scripts"))
from game_state_tracker import GameStateTracker  # noqa: E402
from legal_move_advisor import beats, legal_actions  # noqa: E402
from live_advisor import LiveResult  # noqa: E402


class LegalMoveAdvisorTest(unittest.TestCase):
    def test_offers_conditional_opening_without_guessing_role(self):
        tracker = GameStateTracker()
        tracker.hand = {"8D"}
        tracker.initialized = True
        advice = legal_actions(tracker, "S")
        self.assertEqual(advice.actions, ("ход 8D",))
        self.assertIn("если первый ход ваш", advice.note)
        self.assertIsNone(tracker.player_attacker)

    def test_overlay_explains_unknown_opening_role(self):
        result = LiveResult(
            frame="game_0001.png", state="hand=[8D] table=[-] discard=0 role=unknown",
            trump="S", deck_count=24, actions=("ход 8D",), recommendation="ход 8D", ranking=(),
            phase_note="если первый ход ваш", conditional=True, warnings=(), changes=(),
        )
        self.assertIn("ЕСЛИ АТАКУЕМ МЫ: ход 8D", result.display_text())
        self.assertNotIn("Ожидание хода соперника", result.display_text())

    def test_beating_rules(self):
        self.assertTrue(beats("9H", "8H", "S"))
        self.assertTrue(beats("6S", "AH", "S"))
        self.assertFalse(beats("AH", "6S", "S"))
        self.assertFalse(beats("9D", "8H", "S"))

    def test_defense_and_transfer(self):
        tracker = GameStateTracker()
        tracker.hand = {"8D", "9H", "6S"}
        tracker.initialized = True
        tracker.player_attacker = False
        tracker.table_attack = ["8H"]
        tracker.table = {"8H"}
        actions = legal_actions(tracker, "S").actions
        self.assertIn("отбить 8H картой 9H", actions)
        self.assertIn("отбить 8H картой 6S", actions)
        self.assertIn("перевести картой 8D", actions)
        self.assertIn("взять", actions)

    def test_attacker_can_only_throw_matching_rank(self):
        tracker = GameStateTracker()
        tracker.hand = {"8D", "9C", "AH"}
        tracker.initialized = True
        tracker.player_attacker = True
        tracker.table_attack = ["8H"]
        tracker.table_defense = ["9H"]
        tracker.table = {"8H", "9H"}
        self.assertEqual(
            legal_actions(tracker, "S").actions,
            ("подкинуть 8D", "подкинуть 9C", "бито / пас"),
        )


if __name__ == "__main__":
    unittest.main()
