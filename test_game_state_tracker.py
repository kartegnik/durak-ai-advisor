import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "cv/scripts"))

from game_state_tracker import FrameObservation, GameStateTracker, ObservedCard  # noqa: E402


def card(name, zone):
    return ObservedCard(name, zone)


class GameStateTrackerTest(unittest.TestCase):
    def test_opponent_take_is_remembered_until_cards_are_played(self):
        tracker = GameStateTracker()
        tracker.initialized = True
        tracker.hand = {"6H"}
        tracker.table = {"7C", "8D"}
        tracker.table_attack = ["7C"]
        tracker.table_defense = ["8D"]
        tracker.observe(FrameObservation(events=("OPPONENT_TAKES",), deck_count=20))
        self.assertEqual(tracker.opponent_known, {"7C", "8D"})
        tracker.observe(FrameObservation(
            cards=(ObservedCard("7C", "table"),), deck_count=20,
        ))
        self.assertNotIn("7C", tracker.opponent_known)

    def test_empty_deck_rejects_reclassified_hand_card(self):
        tracker = GameStateTracker()
        tracker.hand = {"7D", "7S", "KS", "QD"}
        tracker.initialized = True
        tracker.hand_reconcile_pending = True

        mistaken_snapshot = FrameObservation(
            cards=(
                ObservedCard("7D", "hand"),
                ObservedCard("KS", "hand"),
                ObservedCard("JD", "hand"),
            ),
            deck_count=0,
        )
        tracker.observe(mistaken_snapshot)
        tracker.observe(mistaken_snapshot)

        self.assertEqual(tracker.hand, {"7D", "7S", "KS", "QD"})

    def test_empty_deck_rejects_apparent_draw(self):
        tracker = GameStateTracker()
        tracker.hand = {"KS"}
        tracker.initialized = True
        observation = FrameObservation(
            cards=(ObservedCard("KS", "hand"), ObservedCard("QD", "hand")),
            deck_count=0,
        )

        tracker.observe(observation)
        tracker.observe(observation)

        self.assertEqual(tracker.hand, {"KS"})

    def test_last_deal_is_accepted_when_counter_reaches_zero(self):
        tracker = GameStateTracker()
        tracker.hand = {"7D", "QD"}
        tracker.initialized = True
        tracker.hand_reconcile_pending = True
        tracker.last_deck_count = 2
        final_hand = FrameObservation(
            cards=(
                ObservedCard("7D", "hand"),
                ObservedCard("QD", "hand"),
                ObservedCard("KS", "hand"),
            ),
            deck_count=0,
        )

        tracker.observe(final_hand)
        tracker.observe(final_hand)

        self.assertEqual(tracker.hand, {"7D", "QD", "KS"})
        self.assertEqual(tracker.empty_deck_grace, 0)

    def test_repeated_moderate_table_card_is_accepted(self):
        tracker = GameStateTracker()
        tracker.hand = {"9C"}
        tracker.table = {"JS"}
        tracker.table_attack = ["JS"]
        tracker.player_attacker = True
        tracker.initialized = True
        observation = FrameObservation(cards=(
            ObservedCard("JS", "table"),
            ObservedCard("6H", "table", class_confidence=0.62, box_confidence=0.49),
        ))

        tracker.observe(observation)
        self.assertNotIn("6H", tracker.table)
        tracker.observe(observation)

        self.assertIn("6H", tracker.table)
        self.assertEqual(tracker.table_defense, ["6H"])

    def test_weak_reclassification_does_not_rewrite_known_table_card(self):
        tracker = GameStateTracker()
        tracker.hand = {"KH"}
        tracker.table = {"JH", "JC"}
        tracker.table_attack = ["JH"]
        tracker.table_defense = ["JC"]
        tracker.player_attacker = False
        tracker.initialized = True
        observation = FrameObservation(cards=(
            ObservedCard("JH", "table"),
            ObservedCard("JS", "table", class_confidence=0.53, box_confidence=0.67),
        ))

        tracker.observe(observation)
        tracker.observe(observation)

        self.assertEqual(tracker.table, {"JH", "JC"})
        self.assertEqual(tracker.table_defense, ["JC"])

    def test_play_and_take_remembers_overlapped_cards(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("6C", "hand"), card("8H", "hand"))))
        tracker.observe(FrameObservation((card("8H", "table"), card("9D", "table"))))
        self.assertEqual(tracker.hand, {"6C"})
        tracker.observe(FrameObservation(events=("PLAYER_TAKES",)))
        self.assertEqual(tracker.hand, {"6C", "8H", "9D"})
        self.assertEqual(tracker.table, set())
        self.assertEqual(tracker.discard, set())

    def test_bito_discards_table_without_changing_hand(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("6C", "hand"), card("8H", "table"))))
        tracker.observe(FrameObservation(events=("OPPONENT_DEFENDED",)))
        self.assertEqual(tracker.hand, {"6C"})
        self.assertEqual(tracker.table, set())
        self.assertEqual(tracker.discard, {"8H"})

    def test_repeated_event_is_processed_once(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("6C", "hand"), card("8H", "table"))))
        first = tracker.observe(FrameObservation(events=("PLAYER_TAKES",)))
        second = tracker.observe(FrameObservation(
            (card("8H", "table"),), events=("PLAYER_TAKES",)
        ))
        self.assertTrue(first.changes)
        self.assertFalse(second.changes)
        self.assertEqual(tracker.table, set())

    def test_new_visible_card_is_added_as_draw(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("6C", "hand"),)))
        tracker.observe(FrameObservation((card("6C", "hand"), card("AS", "hand"))))
        self.assertEqual(tracker.hand, {"6C"})
        tracker.observe(FrameObservation((card("6C", "hand"), card("AS", "hand"))))
        self.assertEqual(tracker.hand, {"6C", "AS"})

    def test_one_frame_false_draw_is_not_remembered(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("7C", "hand"),)))
        tracker.observe(FrameObservation((card("6C", "hand"),)))
        tracker.observe(FrameObservation((card("7C", "hand"),)))
        self.assertEqual(tracker.hand, {"7C"})

    def test_does_not_add_new_card_during_active_bout(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("8C", "hand"),)))
        tracker.table = {"7H"}
        tracker.table_attack = ["7H"]
        for _ in range(3):
            tracker.observe(FrameObservation((card("8C", "hand"), card("10H", "hand"))))
        self.assertEqual(tracker.hand, {"8C"})

    def test_stable_post_bout_snapshot_removes_false_memory(self):
        tracker = GameStateTracker()
        tracker.hand = {"8C", "10H", "JS"}
        tracker.initialized = True
        tracker.observe(FrameObservation(events=("OPPONENT_DEFENDED",)))
        snapshot = FrameObservation((card("8C", "hand"), card("AH", "hand")))
        tracker.observe(snapshot)
        update = tracker.observe(snapshot)
        self.assertEqual(tracker.hand, {"8C", "AH"})
        self.assertIn("hand reconciled: removed 10H JS; added AH", update.changes)

    def test_infers_attack_and_defense_cards(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("6C", "hand"), card("8H", "hand"))))
        tracker.observe(FrameObservation((card("8H", "table"),)))
        tracker.observe(FrameObservation((card("9H", "table"), card("8H", "table"))))
        self.assertTrue(tracker.player_attacker)
        self.assertEqual(tracker.table_attack, ["8H"])
        self.assertEqual(tracker.table_defense, ["9H"])
        self.assertEqual(tracker.last_event_kind, 2)
        self.assertEqual(tracker.last_event_cards, ("9H",))
        self.assertFalse(tracker.last_event_actor_self)

    def test_live_event_memory_records_own_opening(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("6C", "hand"), card("8H", "hand"))))
        tracker.observe(FrameObservation((card("8H", "table"),)))
        self.assertEqual(tracker.last_event_kind, 0)
        self.assertEqual(tracker.last_event_cards, ("8H",))
        self.assertTrue(tracker.last_event_actor_self)

    def test_next_attacker_after_terminal_events(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("6C", "hand"),)))
        tracker.observe(FrameObservation(events=("PLAYER_DEFENDED",)))
        self.assertTrue(tracker.player_attacker)

    def test_transfer_swaps_roles(self):
        tracker = GameStateTracker()
        tracker.observe(FrameObservation((card("8D", "hand"), card("6C", "hand"))))
        tracker.observe(FrameObservation((card("8H", "table"),)))
        self.assertFalse(tracker.player_attacker)
        tracker.observe(FrameObservation((card("8H", "table"), card("8D", "table"))))
        self.assertTrue(tracker.player_attacker)
        self.assertEqual(tracker.table_attack, ["8H", "8D"])
        self.assertEqual(tracker.table_defense, [])

    def test_reclassification_does_not_add_a_physical_card(self):
        tracker = GameStateTracker()
        tracker.hand = {"KS"}
        tracker.initialized = True
        tracker.player_attacker = False
        tracker.table = {"JH", "QH", "JS", "QC"}
        tracker.table_attack = ["JH", "JS"]
        tracker.table_defense = ["QH", "QC"]
        update = tracker.observe(FrameObservation(tuple(
            card(name, "table") for name in ("JH", "QH", "JC", "QC", "JD")
        )))
        self.assertIn("table corrected: JS -> JC", update.changes)
        self.assertEqual(tracker.table, {"JH", "QH", "JC", "QC", "JD"})
        self.assertEqual(tracker.table_attack, ["JH", "JC", "JD"])

    def test_rejects_more_defenses_than_attacks(self):
        tracker = GameStateTracker()
        tracker.hand = {"AD"}
        tracker.initialized = True
        tracker.player_attacker = True
        tracker.table = {"6C", "AC"}
        tracker.table_attack = ["6C"]
        tracker.table_defense = ["AC"]
        update = tracker.observe(FrameObservation((
            card("6C", "table"), card("AC", "table"), card("JD", "table")
        )))
        self.assertIn("ignored impossible extra defense: JD", update.changes)
        self.assertNotIn("JD", tracker.table)


if __name__ == "__main__":
    unittest.main()
