import unittest

from durakgame import (
    Card, CardValue, DefensiveMove, FinishingMove, ForfeitingMove, OpeningMove,
    Suit, Table,
)
from durak_v3.model import RecurrentActorCritic
from durak_v3.observation import (
    BeliefMemory, OBSERVATION_DIM, PRIVILEGED_DIM, PublicView, encode_observation,
)
from state import OPTION_DIM
import torch


def card(value, suit):
    return Card(value, suit)


class V3ObservationTest(unittest.TestCase):
    def test_opponent_pickup_is_remembered_and_play_removes_card(self):
        six_h = card(CardValue.Six, Suit.Heart)
        seven_h = card(CardValue.Seven, Suit.Heart)
        table = Table(attack=[six_h], defense=[seven_h], isForfeited=True)
        memory = BeliefMemory()
        memory.observe(FinishingMove([]), actor_is_self=True, table_before=table)
        self.assertEqual(memory.known_opponent, {six_h, seven_h})
        memory.observe(OpeningMove(six_h), actor_is_self=False, table_before=Table())
        self.assertEqual(memory.known_opponent, {seven_h})

    def test_mirror_memory_tracks_what_opponent_knows_about_us(self):
        six_h = card(CardValue.Six, Suit.Heart)
        seven_h = card(CardValue.Seven, Suit.Heart)
        table = Table(attack=[six_h], defense=[seven_h], isForfeited=True)
        memory = BeliefMemory()
        memory.observe(FinishingMove([]), actor_is_self=False, table_before=table)
        self.assertEqual(memory.known_self_public, {six_h, seven_h})
        mirrored = memory.opponent_view()
        self.assertEqual(mirrored.known_opponent, {six_h, seven_h})
        memory.observe(OpeningMove(six_h), actor_is_self=True, table_before=Table())
        self.assertEqual(memory.known_self_public, {seven_h})

    def test_declined_defense_is_soft_public_evidence(self):
        six_h = card(CardValue.Six, Suit.Heart)
        eight_h = card(CardValue.Eight, Suit.Heart)
        table = Table(attack=[six_h])
        memory = BeliefMemory()
        memory.observe(ForfeitingMove(), actor_is_self=False, table_before=table)
        self.assertEqual(memory.opponent_declined_attacks, {six_h})
        memory.observe(DefensiveMove(eight_h), actor_is_self=False, table_before=table)
        self.assertFalse(memory.opponent_declined_attacks)

    def test_observation_and_model_shapes(self):
        six_h = card(CardValue.Six, Suit.Heart)
        view = PublicView(
            hand=(six_h,), attack=(), defense=(), discard=frozenset(),
            known_opponent=frozenset(), trump_suit=Suit.Spade.value,
            trump_rank=None, deck_count=24, opponent_count=6,
            is_attacker=True, table_forfeited=False,
            defender_has_defended=False,
            last_event=tuple(BeliefMemory().event_vector()),
        )
        observation = encode_observation(view)
        self.assertEqual(observation.shape, (OBSERVATION_DIM,))
        model = RecurrentActorCritic(hidden_size=32, option_size=16)
        logits, value, hidden = model.forward_step(
            observation, torch.zeros((3, OPTION_DIM)),
            privileged=torch.zeros(PRIVILEGED_DIM),
        )
        self.assertEqual(logits.shape, (3,))
        self.assertEqual(value.shape, ())
        self.assertEqual(hidden.shape, (32,))


if __name__ == "__main__":
    unittest.main()
