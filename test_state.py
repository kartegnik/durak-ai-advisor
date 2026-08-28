import unittest

import torch

from durakgame import Card, CardValue, Suit, TransferMove
from model import DurakNet
from state import CARD_TO_INDEX, COMBINED_DIM, MOVE_TYPE_DIM, STATE_DIM, encode_option


class TransferEncodingTest(unittest.TestCase):
    def test_transfer_encodes_type_and_card(self):
        card = Card(CardValue.Seven, Suit.Heart)

        encoded = encode_option(TransferMove(card))

        self.assertEqual(len(encoded), MOVE_TYPE_DIM + 36)
        self.assertEqual(encoded[:MOVE_TYPE_DIM], [0.0] * 6 + [1.0])
        self.assertEqual(sum(encoded[MOVE_TYPE_DIM:]), 1.0)
        self.assertEqual(encoded[MOVE_TYPE_DIM + CARD_TO_INDEX[card]], 1.0)

    def test_legacy_checkpoint_migration_preserves_columns(self):
        model = DurakNet()
        state_dict = model.state_dict()
        old_weight = torch.arange(256 * (COMBINED_DIM - 1), dtype=torch.float32).reshape(256, -1)
        state_dict["net.0.weight"] = old_weight

        migrated = model.load_compatible_state_dict(state_dict)
        new_weight = model.state_dict()["net.0.weight"]
        legacy_prefix = STATE_DIM + 6

        self.assertTrue(migrated)
        self.assertTrue(torch.equal(new_weight[:, :legacy_prefix], old_weight[:, :legacy_prefix]))
        self.assertTrue(torch.equal(new_weight[:, legacy_prefix], torch.zeros(256)))
        self.assertTrue(torch.equal(new_weight[:, legacy_prefix + 1:], old_weight[:, legacy_prefix:]))


if __name__ == "__main__":
    unittest.main()
