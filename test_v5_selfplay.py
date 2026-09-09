import unittest

import torch

from durakgame import Player
from durakgame.game import Game
from durak_v3.environment import GAME_CONFIG, shuffled_deck
from durak_v3.observation import BeliefMemory
from durak_v3.search import InformationSetMCTS, root_from_game
from durak_v4.model import BeliefActorCritic
from durak_v5.selfplay import SearchSelfPlayController
from durak_v5.training import alphazero_loss


class V5SelfPlayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(55)
        self.model = BeliefActorCritic(hidden_size=32, option_size=16)
        self.game = Game(shuffled_deck(1515), Player(), Player(), config=GAME_CONFIG)

    def test_noisy_search_still_returns_normalized_visits(self):
        options = self.game.generateOptions()
        search = InformationSetMCTS(
            self.model, simulations=8, time_limit_ms=1000, max_depth=8,
        )
        result = search.search(
            root_from_game(self.game, 1, BeliefMemory()), options,
            root_noise_fraction=0.25,
        )
        self.assertEqual(sum(result.visits), result.simulations)
        self.assertIn(result.action_index, range(len(options)))

    def test_controller_records_search_distribution_and_loss_backpropagates(self):
        controller = SearchSelfPlayController(
            self.model, simulations=6, time_limit_ms=1000,
        )
        controller.reset(1)
        options = self.game.generateOptions()
        action = controller.choose(self.game, 1, options)
        self.assertIn(action, range(len(options)))
        self.assertEqual(len(controller.steps), 1)
        self.assertAlmostEqual(float(controller.steps[0].policy_target.sum()), 1.0)
        loss, metrics = alphazero_loss(
            self.model, [controller.steps], [1.0], belief_coefficient=0.1,
        )
        loss.backward()
        self.assertGreater(metrics["policy"], 0.0)
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
