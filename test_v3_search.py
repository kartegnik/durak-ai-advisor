import random
import unittest

from durakgame import Card, CardValue, Player, Suit, Table, standardDeck
from durakgame.game import Game
from durak_v3.environment import GAME_CONFIG, shuffled_deck
from durak_v3.model import RecurrentActorCritic
from durak_v3.agents import HeuristicController, SearchController
from durak_v3.environment import play_episode
from durak_v3.observation import BeliefMemory
from durak_v3.search import InformationSetMCTS, determinize, root_from_game


class V3SearchTest(unittest.TestCase):
    def setUp(self):
        self.game = Game(shuffled_deck(555), Player(), Player(), config=GAME_CONFIG)
        self.root = root_from_game(self.game, 1, BeliefMemory())

    def test_determinization_respects_public_partition(self):
        sampled = determinize(self.root, random.Random(9))
        self.assertEqual(set(sampled.player1.hand), set(self.game.player1.hand))
        self.assertFalse(set(sampled.player1.hand) & set(sampled.player2.hand))
        self.assertEqual(len(sampled.player2.hand), len(self.game.player2.hand))
        self.assertEqual(len(sampled.deck), len(self.game.deck))

    def test_search_returns_a_legal_root_action(self):
        model = RecurrentActorCritic(hidden_size=32, option_size=16)
        options = self.game.generateOptions()
        search = InformationSetMCTS(model, simulations=8, time_limit_ms=1000, max_depth=8)
        result = search.search(self.root, options)
        self.assertIn(result.action_index, range(len(options)))
        self.assertGreater(result.simulations, 0)
        self.assertEqual(sum(result.visits), result.simulations)
        self.assertGreater(result.mean_quiescence_steps, 0.0)
        self.assertEqual(result.unique_determinizations, 3)

    def test_tree_is_reused_for_the_same_information_state(self):
        options = self.game.generateOptions()
        search = InformationSetMCTS(
            RecurrentActorCritic(hidden_size=32, option_size=16),
            simulations=4, time_limit_ms=1000, max_depth=8,
        )
        first = search.search(self.root, options)
        second = search.search(self.root, options)
        self.assertEqual(first.reused_root_visits, 0)
        self.assertGreaterEqual(second.reused_root_visits, first.simulations)
        self.assertGreater(sum(second.visits), sum(first.visits))

    def test_small_empty_deck_endgame_is_solved_exactly(self):
        game = Game(shuffled_deck(1717), Player(), Player(), config=GAME_CONFIG)
        first = Card(CardValue.Six, Suit.Heart)
        second = Card(CardValue.Seven, Suit.Cross)
        game.player1.hand = [first]
        game.player2.hand = [second]
        game.deck = []
        game.table = Table()
        game.discardPile = set(standardDeck()) - {first, second}
        root = root_from_game(game, 1, BeliefMemory())
        options = game.generateOptions()
        search = InformationSetMCTS(
            RecurrentActorCritic(hidden_size=32, option_size=16),
            simulations=8, time_limit_ms=1000,
        )
        result = search.search(root, options)
        self.assertTrue(result.solved_exactly)
        self.assertEqual(result.values, (1.0,))
        self.assertEqual(sum(result.visits), result.simulations)

    def test_search_controller_finishes_a_game(self):
        model = RecurrentActorCritic(hidden_size=32, option_size=16)
        episode = play_episode(
            SearchController(model, simulations=4, time_limit_ms=100),
            HeuristicController(), shuffled_deck(777),
        )
        self.assertLess(episode.actions, 300)


if __name__ == "__main__":
    unittest.main()
