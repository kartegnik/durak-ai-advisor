import random
import unittest

from durak_v3.agents import HeuristicController, RandomController, RecurrentController
from durak_v3.environment import play_episode, shuffled_deck
from durak_v3.model import RecurrentActorCritic


class V3EnvironmentTest(unittest.TestCase):
    def test_seeded_episode_finishes_and_records_strategy_metrics(self):
        random.seed(1)
        result = play_episode(
            HeuristicController(), RandomController(), shuffled_deck(123),
        )
        self.assertLess(result.actions, 300)
        self.assertEqual(len(result.trump_plays_with_deck), 2)

    def test_recurrent_controller_records_honest_trajectory(self):
        model = RecurrentActorCritic(hidden_size=32, option_size=16)
        learner = RecurrentController(model, deterministic=False, record=True)
        play_episode(learner, HeuristicController(), shuffled_deck(321))
        self.assertGreater(len(learner.trajectory), 0)
        self.assertTrue(all(step.options.ndim == 2 for step in learner.trajectory))


if __name__ == "__main__":
    unittest.main()
