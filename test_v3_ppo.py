import unittest

import torch

from durak_v3.agents import HeuristicController, RecurrentController
from durak_v3.environment import play_episode, result_for_seat, shuffled_deck
from durak_v3.model import RecurrentActorCritic
from durak_v3.ppo import ppo_update, prepare_trajectory
from durakgame import GameResult


class V3PPOTest(unittest.TestCase):
    def test_rollout_records_bounded_tactical_rewards(self):
        torch.manual_seed(3)
        model = RecurrentActorCritic(hidden_size=32, option_size=16)
        learner = RecurrentController(model, deterministic=False, record=True)
        play_episode(learner, HeuristicController(), shuffled_deck(17))
        self.assertTrue(all(step.reward <= 0.0 for step in learner.trajectory))
        self.assertTrue(all(step.reward >= -0.5 for step in learner.trajectory))
        self.assertTrue(all(step.tactical_rewards is not None for step in learner.trajectory))

    def test_recurrent_ppo_update_changes_parameters(self):
        torch.manual_seed(7)
        model = RecurrentActorCritic(hidden_size=32, option_size=16)
        learner = RecurrentController(model, deterministic=False, record=True)
        episode = play_episode(learner, HeuristicController(), shuffled_deck(91))
        result = result_for_seat(episode.result, 1)
        reward = 1.0 if result == GameResult.Win else -1.0 if result == GameResult.Loss else 0.0
        trajectory = prepare_trajectory(learner.trajectory, reward)
        before = model.policy[-1].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = ppo_update(model, optimizer, [trajectory], epochs=1)
        self.assertNotEqual(metrics["loss"], 0.0)
        self.assertFalse(torch.equal(before, model.policy[-1].weight))


if __name__ == "__main__":
    unittest.main()
