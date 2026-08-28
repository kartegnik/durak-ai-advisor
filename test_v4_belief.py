import random
import unittest

import torch

from durakgame import Player
from durakgame.game import Game
from durak_v3.agents import HeuristicController, RecurrentController
from durak_v3.environment import GAME_CONFIG, play_episode, shuffled_deck
from durak_v3.model import RecurrentActorCritic
from durak_v3.observation import BeliefMemory, encode_observation, encode_privileged, public_view_from_game
from durak_v3.ppo import ppo_update, prepare_trajectory
from durak_v3.search import determinize, root_from_game
from durak_v4.belief import belief_cross_entropy, belief_targets, hidden_hand_recall
from durak_v4.model import BeliefActorCritic, DECK, OPPONENT


class V4BeliefTest(unittest.TestCase):
    def setUp(self):
        self.game = Game(shuffled_deck(515), Player(), Player(), config=GAME_CONFIG)

    def test_v3_weights_load_without_changing_policy(self):
        torch.manual_seed(4)
        v3 = RecurrentActorCritic(hidden_size=32, option_size=16)
        v4 = BeliefActorCritic(hidden_size=32, option_size=16)
        v4.load_v3_state_dict(v3.state_dict())
        view = public_view_from_game(self.game, 1, BeliefMemory())
        observation = encode_observation(view)
        from state import encode_options_tensor
        options = encode_options_tensor(self.game.generateOptions())
        with torch.no_grad():
            logits3, value3, hidden3 = v3.forward_step(
                observation, options, privileged=encode_privileged(self.game, 1),
            )
            logits4, value4, hidden4 = v4.forward_step(
                observation, options, privileged=encode_privileged(self.game, 1),
            )
            _, belief_hidden = v4.belief_step(observation)
            logits4_belief, _, _ = v4.forward_step(
                observation, options, privileged=encode_privileged(self.game, 1),
                belief_hidden=belief_hidden,
            )
        self.assertTrue(torch.equal(logits3, logits4))
        self.assertTrue(torch.equal(value3, value4))
        self.assertTrue(torch.equal(hidden3, hidden4))
        self.assertTrue(torch.equal(logits3, logits4_belief))

    def test_belief_targets_and_loss_cover_hidden_partition(self):
        model = BeliefActorCritic(hidden_size=32, option_size=16)
        observation = encode_observation(public_view_from_game(self.game, 1, BeliefMemory()))
        privileged = encode_privileged(self.game, 1)
        logits, _ = model.belief_step(observation)
        targets = belief_targets(privileged)
        self.assertEqual(int((targets == OPPONENT).sum()), len(self.game.player2.hand))
        self.assertEqual(int((targets == DECK).sum()), len(self.game.deck))
        self.assertGreater(float(belief_cross_entropy(
            logits, observation, privileged,
        ).detach()), 0.0)

    def test_belief_auxiliary_ppo_updates_prediction_head(self):
        torch.manual_seed(8)
        model = BeliefActorCritic(hidden_size=32, option_size=16)
        learner = RecurrentController(model, deterministic=False, record=True)
        episode = play_episode(learner, HeuristicController(), shuffled_deck(616))
        trajectory = prepare_trajectory(learner.trajectory, 0.0)
        before = model.belief_head[-1].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = ppo_update(
            model, optimizer, [trajectory], epochs=1, belief_coefficient=0.2,
        )
        self.assertGreater(metrics["belief"], 0.0)
        self.assertFalse(torch.equal(before, model.belief_head[-1].weight))

    def test_weighted_determinization_uses_predicted_opponent_cards(self):
        root = root_from_game(self.game, 1, BeliefMemory())
        public = set(root.hand) | set(root.attack) | set(root.defense) | set(root.discard)
        target = next(card for card in sorted(set(self.game.deck) - public, key=str))
        from state import CARD_TO_INDEX, NUM_CARDS
        weights = torch.ones(NUM_CARDS)
        weights[CARD_TO_INDEX[target]] = 1e9
        appearances = sum(
            target in determinize(root, random.Random(seed), weights).player2.hand
            for seed in range(50)
        )
        self.assertGreaterEqual(appearances, 40)

    def test_hidden_hand_recall_reports_uniform_baseline(self):
        model = BeliefActorCritic(hidden_size=32, option_size=16)
        observation = encode_observation(public_view_from_game(self.game, 1, BeliefMemory()))
        privileged = encode_privileged(self.game, 1)
        recall, baseline, positions = hidden_hand_recall(
            model.belief_step(observation)[0], observation, privileged,
        )
        self.assertEqual(positions, 1)
        self.assertGreaterEqual(recall, 0.0)
        self.assertGreater(baseline, 0.0)
        self.assertLess(baseline, 1.0)


if __name__ == "__main__":
    unittest.main()
