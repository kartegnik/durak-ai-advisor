import copy
import unittest

from durakgame import Card, CardValue, OpeningMove, Player, Suit
from durakgame.game import Game
from durak_habr import HabrMemoryController, hand_value
from durak_v3.environment import GAME_CONFIG, shuffled_deck


class HabrMemoryAgentTest(unittest.TestCase):
    def test_article_rank_scale_and_trump_bonus(self):
        six = Card(CardValue.Six, Suit.Heart)
        ace = Card(CardValue.Ace, Suit.Heart)
        trump_six = Card(CardValue.Six, Suit.Spade)
        six_score = hand_value([six], Suit.Spade, 24, 6)
        ace_score = hand_value([ace], Suit.Spade, 24, 6)
        trump_score = hand_value([trump_six], Suit.Spade, 24, 6)
        self.assertGreater(ace_score, six_score)
        self.assertGreater(trump_score, ace_score)

    def test_opening_conserves_trump_when_nontrump_is_available(self):
        game = Game(shuffled_deck(812), Player(), Player(), config=GAME_CONFIG)
        trump = Card(CardValue.Six, game.trump.suit)
        nontrump = next(card for card in game.player1.hand if card.suit != game.trump.suit)
        game.player1.hand = [trump, nontrump]
        options = [OpeningMove(trump), OpeningMove(nontrump)]
        agent = HabrMemoryController()
        agent.reset(1)
        self.assertEqual(agent.choose(game, 1, options), 1)

    def test_decision_does_not_depend_on_hidden_opponent_cards(self):
        game = Game(shuffled_deck(913), Player(), Player(), config=GAME_CONFIG)
        other = copy.deepcopy(game)
        other.player2.hand = list(reversed(other.player2.hand))
        options = game.generateOptions()
        agent1 = HabrMemoryController()
        agent2 = HabrMemoryController()
        agent1.reset(1)
        agent2.reset(1)
        self.assertEqual(agent1.choose(game, 1, options), agent2.choose(other, 1, options))

    def test_seen_cards_change_probability_without_peeking(self):
        game = Game(shuffled_deck(1014), Player(), Player(), config=GAME_CONFIG)
        attack = next(card for card in game.player1.hand if card.suit != game.trump.suit)
        agent = HabrMemoryController()
        agent.reset(1)
        before = agent.opponent_can_beat_probability(game, 1, attack)
        beating = {
            card for card in agent.unseen_cards(game, 1)
            if card.suit == attack.suit and card.value > attack.value
        }
        game.discardPile.update(beating)
        after = agent.opponent_can_beat_probability(game, 1, attack)
        self.assertLessEqual(after, before)


if __name__ == "__main__":
    unittest.main()
