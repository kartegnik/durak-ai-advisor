"""Information-set PUCT search over hidden hands and deck orders."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field

import torch

from durakgame import Card, CardValue, GameConfiguration, GameResult, Player, Suit, Table
from durakgame.game import Game
from durakgame.history import GameHistory
from durakgame.primitives import Switch
from state import CARD_TO_INDEX, encode_options_tensor
from .environment import GAME_CONFIG
from .model import RecurrentActorCritic
from .observation import (
    BeliefMemory, PublicView, encode_observation, encode_privileged,
    move_cards, move_kind, public_view_from_game,
)


@dataclass(frozen=True)
class SearchRoot:
    hand: tuple[Card, ...]
    attack: tuple[Card, ...]
    defense: tuple[Card, ...]
    discard: frozenset[Card]
    known_opponent: frozenset[Card]
    trump_suit: Suit
    trump_card: Card | None
    deck_count: int
    opponent_count: int
    is_attacker: bool
    table_forfeited: bool = False
    defender_has_defended: bool = False
    memory: BeliefMemory = field(default_factory=BeliefMemory, compare=False)


@dataclass
class Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    edges: dict[tuple, Edge] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    action_index: int
    visits: tuple[int, ...]
    values: tuple[float, ...]
    simulations: int


def root_from_game(game: Game, seat: int, memory: BeliefMemory) -> SearchRoot:
    own = game.player1 if seat == 1 else game.player2
    opponent = game.player2 if seat == 1 else game.player1
    attacker_seat = 1 if game.turn == Switch.First else 2
    return SearchRoot(
        hand=tuple(own.hand), attack=tuple(game.table.attack),
        defense=tuple(game.table.defense), discard=frozenset(game.discardPile),
        known_opponent=frozenset(memory.known_opponent),
        trump_suit=game.trump.suit, trump_card=game.trump,
        deck_count=len(game.deck), opponent_count=len(opponent.hand),
        is_attacker=attacker_seat == seat,
        table_forfeited=game.table.isForfeited,
        defender_has_defended=game.table.defenderHasDefended,
        memory=memory.clone(),
    )


def _choose_trump(root: SearchRoot, remaining: set[Card], rng: random.Random) -> Card:
    if root.trump_card is not None:
        return root.trump_card
    candidates = [card for card in remaining if card.suit == root.trump_suit]
    if not candidates:
        # Only possible with inconsistent CV memory; preserve search availability.
        return Card(CardValue.Six, root.trump_suit)
    return rng.choice(candidates)


def _weighted_sample_without_replacement(
    cards: set[Card], count: int, weights, rng: random.Random,
) -> set[Card]:
    pool = list(cards)
    selected: set[Card] = set()
    for _ in range(min(count, len(pool))):
        card_weights = [
            max(1e-6, float(weights[CARD_TO_INDEX[card]])) for card in pool
        ]
        choice = rng.choices(range(len(pool)), weights=card_weights, k=1)[0]
        selected.add(pool.pop(choice))
    return selected


def determinize(root: SearchRoot, rng: random.Random, opponent_weights=None) -> Game:
    all_cards = set(CARD_TO_INDEX)
    public = set(root.hand) | set(root.attack) | set(root.defense) | set(root.discard)
    public |= set(root.known_opponent)
    remaining = all_cards - public

    trump = _choose_trump(root, remaining, rng)
    if root.deck_count > 0:
        remaining.discard(trump)
    opponent_needed = max(0, root.opponent_count - len(root.known_opponent))
    if opponent_needed > len(remaining):
        raise ValueError("inconsistent opponent card count")
    opponent_extra = (
        set(rng.sample(list(remaining), opponent_needed))
        if opponent_weights is None
        else _weighted_sample_without_replacement(
            remaining, opponent_needed, opponent_weights, rng,
        )
    )
    opponent_hand = list(root.known_opponent | opponent_extra)
    remaining -= opponent_extra

    deck_needed = root.deck_count - (1 if root.deck_count > 0 else 0)
    deck_pool = list(remaining)
    rng.shuffle(deck_pool)
    deck = deck_pool[:max(0, deck_needed)]
    if root.deck_count > 0:
        deck.append(trump)

    game = object.__new__(Game)
    game.deck = deck
    game.player1 = Player()
    game.player2 = Player()
    game.player1.hand = list(root.hand)
    game.player2.hand = opponent_hand
    game.turn = Switch.First if root.is_attacker else Switch.Second
    game.move = Switch.First
    game.trump = trump
    game.table = Table(
        attack=list(root.attack), defense=list(root.defense),
        isForfeited=root.table_forfeited,
        defenderHasDefended=root.defender_has_defended,
    )
    game.discardPile = set(root.discard)
    game.config = GAME_CONFIG
    game.history = GameHistory(list(root.hand), opponent_hand, list(deck))
    return game


def action_key(move) -> tuple:
    return (move_kind(move), tuple(sorted(CARD_TO_INDEX[card] for card in move_cards(move))))


def information_key(game: Game, seat: int, memory: BeliefMemory) -> tuple:
    own = game.player1 if seat == 1 else game.player2
    return (
        seat,
        tuple(sorted(CARD_TO_INDEX[card] for card in own.hand)),
        tuple(CARD_TO_INDEX[card] for card in game.table.attack),
        tuple(CARD_TO_INDEX[card] for card in game.table.defense),
        tuple(sorted(CARD_TO_INDEX[card] for card in game.discardPile)),
        tuple(sorted(CARD_TO_INDEX[card] for card in memory.known_opponent)),
        len(game.deck), game.turn.value, game.move.value,
        game.table.isForfeited, game.table.defenderHasDefended,
    )


class InformationSetMCTS:
    def __init__(
        self, model: RecurrentActorCritic, simulations: int = 128,
        time_limit_ms: int = 300, max_depth: int = 80,
        exploration: float = 1.35, seed: int = 20260827,
    ):
        self.model = model
        self.simulations = simulations
        self.time_limit_ms = time_limit_ms
        self.max_depth = max_depth
        self.exploration = exploration
        self.rng = random.Random(seed)

    def search(
        self, root: SearchRoot, root_options,
        root_hidden: torch.Tensor | None = None,
        opponent_weights=None,
    ) -> SearchResult:
        tree: dict[tuple, Node] = {}
        deadline = time.monotonic() + self.time_limit_ms / 1000.0
        root_keys = [action_key(move) for move in root_options]
        completed = 0
        with torch.no_grad():
            while completed < self.simulations and time.monotonic() < deadline:
                game = determinize(root, self.rng, opponent_weights)
                memories = {1: root.memory.clone(), 2: BeliefMemory()}
                hidden = {
                    1: root_hidden.clone() if root_hidden is not None else self.model.initial_hidden(),
                    2: self.model.initial_hidden(),
                }
                path: list[tuple[Node, tuple]] = []
                leaf_value = 0.0
                for depth in range(self.max_depth):
                    if game.isFinished:
                        leaf_value = (
                            1.0 if game.result == GameResult.Win
                            else -1.0 if game.result == GameResult.Loss else 0.0
                        )
                        break
                    seat = 1 if game.move == Switch.First else 2
                    options = game.generateOptions()
                    view = public_view_from_game(game, seat, memories[seat])
                    observation = encode_observation(view)
                    logits, value, hidden[seat] = self.model.forward_step(
                        observation, encode_options_tensor(options), hidden[seat],
                        encode_privileged(game, seat),
                    )
                    priors = logits.softmax(dim=0).tolist()
                    node_key = information_key(game, seat, memories[seat])
                    is_new_node = node_key not in tree
                    node = tree.setdefault(node_key, Node())
                    for move, prior in zip(options, priors):
                        node.edges.setdefault(action_key(move), Edge(prior))
                    if is_new_node and path:
                        # Evaluate the state reached by the preceding action.
                        leaf_value = float(value.item()) * (1.0 if seat == 1 else -1.0)
                        break
                    total_visits = sum(edge.visits for edge in node.edges.values())

                    candidates = []
                    allowed = set(root_keys) if depth == 0 else None
                    for index, move in enumerate(options):
                        key = action_key(move)
                        if allowed is not None and key not in allowed:
                            continue
                        edge = node.edges[key]
                        exploitation = edge.mean_value if seat == 1 else -edge.mean_value
                        bonus = self.exploration * edge.prior * math.sqrt(total_visits + 1) / (edge.visits + 1)
                        candidates.append((exploitation + bonus, index, key))
                    _, choice, chosen_key = max(candidates, key=lambda item: item[0])
                    path.append((node, chosen_key))
                    move = options[choice]
                    table_before = copy.deepcopy(game.table)
                    game.updateHistory(move)
                    game.process(move)
                    for observer_seat in (1, 2):
                        memories[observer_seat].observe(
                            move, observer_seat == seat, table_before,
                        )

                for node, key in path:
                    node.edges[key].visits += 1
                    node.edges[key].value_sum += leaf_value
                completed += 1

        root_game = determinize(root, random.Random(0), opponent_weights)
        root_node = tree.get(information_key(root_game, 1, root.memory), Node())
        visits = tuple(root_node.edges.get(key, Edge(0.0)).visits for key in root_keys)
        values = tuple(root_node.edges.get(key, Edge(0.0)).mean_value for key in root_keys)
        action_index = max(range(len(root_options)), key=lambda index: (visits[index], values[index]))
        return SearchResult(action_index, visits, values, completed)
