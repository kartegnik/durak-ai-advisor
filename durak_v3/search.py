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
    unique_determinizations: int = 0
    tree_nodes: int = 0
    reused_root_visits: int = 0
    mean_leaf_depth: float = 0.0
    mean_quiescence_steps: float = 0.0
    solved_exactly: bool = False


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


def _dangerous_weights(root: SearchRoot, opponent_weights=None) -> list[float]:
    """Bias one stratum toward plausible cards that are tactically dangerous."""
    result = []
    for card in CARD_TO_INDEX:
        base = 1.0 if opponent_weights is None else float(
            opponent_weights[CARD_TO_INDEX[card]]
        )
        rank = CARD_TO_INDEX[card] % 9
        danger = 1.0 + rank / 8.0 + 1.5 * (card.suit == root.trump_suit)
        if root.attack:
            danger += 2.5 * any(
                (card.suit == root.trump_suit and attack.suit != root.trump_suit)
                or (card.suit == attack.suit and card.value > attack.value)
                for attack in root.attack[len(root.defense):]
            )
        result.append(max(1e-6, base * danger))
    return result


def _constraint_adjusted_weights(root: SearchRoot, opponent_weights=None) -> list[float]:
    """Apply reversible evidence from attacks the opponent chose not to beat."""
    result = []
    for card in CARD_TO_INDEX:
        weight = 1.0 if opponent_weights is None else float(
            opponent_weights[CARD_TO_INDEX[card]]
        )
        for attack in root.memory.opponent_declined_attacks:
            could_cover = (
                card.suit == root.trump_suit and attack.suit != root.trump_suit
            ) or (card.suit == attack.suit and card.value > attack.value)
            if could_cover:
                weight *= 0.35
        result.append(max(1e-6, weight))
    return result


def _rare_weights(opponent_weights) -> list[float] | None:
    if opponent_weights is None:
        return None
    return [
        1.0 / max(1e-3, float(opponent_weights[index]))
        for index in range(len(CARD_TO_INDEX))
    ]


def sample_determinizations(
    root: SearchRoot,
    count: int,
    rng: random.Random,
    opponent_weights=None,
) -> list[Game]:
    """Cover uniform, likely, dangerous, and low-belief hidden deals."""
    count = max(1, count)
    constrained = _constraint_adjusted_weights(root, opponent_weights)
    modes = (
        None,
        constrained,
        _dangerous_weights(root, constrained),
        _rare_weights(opponent_weights),
    )
    return [
        determinize(root, rng, modes[index % len(modes)])
        for index in range(count)
    ]


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
        tuple(sorted(CARD_TO_INDEX[card] for card in memory.known_self_public)),
        tuple(sorted(CARD_TO_INDEX[card] for card in memory.opponent_declined_attacks)),
        tuple(sorted(CARD_TO_INDEX[card] for card in memory.self_declined_attacks)),
        memory.last_kind, memory.last_actor_self,
        tuple(CARD_TO_INDEX[card] for card in memory.last_cards),
        len(game.deck), game.turn.value, game.move.value,
        game.table.isForfeited, game.table.defenderHasDefended,
    )


def full_state_key(game: Game) -> tuple:
    """Perfect-information key used only when an empty deck reveals all cards."""
    return (
        tuple(sorted(CARD_TO_INDEX[card] for card in game.player1.hand)),
        tuple(sorted(CARD_TO_INDEX[card] for card in game.player2.hand)),
        tuple(CARD_TO_INDEX[card] for card in game.deck),
        tuple(CARD_TO_INDEX[card] for card in game.table.attack),
        tuple(CARD_TO_INDEX[card] for card in game.table.defense),
        tuple(sorted(CARD_TO_INDEX[card] for card in game.discardPile)),
        game.turn.value, game.move.value,
        game.table.isForfeited, game.table.defenderHasDefended,
    )


class InformationSetMCTS:
    def __init__(
        self, model: RecurrentActorCritic, simulations: int = 128,
        time_limit_ms: int = 300, max_depth: int = 80,
        exploration: float = 1.35, seed: int = 20260827,
        determinizations: int | None = None,
        quiescence: bool = True,
        max_quiescence_steps: int = 24,
        habr_leaf_weight: float = 0.10,
        reuse_tree: bool = True,
        exact_endgame_cards: int = 10,
    ):
        self.model = model
        self.simulations = simulations
        self.time_limit_ms = time_limit_ms
        self.max_depth = max_depth
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.determinizations = determinizations
        self.quiescence = quiescence
        self.max_quiescence_steps = max_quiescence_steps
        self.habr_leaf_weight = habr_leaf_weight
        self.reuse_tree = reuse_tree
        self.exact_endgame_cards = exact_endgame_cards
        self.tree: dict[tuple, Node] = {}

    def reset(self) -> None:
        self.tree.clear()

    @staticmethod
    def _terminal_value(game: Game) -> float:
        return (
            1.0 if game.result == GameResult.Win
            else -1.0 if game.result == GameResult.Loss else 0.0
        )

    def _habr_value(self, game: Game) -> float:
        if self.habr_leaf_weight <= 0.0:
            return 0.0
        # Delayed import avoids coupling the core simulator to the optional
        # interpretable controller.
        from durak_habr import strategic_hand_value
        active = (
            set(game.player1.hand) | set(game.player2.hand) | set(game.deck)
            | set(game.table.attack) | set(game.table.defense)
        )
        first = strategic_hand_value(
            game.player1.hand, game.trump.suit,
            len(game.deck), len(game.player2.hand), active,
        )
        second = strategic_hand_value(
            game.player2.hand, game.trump.suit,
            len(game.deck), len(game.player1.hand), active,
        )
        return math.tanh((first - second) / 4000.0)

    def _blend_value(self, network_value: float, game: Game, seat: int) -> float:
        network_for_first = network_value * (1.0 if seat == 1 else -1.0)
        weight = self.habr_leaf_weight
        return (1.0 - weight) * network_for_first + weight * self._habr_value(game)

    def _evaluate_position(self, game: Game, memories, hidden) -> float:
        if game.isFinished:
            return self._terminal_value(game)
        seat = 1 if game.move == Switch.First else 2
        view = public_view_from_game(game, seat, memories[seat])
        observation = encode_observation(view)
        _, value, hidden[seat] = self.model.forward_step(
            observation, encode_options_tensor(game.generateOptions()), hidden[seat],
            encode_privileged(game, seat),
        )
        return self._blend_value(float(value.item()), game, seat)

    def _finish_bout(
        self, game: Game, memories, hidden,
        first_logits: torch.Tensor | None = None,
    ) -> tuple[float, int]:
        """Resolve tactical noise before asking the value head for a judgement."""
        steps = 0
        while (
            not game.isFinished and not game.table.isEmpty
            and steps < self.max_quiescence_steps
        ):
            seat = 1 if game.move == Switch.First else 2
            options = game.generateOptions()
            if first_logits is not None:
                logits = first_logits
                first_logits = None
            else:
                view = public_view_from_game(game, seat, memories[seat])
                observation = encode_observation(view)
                logits, _, hidden[seat] = self.model.forward_step(
                    observation, encode_options_tensor(options), hidden[seat],
                    encode_privileged(game, seat),
                )
            choice = int(logits.argmax().item())
            move = options[choice]
            table_before = copy.deepcopy(game.table)
            game.updateHistory(move)
            game.process(move)
            for observer_seat in (1, 2):
                memories[observer_seat].observe(
                    move, observer_seat == seat, table_before,
                )
            steps += 1
        return self._evaluate_position(game, memories, hidden), steps

    def _exact_heuristic(self, game: Game) -> float:
        return self._habr_value(game)

    def _solve_exact(
        self, game: Game, deadline: float,
        cache: dict[tuple, float], visiting: set[tuple],
        alpha: float = -2.0, beta: float = 2.0, depth: int = 0,
    ) -> tuple[float, bool]:
        if time.monotonic() >= deadline:
            return self._exact_heuristic(game), False
        if game.isFinished:
            return self._terminal_value(game), True
        if depth >= 160:
            return self._exact_heuristic(game), False
        key = full_state_key(game)
        if key in cache:
            return cache[key], True
        if key in visiting:
            # Repeating a complete state forever cannot force a win.
            return 0.0, True
        visiting.add(key)
        maximizing = game.move == Switch.First
        best = -2.0 if maximizing else 2.0
        exact = True
        cutoff = False
        options = game.generateOptions()
        for move in options:
            child = copy.deepcopy(game)
            child.updateHistory(move)
            child.process(move)
            value, child_exact = self._solve_exact(
                child, deadline, cache, visiting, alpha, beta, depth + 1,
            )
            exact &= child_exact
            if maximizing:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            if beta <= alpha or not exact:
                cutoff = beta <= alpha
                break
        visiting.remove(key)
        if exact and not cutoff:
            cache[key] = best
        return best, exact

    def _try_exact_endgame(
        self, root: SearchRoot, root_options, opponent_weights,
        deadline: float,
    ) -> SearchResult | None:
        live_cards = (
            len(root.hand) + root.opponent_count
            + len(root.attack) + len(root.defense)
        )
        if root.deck_count or live_cards > self.exact_endgame_cards:
            return None
        game = determinize(root, random.Random(0), opponent_weights)
        cache: dict[tuple, float] = {}
        values = []
        all_exact = True
        for move in root_options:
            child = copy.deepcopy(game)
            child.updateHistory(move)
            child.process(move)
            value, exact = self._solve_exact(
                child, deadline, cache, set(),
            )
            values.append(value)
            all_exact &= exact
            if not exact:
                return None
        best_value = max(values)
        best_indices = [
            index for index, value in enumerate(values)
            if abs(value - best_value) < 1e-9
        ]
        visits = [0] * len(root_options)
        for offset in range(self.simulations):
            visits[best_indices[offset % len(best_indices)]] += 1
        return SearchResult(
            action_index=best_indices[0],
            visits=tuple(visits), values=tuple(values),
            simulations=self.simulations,
            unique_determinizations=1,
            tree_nodes=len(cache), solved_exactly=all_exact,
        )

    def search(
        self, root: SearchRoot, root_options,
        root_hidden: torch.Tensor | None = None,
        opponent_weights=None,
        root_noise_fraction: float = 0.0,
        root_dirichlet_alpha: float = 0.3,
    ) -> SearchResult:
        tree = self.tree if self.reuse_tree else {}
        deadline = time.monotonic() + self.time_limit_ms / 1000.0
        exact_result = self._try_exact_endgame(
            root, root_options, opponent_weights,
            time.monotonic() + min(0.25, self.time_limit_ms / 5000.0),
        )
        if exact_result is not None:
            return exact_result
        # A failed exact attempt gets a fresh normal budget.
        deadline = time.monotonic() + self.time_limit_ms / 1000.0
        root_keys = [action_key(move) for move in root_options]
        root_probe = determinize(root, random.Random(0), opponent_weights)
        root_key = information_key(root_probe, 1, root.memory)
        reused_root_visits = sum(
            edge.visits for edge in tree.get(root_key, Node()).edges.values()
        )
        root_noise: dict[tuple, float] = {}
        if root_noise_fraction > 0.0 and root_keys:
            samples = [
                self.rng.gammavariate(root_dirichlet_alpha, 1.0)
                for _ in root_keys
            ]
            total = sum(samples)
            root_noise = {
                key: sample / total for key, sample in zip(root_keys, samples)
            }
        requested_determinizations = self.determinizations or max(
            1, round(math.sqrt(self.simulations)),
        )
        sampled_games = sample_determinizations(
            root, min(self.simulations, requested_determinizations),
            self.rng, opponent_weights,
        )
        completed = 0
        leaf_depth_total = 0
        quiescence_total = 0
        with torch.no_grad():
            while completed < self.simulations and (
                completed == 0 or time.monotonic() < deadline
            ):
                game = copy.deepcopy(sampled_games[completed % len(sampled_games)])
                memories = {
                    1: root.memory.clone(),
                    2: root.memory.opponent_view(),
                }
                hidden = {
                    1: root_hidden.clone() if root_hidden is not None else self.model.initial_hidden(),
                    2: self.model.initial_hidden(),
                }
                path: list[tuple[Node, tuple]] = []
                leaf_value = 0.0
                reached_leaf = False
                for depth in range(self.max_depth):
                    if game.isFinished:
                        leaf_value = self._terminal_value(game)
                        leaf_depth_total += depth
                        reached_leaf = True
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
                        key = action_key(move)
                        if depth == 0 and root_noise:
                            prior = (
                                (1.0 - root_noise_fraction) * prior
                                + root_noise_fraction * root_noise[key]
                            )
                        node.edges.setdefault(key, Edge(prior))
                    if is_new_node and path:
                        if self.quiescence and not game.table.isEmpty:
                            leaf_value, quiet_steps = self._finish_bout(
                                game, memories, hidden, logits,
                            )
                            quiescence_total += quiet_steps
                        else:
                            leaf_value = self._blend_value(
                                float(value.item()), game, seat,
                            )
                        leaf_depth_total += depth
                        reached_leaf = True
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

                if not reached_leaf:
                    if self.quiescence and not game.table.isEmpty:
                        leaf_value, quiet_steps = self._finish_bout(
                            game, memories, hidden,
                        )
                        quiescence_total += quiet_steps
                    else:
                        leaf_value = self._evaluate_position(game, memories, hidden)
                    leaf_depth_total += self.max_depth

                for node, key in path:
                    node.edges[key].visits += 1
                    node.edges[key].value_sum += leaf_value
                completed += 1

        root_node = tree.get(root_key, Node())
        visits = tuple(root_node.edges.get(key, Edge(0.0)).visits for key in root_keys)
        values = tuple(root_node.edges.get(key, Edge(0.0)).mean_value for key in root_keys)
        action_index = max(range(len(root_options)), key=lambda index: (visits[index], values[index]))
        return SearchResult(
            action_index, visits, values, completed,
            unique_determinizations=len(sampled_games),
            tree_nodes=len(tree),
            reused_root_visits=reused_root_visits,
            mean_leaf_depth=leaf_depth_total / max(1, completed),
            mean_quiescence_steps=quiescence_total / max(1, completed),
        )
