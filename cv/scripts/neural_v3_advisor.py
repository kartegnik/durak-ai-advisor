#!/usr/bin/env python3
"""Recurrent v3 recommendations with optional information-set search."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from durak_v3.model import RecurrentActorCritic
from durak_v3.observation import BeliefMemory, PublicView, encode_observation
from durak_v3.search import InformationSetMCTS, SearchRoot
from neural_move_advisor import (
    NeuralRecommendation, SUITS, action_to_move, to_card,
)
from state import encode_options_tensor


class NeuralV3Advisor:
    def __init__(
        self, checkpoint: Path, search_ms: int = 0,
        search_simulations: int = 128,
        model_class=RecurrentActorCritic,
    ):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = model_class(**saved.get("model_config", {}))
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.search = (
            InformationSetMCTS(
                self.model, simulations=search_simulations,
                time_limit_ms=search_ms,
            )
            if search_ms > 0 else None
        )
        self.reset()

    def reset(self) -> None:
        self.hidden = self.model.initial_hidden()
        self.belief_hidden = (
            self.model.initial_belief_hidden()
            if hasattr(self.model, "initial_belief_hidden") else None
        )
        self.last_signature = None
        self.last_result: NeuralRecommendation | None = None

    def recommend(self, tracker, advice, trump: str, deck_count: int) -> NeuralRecommendation:
        opponent_count = min(36, max(
            len(tracker.opponent_known),
            36 - deck_count - len(tracker.hand) - len(tracker.table) - len(tracker.discard),
        ))
        memory = BeliefMemory(
            known_opponent={to_card(card) for card in tracker.opponent_known},
            last_kind=tracker.last_event_kind,
            last_cards=tuple(to_card(card) for card in tracker.last_event_cards),
            last_actor_self=tracker.last_event_actor_self,
        )
        view = PublicView(
            hand=tuple(to_card(card) for card in tracker.hand),
            attack=tuple(to_card(card) for card in tracker.table_attack),
            defense=tuple(to_card(card) for card in tracker.table_defense),
            discard=frozenset(to_card(card) for card in tracker.discard),
            known_opponent=frozenset(memory.known_opponent),
            trump_suit=SUITS[trump].value,
            # Manual suit entry is reliable; the exact face-up rank is not yet.
            trump_rank=None,
            deck_count=deck_count,
            opponent_count=opponent_count,
            is_attacker=tracker.player_attacker is not False,
            table_forfeited=False,
            defender_has_defended=bool(tracker.table_defense),
            last_event=tuple(memory.event_vector()),
        )
        signature = (
            tuple(sorted(tracker.hand)), tuple(tracker.table_attack),
            tuple(tracker.table_defense), tuple(sorted(tracker.discard)),
            tuple(sorted(tracker.opponent_known)), trump, deck_count,
            tracker.player_attacker, tracker.last_event_kind,
            tracker.last_event_cards, tracker.last_event_actor_self,
            advice.actions,
        )
        if signature == self.last_signature and self.last_result is not None:
            return self.last_result

        moves = [action_to_move(action) for action in advice.actions]
        observation = encode_observation(view)
        hidden_before = self.hidden
        with torch.no_grad():
            if self.belief_hidden is not None:
                _, self.belief_hidden = self.model.belief_step(
                    observation, self.belief_hidden,
                )
            logits, _, self.hidden = self.model.forward_step(
                observation, encode_options_tensor(moves), self.hidden,
                belief_hidden=self.belief_hidden,
            )
            if self.belief_hidden is not None:
                opponent_weights = self.model.opponent_weights(self.belief_hidden)
            else:
                opponent_weights = None

        if self.search is not None and tracker.player_attacker is not None:
            root = SearchRoot(
                hand=view.hand, attack=view.attack, defense=view.defense,
                discard=view.discard, known_opponent=view.known_opponent,
                trump_suit=SUITS[trump], trump_card=None,
                deck_count=deck_count, opponent_count=opponent_count,
                is_attacker=view.is_attacker,
                defender_has_defended=view.defender_has_defended,
                memory=memory,
            )
            try:
                searched = self.search.search(
                    root, moves, hidden_before, opponent_weights,
                )
                values = [
                    visits / max(1, searched.simulations)
                    for visits in searched.visits
                ]
                selected = searched.action_index
            except (AssertionError, IndexError, ValueError):
                values = logits.tolist()
                selected = int(logits.argmax().item())
        else:
            values = logits.tolist()
            selected = int(logits.argmax().item())

        ranked = tuple(sorted(zip(advice.actions, values), key=lambda item: item[1], reverse=True))
        self.last_signature = signature
        self.last_result = NeuralRecommendation(advice.actions[selected], ranked)
        return self.last_result
