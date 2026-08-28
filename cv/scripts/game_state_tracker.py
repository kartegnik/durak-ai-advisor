#!/usr/bin/env python3
"""Persistent Durak game state reconstructed from consecutive CV observations."""

from __future__ import annotations

from dataclasses import dataclass, field


TERMINAL_EVENTS = {
    "PLAYER_TAKES",
    "OPPONENT_TAKES",
    "PLAYER_DEFENDED",
    "OPPONENT_DEFENDED",
}


@dataclass(frozen=True)
class ObservedCard:
    card: str
    zone: str
    class_confidence: float = 1.0
    box_confidence: float = 1.0
    x: float = 0.0
    y: float = 0.0

    @property
    def reliable(self) -> bool:
        return self.class_confidence >= 0.65 and self.box_confidence >= 0.20


@dataclass(frozen=True)
class FrameObservation:
    cards: tuple[ObservedCard, ...] = ()
    events: tuple[str, ...] = ()
    deck_count: int | None = None


@dataclass
class TrackerUpdate:
    changes: list[str] = field(default_factory=list)


class GameStateTracker:
    """Remember known cards instead of re-recognizing an overlapped hand."""

    def __init__(self, minimum_initial_hand: int = 1) -> None:
        self.hand: set[str] = set()
        self.table: set[str] = set()
        self.table_attack: list[str] = []
        self.table_defense: list[str] = []
        self.discard: set[str] = set()
        self.opponent_known: set[str] = set()
        self.player_attacker: bool | None = None
        self.initialized = False
        self.active_events: set[str] = set()
        self.draw_candidates: dict[str, int] = {}
        self.table_candidates: dict[str, int] = {}
        self.hand_reconcile_pending = False
        self.hand_reconcile_age = 0
        self.hand_snapshot_candidate: frozenset[str] = frozenset()
        self.hand_snapshot_count = 0
        self.last_deck_count: int | None = None
        self.empty_deck_grace = 0
        self.last_event_kind: int | None = None
        self.last_event_cards: tuple[str, ...] = ()
        self.last_event_actor_self: bool | None = None
        self.minimum_initial_hand = minimum_initial_hand

    def _remember_event(
        self, kind: int, cards=(), actor_self: bool | None = None,
    ) -> None:
        """Store the most recent inferred move using durak_v3 MOVE_KINDS indices."""
        self.last_event_kind = kind
        self.last_event_cards = tuple(cards)
        self.last_event_actor_self = actor_self

    def observe(self, observation: FrameObservation) -> TrackerUpdate:
        update = TrackerUpdate()
        if observation.deck_count == 0 and self.last_deck_count not in {None, 0}:
            # The last cards have just been dealt. Give the refreshed hand a
            # few frames to settle before enforcing the no-more-draws rule.
            self.empty_deck_grace = 3
        elif observation.deck_count not in {None, 0}:
            self.empty_deck_grace = 0
        deck_empty_locked = observation.deck_count == 0 and self.empty_deck_grace == 0
        current_events = set(observation.events)
        new_events = current_events - self.active_events
        visible_hand = {
            item.card for item in observation.cards
            if item.zone == "hand" and item.reliable
        }
        raw_table_items = sorted(
            (item for item in observation.cards if item.zone == "table"),
            key=lambda item: (item.y, item.x),
        )
        moderate_table = {
            item.card for item in raw_table_items
            if item.class_confidence >= 0.40 and item.box_confidence >= 0.30
        }
        self.table_candidates = {
            name: self.table_candidates.get(name, 0) + 1
            for name in moderate_table
        }
        confirmed_table = {
            name for name, confirmations in self.table_candidates.items()
            if confirmations >= 2
        }
        table_items = [
            item for item in raw_table_items
            if item.reliable or item.card in confirmed_table
        ]
        table_evidence = {
            item.card: item
            for item in sorted(
                table_items,
                key=lambda item: item.class_confidence * item.box_confidence,
            )
        }
        visible_table = {item.card for item in table_items}
        # During the take/discard animation the old cards can remain visible
        # for several frames. Once the terminal event was processed, never add
        # those pixels back as a new table until the bubble disappears.
        repeated_terminal = bool(current_events & TERMINAL_EVENTS) and not bool(
            new_events & TERMINAL_EVENTS
        )
        if repeated_terminal:
            visible_table.clear()
            table_items.clear()
            self.table_candidates.clear()

        if not self.initialized and len(visible_hand) >= self.minimum_initial_hand:
            self.hand = set(visible_hand)
            self.initialized = True
            update.changes.append("initial hand: " + " ".join(sorted(self.hand)))

        # After a completed bout (except when we took the table), the game fans
        # our refreshed hand out again. Two identical snapshots are safer than
        # carrying missed played cards forever, and remove false remembered
        # cards that were created from overlapped corners.
        if self.hand_reconcile_pending:
            self.hand_reconcile_age += 1
            snapshot = frozenset(visible_hand)
            if snapshot and len(snapshot) <= 6:
                if snapshot == self.hand_snapshot_candidate:
                    self.hand_snapshot_count += 1
                else:
                    self.hand_snapshot_candidate = snapshot
                    self.hand_snapshot_count = 1
                if self.hand_snapshot_count >= 2:
                    before = set(self.hand)
                    proposed = set(snapshot)
                    # With an empty deck, an unseen card cannot be a draw. A
                    # changed class name is therefore a crop error, not a real
                    # hand replacement (for example QD -> JD on an overlapped
                    # face card). Keep the remembered identities in that case;
                    # actual plays are removed from the table observation.
                    if deck_empty_locked and not proposed <= before:
                        proposed = before
                    self.hand = proposed
                    removed = before - self.hand
                    added = self.hand - before
                    details = []
                    if removed:
                        details.append("removed " + " ".join(sorted(removed)))
                    if added:
                        details.append("added " + " ".join(sorted(added)))
                    if details:
                        update.changes.append("hand reconciled: " + "; ".join(details))
                    self.hand_reconcile_pending = False
                    self.hand_snapshot_candidate = frozenset()
                    self.hand_snapshot_count = 0
                    self.draw_candidates.clear()
                    if observation.deck_count == 0:
                        self.empty_deck_grace = 0
            if self.hand_reconcile_age >= 5:
                self.hand_reconcile_pending = False
                self.hand_snapshot_candidate = frozenset()
                self.hand_snapshot_count = 0

        old_hand = set(self.hand)
        new_table = visible_table - self.table
        missing_table = self.table - visible_table
        # If the visible number of physical cards grew by N but there are more
        # than N new class names, some previous card was merely reclassified.
        # Prefer the very common correction where rank stays the same and only
        # the suit changes (for example JS -> JC on a narrow corner crop).
        physical_growth = max(0, len(visible_table) - len(self.table))
        corrections_needed = max(0, len(new_table) - physical_growth)
        corrections = []
        for old in sorted(missing_table):
            if corrections_needed <= 0:
                break
            same_rank = next((new for new in sorted(new_table) if new[:-1] == old[:-1]), None)
            if same_rank:
                # Temporal confirmation lets a moderately confident new card
                # join the table, but it must not rewrite an already known
                # card identity. Suit corrections require one independently
                # reliable observation; otherwise keep the remembered card.
                evidence = table_evidence.get(same_rank)
                if evidence is not None and evidence.reliable:
                    corrections.append((old, same_rank))
                new_table.remove(same_rank)
                missing_table.remove(old)
                corrections_needed -= 1
        for old, new in corrections:
            self.table.remove(old)
            self.table.add(new)
            self.table_attack = [new if card == old else card for card in self.table_attack]
            self.table_defense = [new if card == old else card for card in self.table_defense]
            update.changes.append(f"table corrected: {old} -> {new}")
        if new_table:
            table_was_empty = not self.table
            own_cards = {name for name in new_table if name in old_hand}
            opponent_cards = new_table - own_cards
            # Public cards played by the opponent are no longer in the subset
            # of their hand that we know exactly.
            self.opponent_known.difference_update(opponent_cards)
            if self.player_attacker is None:
                # The first card of a bout identifies who opened it.
                first = next(item.card for item in table_items if item.card in new_table)
                self.player_attacker = first in own_cards
            accepted_new = set()
            for item in table_items:
                if item.card not in new_table:
                    continue
                played_by_player = item.card in own_cards
                is_attack = played_by_player == self.player_attacker
                # A matching-rank card played before any defense is a transfer:
                # it joins the attacks and swaps attacker/defender roles.
                is_transfer = (
                    not is_attack
                    and bool(self.table_attack)
                    and not self.table_defense
                    and item.card[:-1] == self.table_attack[0][:-1]
                )
                if is_transfer:
                    is_attack = True
                    self.player_attacker = not self.player_attacker
                if not is_attack and len(self.table_defense) >= len(self.table_attack):
                    update.changes.append(f"ignored impossible extra defense: {item.card}")
                    continue
                target = self.table_attack if is_attack else self.table_defense
                target.append(item.card)
                accepted_new.add(item.card)
                event_kind = (
                    6 if is_transfer else
                    0 if is_attack and table_was_empty else
                    1 if is_attack else 2
                )
                self._remember_event(event_kind, (item.card,), played_by_player)
                table_was_empty = False
            self.table.update(accepted_new)
            if accepted_new:
                update.changes.append("table added: " + " ".join(sorted(accepted_new)))

        # A known card appearing on the table must have left our hand. This
        # covers attacks, defenses, throw-ins, and transfers uniformly.
        played = self.table & self.hand
        if played:
            self.hand.difference_update(played)
            update.changes.append("played: " + " ".join(sorted(played)))

        # Newly visible cards in a small hand are draws. Cards received after
        # taking are added from the remembered table below, even if overlapped.
        draws_allowed = (
            not deck_empty_locked
            and (not self.table or self.hand_reconcile_pending)
        )
        possible_draws = (
            visible_hand - self.hand - self.table - self.discard
            if draws_allowed else set()
        )
        self.draw_candidates = {
            name: self.draw_candidates.get(name, 0) + 1
            for name in possible_draws
        }
        # A narrow exposed corner is occasionally localized as a second card
        # (for example 7C -> 6C). New hand cards therefore need to agree in two
        # consecutive frames before they enter persistent memory.
        drawn = {
            name for name, confirmations in self.draw_candidates.items()
            if confirmations >= 2
        }
        if self.initialized and drawn:
            self.hand.update(drawn)
            for name in drawn:
                self.draw_candidates.pop(name, None)
            update.changes.append("recognized draw: " + " ".join(sorted(drawn)))
            if observation.deck_count == 0:
                self.empty_deck_grace = 0

        for event in observation.events:
            if event not in new_events:
                continue
            if event == "PLAYER_TAKES":
                self.draw_candidates.clear()
                self.table_candidates.clear()
                self.hand_reconcile_pending = False
                previous_attacker = self.player_attacker
                taken = set(self.table)
                self.hand.update(taken)
                self.table.clear()
                self.table_attack.clear()
                self.table_defense.clear()
                self.player_attacker = False
                # The attacker finishes the take after the defender forfeits.
                self._remember_event(5, (), previous_attacker is True)
                update.changes.append("player takes: " + (" ".join(sorted(taken)) or "empty table"))
            elif event in {"OPPONENT_TAKES", "PLAYER_DEFENDED", "OPPONENT_DEFENDED"}:
                self.draw_candidates.clear()
                self.table_candidates.clear()
                discarded = set(self.table)
                if event == "OPPONENT_TAKES":
                    self.opponent_known.update(discarded)
                if event in {"PLAYER_DEFENDED", "OPPONENT_DEFENDED"}:
                    self.discard.update(discarded)
                previous_attacker = self.player_attacker
                self.table.clear()
                self.table_attack.clear()
                self.table_defense.clear()
                self.player_attacker = event in {"OPPONENT_TAKES", "PLAYER_DEFENDED"}
                if event == "OPPONENT_TAKES":
                    self._remember_event(5, (), previous_attacker is True)
                else:
                    self._remember_event(4, (), previous_attacker is True)
                self.hand_reconcile_pending = True
                self.hand_reconcile_age = 0
                self.hand_snapshot_candidate = frozenset()
                self.hand_snapshot_count = 0
                update.changes.append(f"{event.lower()}: " + (" ".join(sorted(discarded)) or "empty table"))
            elif event in {"PLAYER_STOPPED_THROWING", "OPPONENT_STOPPED_THROWING"}:
                update.changes.append(event.lower())
        self.active_events = current_events
        if observation.deck_count is not None:
            self.last_deck_count = observation.deck_count
        if self.empty_deck_grace > 0:
            self.empty_deck_grace -= 1
        return update

    def summary(self) -> str:
        hand = " ".join(sorted(self.hand)) or "-"
        table = " ".join(sorted(self.table)) or "-"
        role = {True: "attack", False: "defense", None: "unknown"}[self.player_attacker]
        known = " ".join(sorted(self.opponent_known)) or "-"
        return (
            f"hand=[{hand}] table=[{table}] discard={len(self.discard)} "
            f"opponent_known=[{known}] role={role}"
        )
