package ru.kartegnik.durakadvisor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Reconstructs public game state from a sequence of card detections. */
final class GameStateTracker {
    enum Zone {
        OPPONENT,
        TABLE,
        HAND,
    }

    static final class ObservedCard {
        final String card;
        final Zone zone;
        final float classConfidence;
        final float boxConfidence;
        final float centerX;
        final float centerY;

        ObservedCard(
                String card, Zone zone, float classConfidence, float boxConfidence,
                float centerX, float centerY) {
            this.card = card;
            this.zone = zone;
            this.classConfidence = classConfidence;
            this.boxConfidence = boxConfidence;
            this.centerX = centerX;
            this.centerY = centerY;
        }

        boolean reliable() {
            return classConfidence >= 0.65f && boxConfidence >= 0.30f;
        }
    }

    private final Set<String> hand = new LinkedHashSet<>();
    private final Set<String> table = new LinkedHashSet<>();
    private final List<String> attacks = new ArrayList<>();
    private final List<String> defenses = new ArrayList<>();
    private final Set<String> discard = new LinkedHashSet<>();
    private final Set<String> knownOpponent = new LinkedHashSet<>();
    private final Map<String, Integer> drawCandidates = new HashMap<>();
    private Set<String> initialCandidate = Set.of();
    private Set<String> refreshCandidate = Set.of();
    private int initialCandidateFrames;
    private int refreshCandidateFrames;
    private int refreshAge;
    private int absentTableFrames;
    private int disjointTableFrames;
    private boolean initialized;
    private boolean awaitingRefreshedHand;
    private Boolean playerAttacker;
    private int deckCount = 24;
    private int opponentCount = 6;
    private int expectedHandCount;
    private Integer lastEventKind;
    private List<String> lastEventCards = List.of();
    private Boolean lastEventActorSelf;

    void reset() {
        hand.clear();
        table.clear();
        attacks.clear();
        defenses.clear();
        discard.clear();
        knownOpponent.clear();
        drawCandidates.clear();
        initialCandidate = Set.of();
        refreshCandidate = Set.of();
        initialCandidateFrames = 0;
        refreshCandidateFrames = 0;
        refreshAge = 0;
        absentTableFrames = 0;
        disjointTableFrames = 0;
        initialized = false;
        awaitingRefreshedHand = false;
        playerAttacker = null;
        deckCount = 24;
        opponentCount = 6;
        expectedHandCount = 0;
        lastEventKind = null;
        lastEventCards = List.of();
        lastEventActorSelf = null;
    }

    void observe(List<ObservedCard> observed) {
        Set<String> visibleHand = new LinkedHashSet<>();
        List<ObservedCard> visibleTableItems = new ArrayList<>();
        for (ObservedCard card : observed) {
            if (!card.reliable()) {
                continue;
            }
            if (card.zone == Zone.HAND) {
                visibleHand.add(card.card);
            } else if (card.zone == Zone.TABLE) {
                visibleTableItems.add(card);
            }
        }
        visibleTableItems.sort(Comparator
                .comparingDouble((ObservedCard card) -> card.centerY)
                .thenComparingDouble(card -> card.centerX));
        Set<String> visibleTable = new LinkedHashSet<>();
        for (ObservedCard card : visibleTableItems) {
            visibleTable.add(card.card);
        }

        if (!initialized) {
            stabilizeInitialHand(visibleHand);
            if (!initialized) {
                return;
            }
        }

        if (!table.isEmpty()) {
            Set<String> overlap = new HashSet<>(visibleTable);
            overlap.retainAll(table);
            if (visibleTable.isEmpty()) {
                absentTableFrames++;
                disjointTableFrames = 0;
                if (absentTableFrames >= 2) {
                    settleBout();
                }
            } else if (overlap.isEmpty()) {
                disjointTableFrames++;
                absentTableFrames = 0;
                if (disjointTableFrames >= 2) {
                    settleBout();
                } else {
                    observeHand(visibleHand);
                    return;
                }
            } else {
                absentTableFrames = 0;
                disjointTableFrames = 0;
            }
        }

        processNewTableCards(visibleTableItems);
        observeHand(visibleHand);
    }

    private void stabilizeInitialHand(Set<String> visibleHand) {
        if (visibleHand.isEmpty()) {
            initialCandidate = Set.of();
            initialCandidateFrames = 0;
            return;
        }
        if (visibleHand.equals(initialCandidate)) {
            initialCandidateFrames++;
        } else {
            initialCandidate = Set.copyOf(visibleHand);
            initialCandidateFrames = 1;
        }
        if (initialCandidateFrames >= 2) {
            hand.addAll(visibleHand);
            initialized = true;
        }
    }

    private void processNewTableCards(List<ObservedCard> visibleTableItems) {
        Set<String> oldHand = new HashSet<>(hand);
        boolean tableWasEmpty = table.isEmpty();
        for (ObservedCard item : visibleTableItems) {
            if (table.contains(item.card)) {
                continue;
            }
            boolean playedByPlayer = oldHand.contains(item.card);
            if (playerAttacker == null) {
                playerAttacker = playedByPlayer;
            }
            boolean isAttack = playedByPlayer == playerAttacker;
            boolean transfer = !isAttack && !attacks.isEmpty() && defenses.isEmpty()
                    && rank(item.card).equals(rank(attacks.get(0)));
            if (transfer) {
                isAttack = true;
                playerAttacker = !playerAttacker;
            }
            if (!isAttack && defenses.size() >= attacks.size()) {
                continue;
            }
            if (isAttack) {
                attacks.add(item.card);
            } else {
                defenses.add(item.card);
            }
            table.add(item.card);
            if (playedByPlayer) {
                hand.remove(item.card);
            } else {
                knownOpponent.remove(item.card);
                opponentCount = Math.max(0, opponentCount - 1);
            }
            rememberEvent(
                    transfer ? 6 : isAttack && tableWasEmpty ? 0 : isAttack ? 1 : 2,
                    List.of(item.card), playedByPlayer);
            tableWasEmpty = false;
        }
    }

    private void observeHand(Set<String> visibleHand) {
        if (awaitingRefreshedHand) {
            refreshAge++;
            if (!visibleHand.isEmpty() && visibleHand.equals(refreshCandidate)) {
                refreshCandidateFrames++;
            } else {
                refreshCandidate = Set.copyOf(visibleHand);
                refreshCandidateFrames = visibleHand.isEmpty() ? 0 : 1;
            }
            boolean complete = visibleHand.size() >= expectedHandCount;
            if ((complete && refreshCandidateFrames >= 2) || refreshAge >= 6) {
                if (complete) {
                    hand.clear();
                }
                hand.addAll(visibleHand);
                awaitingRefreshedHand = false;
                refreshCandidate = Set.of();
                refreshCandidateFrames = 0;
                drawCandidates.clear();
            }
            return;
        }

        Set<String> possibleDraws = new HashSet<>(visibleHand);
        possibleDraws.removeAll(hand);
        possibleDraws.removeAll(table);
        possibleDraws.removeAll(discard);
        Map<String, Integer> updated = new HashMap<>();
        for (String card : possibleDraws) {
            int confirmations = drawCandidates.getOrDefault(card, 0) + 1;
            if (confirmations >= 2) {
                hand.add(card);
            } else {
                updated.put(card, confirmations);
            }
        }
        drawCandidates.clear();
        drawCandidates.putAll(updated);
    }

    private void settleBout() {
        boolean fullyDefended = !attacks.isEmpty() && defenses.size() >= attacks.size();
        boolean previousPlayerAttacker = Boolean.TRUE.equals(playerAttacker);
        if (fullyDefended) {
            discard.addAll(table);
            playerAttacker = !previousPlayerAttacker;
            rememberEvent(4, List.of(), previousPlayerAttacker);
        } else {
            if (previousPlayerAttacker) {
                knownOpponent.addAll(table);
                opponentCount += table.size();
            } else {
                hand.addAll(table);
            }
            // After a take the same attacker opens the next bout.
            playerAttacker = previousPlayerAttacker;
            rememberEvent(5, List.of(), previousPlayerAttacker);
        }
        table.clear();
        attacks.clear();
        defenses.clear();
        absentTableFrames = 0;
        disjointTableFrames = 0;
        refillCounts(previousPlayerAttacker);
        awaitingRefreshedHand = true;
        refreshAge = 0;
        refreshCandidate = Set.of();
        refreshCandidateFrames = 0;
    }

    private void refillCounts(boolean playerDrewFirst) {
        int ownDraw = Math.max(0, 6 - hand.size());
        int opponentDraw = Math.max(0, 6 - opponentCount);
        if (playerDrewFirst) {
            int actualOwn = Math.min(deckCount, ownDraw);
            deckCount -= actualOwn;
            expectedHandCount = hand.size() + actualOwn;
            int actualOpponent = Math.min(deckCount, opponentDraw);
            deckCount -= actualOpponent;
            opponentCount += actualOpponent;
        } else {
            int actualOpponent = Math.min(deckCount, opponentDraw);
            deckCount -= actualOpponent;
            opponentCount += actualOpponent;
            int actualOwn = Math.min(deckCount, ownDraw);
            deckCount -= actualOwn;
            expectedHandCount = hand.size() + actualOwn;
        }
    }

    private void rememberEvent(int kind, Collection<String> cards, Boolean actorSelf) {
        lastEventKind = kind;
        lastEventCards = List.copyOf(cards);
        lastEventActorSelf = actorSelf;
    }

    boolean initialized() {
        return initialized;
    }

    boolean readyForAdvice() {
        return initialized && !awaitingRefreshedHand;
    }

    Set<String> hand() {
        return Set.copyOf(hand);
    }

    Set<String> table() {
        return Set.copyOf(table);
    }

    List<String> attacks() {
        return List.copyOf(attacks);
    }

    List<String> defenses() {
        return List.copyOf(defenses);
    }

    Set<String> discard() {
        return Set.copyOf(discard);
    }

    Set<String> knownOpponent() {
        return Set.copyOf(knownOpponent);
    }

    Boolean playerAttacker() {
        return playerAttacker;
    }

    int deckCount() {
        return deckCount;
    }

    int opponentCount() {
        return Math.max(knownOpponent.size(), opponentCount);
    }

    Integer lastEventKind() {
        return lastEventKind;
    }

    List<String> lastEventCards() {
        return lastEventCards;
    }

    Boolean lastEventActorSelf() {
        return lastEventActorSelf;
    }

    private static String rank(String card) {
        return card.substring(0, card.length() - 1);
    }
}
