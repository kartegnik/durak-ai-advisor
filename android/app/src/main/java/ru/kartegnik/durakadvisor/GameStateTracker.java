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
    private Set<String> lastVisibleHand = Set.of();
    private Set<String> initialCandidate = Set.of();
    private Set<String> refreshCandidate = Set.of();
    private int initialCandidateFrames;
    private int refreshCandidateFrames;
    private int refreshAge;
    private int absentTableFrames;
    private boolean initialized;
    private boolean awaitingRefreshedHand;
    private Boolean playerAttacker;
    private int deckCount = 24;
    private int opponentCount = 6;
    private int expectedHandCount;
    private Integer lastEventKind;
    private List<String> lastEventCards = List.of();
    private Boolean lastEventActorSelf;
    private String suggestedOpeningCard;

    void reset() {
        hand.clear();
        table.clear();
        attacks.clear();
        defenses.clear();
        discard.clear();
        knownOpponent.clear();
        drawCandidates.clear();
        lastVisibleHand = Set.of();
        initialCandidate = Set.of();
        refreshCandidate = Set.of();
        initialCandidateFrames = 0;
        refreshCandidateFrames = 0;
        refreshAge = 0;
        absentTableFrames = 0;
        initialized = false;
        awaitingRefreshedHand = false;
        playerAttacker = null;
        deckCount = 24;
        opponentCount = 6;
        expectedHandCount = 0;
        lastEventKind = null;
        lastEventCards = List.of();
        lastEventActorSelf = null;
        suggestedOpeningCard = null;
    }

    void observe(List<ObservedCard> observed) {
        Set<String> visibleHand = new LinkedHashSet<>();
        List<ObservedCard> visibleTableItems = new ArrayList<>();
        for (ObservedCard card : observed) {
            if (card.zone == Zone.HAND && card.reliable()) {
                visibleHand.add(card.card);
            } else if (card.zone == Zone.TABLE
                    && card.classConfidence >= 0.40f
                    && card.boxConfidence >= 0.20f) {
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
                lastVisibleHand = Set.copyOf(visibleHand);
                return;
            }
        }

        boolean handCountDecreased = !lastVisibleHand.isEmpty()
                && visibleHand.size() < lastVisibleHand.size();

        if (!table.isEmpty()) {
            Set<String> overlap = new HashSet<>(visibleTable);
            overlap.retainAll(table);
            if (visibleTable.isEmpty()) {
                absentTableFrames++;
                if (absentTableFrames >= 2) {
                    settleBout();
                }
            } else if (overlap.isEmpty()) {
                // The previous implementation waited for a second disjoint
                // frame. During that delay the overlay could recommend a move
                // for the previous bout while already displaying the new
                // table. A completely different non-empty table is stronger
                // evidence than a transient empty detector frame, so switch
                // bouts immediately and process the visible opening card.
                settleBout();
                absentTableFrames = 0;
            } else {
                absentTableFrames = 0;
            }
        }

        processNewTableCards(visibleTableItems, handCountDecreased);
        // Card identities are unique in a 36-card deck. Even if a table card
        // was already remembered before this frame, it can no longer remain
        // in our hand. Keeping this invariant prevents stale suggestions such
        // as transferring with the same card already visible on the table.
        hand.removeAll(table);
        observeHand(visibleHand);
        lastVisibleHand = Set.copyOf(visibleHand);
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

    private void processNewTableCards(
            List<ObservedCard> visibleTableItems, boolean handCountDecreased) {
        Set<String> oldHand = new HashSet<>(hand);
        oldHand.addAll(lastVisibleHand);
        boolean tableWasEmpty = table.isEmpty();
        int unseenCards = 0;
        for (ObservedCard item : visibleTableItems) {
            unseenCards += table.contains(item.card) ? 0 : 1;
        }
        for (ObservedCard item : visibleTableItems) {
            if (table.contains(item.card)) {
                continue;
            }
            boolean playedByPlayer = oldHand.contains(item.card);
            if (tableWasEmpty && !playedByPlayer
                    && item.card.equals(suggestedOpeningCard)) {
                playedByPlayer = true;
            }
            if (tableWasEmpty && !playedByPlayer
                    && unseenCards == 1 && handCountDecreased) {
                // Fallback for a rare table misclassification: if exactly one
                // card appeared while our visible hand lost a card, we opened.
                playedByPlayer = true;
            }
            if (tableWasEmpty) {
                // Re-observe the opener every bout instead of trusting a role
                // carried through an animation that CV may have skipped.
                playerAttacker = playedByPlayer;
                suggestedOpeningCard = null;
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
            rememberEvent(4, List.of(), previousPlayerAttacker);
        } else {
            if (previousPlayerAttacker) {
                knownOpponent.addAll(table);
                opponentCount += table.size();
            } else {
                hand.addAll(table);
            }
            rememberEvent(5, List.of(), previousPlayerAttacker);
        }
        table.clear();
        attacks.clear();
        defenses.clear();
        absentTableFrames = 0;
        refillCounts(previousPlayerAttacker);
        // Until the next physical opening card appears, show only a
        // conditional suggestion. Its owner will establish the real role.
        playerAttacker = null;
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

    void rememberSuggestedOpening(String card) {
        suggestedOpeningCard = card;
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
