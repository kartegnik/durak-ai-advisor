package ru.kartegnik.durakadvisor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public final class GameStateTrackerTest {
    @Test
    public void completedBoutMovesCardsToDiscardAndSwapsRoles() {
        GameStateTracker tracker = initializedTracker();
        String[] remaining = {"7C", "8D", "9S", "10H", "AC"};
        tracker.observe(frame(remaining, new String[]{"6H"}));
        assertEquals(Boolean.TRUE, tracker.playerAttacker());
        assertFalse(tracker.hand().contains("6H"));

        tracker.observe(frame(remaining, new String[]{"6H", "7H"}));
        assertEquals(1, tracker.defenses().size());
        assertEquals(5, tracker.opponentCount());

        tracker.observe(frame(remaining, new String[]{}));
        tracker.observe(frame(remaining, new String[]{}));
        assertEquals(null, tracker.playerAttacker());
        assertEquals(2, tracker.discard().size());
        assertTrue(tracker.table().isEmpty());
        assertEquals(22, tracker.deckCount());
        assertEquals(6, tracker.opponentCount());
    }

    @Test
    public void takingCardsKeepsTheSameAttacker() {
        GameStateTracker tracker = initializedTracker();
        String[] hand = {"6H", "7C", "8D", "9S", "10H", "AC"};
        tracker.observe(frame(hand, new String[]{"6D"}));
        assertEquals(Boolean.FALSE, tracker.playerAttacker());

        tracker.observe(frame(hand, new String[]{}));
        tracker.observe(frame(hand, new String[]{}));
        assertEquals(null, tracker.playerAttacker());
        assertTrue(tracker.hand().contains("6D"));
        assertTrue(tracker.discard().isEmpty());
    }

    @Test
    public void sameRankDefenseBeforeAnyBeatIsATransfer() {
        GameStateTracker tracker = initializedTracker();
        String[] hand = {"6H", "7C", "8D", "9S", "10H", "AC"};
        tracker.observe(frame(hand, new String[]{"7D"}));
        String[] afterTransfer = {"6H", "8D", "9S", "10H", "AC"};
        tracker.observe(frame(afterTransfer, new String[]{"7D", "7C"}));

        assertEquals(Boolean.TRUE, tracker.playerAttacker());
        assertEquals(List.of("7D", "7C"), tracker.attacks());
        assertTrue(tracker.defenses().isEmpty());
        assertEquals(Integer.valueOf(6), tracker.lastEventKind());
    }

    @Test
    public void disjointVisibleTableStartsTheNextBoutImmediately() {
        GameStateTracker tracker = initializedTracker();
        String[] remaining = {"7C", "8D", "9S", "10H", "AC"};
        tracker.observe(frame(remaining, new String[]{"6H"}));
        tracker.observe(frame(remaining, new String[]{"6H", "7H"}));

        tracker.observe(frame(remaining, new String[]{"6D"}));

        assertEquals(List.of("6D"), tracker.attacks());
        assertTrue(tracker.defenses().isEmpty());
        assertEquals(Boolean.FALSE, tracker.playerAttacker());
        assertTrue(tracker.discard().contains("6H"));
        assertTrue(tracker.discard().contains("7H"));
    }

    @Test
    public void visibleHandDoesNotExposeRememberedMissingCardAsPlayable() {
        GameStateTracker tracker = initializedTracker();
        String[] visible = {"6H", "7C", "8D", "9S", "AC"};

        tracker.observe(frame(visible, new String[]{"6D"}));

        assertTrue(tracker.hand().contains("10H"));
        assertFalse(tracker.visibleHand().contains("10H"));
        assertEquals(5, tracker.visibleHand().size());
    }

    @Test
    public void differentRankBeatingCardRepairsMissedDefense() {
        GameStateTracker tracker = initializedTracker();
        tracker.setTrumpSuit("C");
        String[] hand = {"6H", "7C", "8D", "9S", "10H", "AC"};
        tracker.observe(frame(hand, new String[]{"QH"}));
        tracker.observe(frame(hand, new String[]{"QH", "KH"}));

        assertEquals(List.of("QH"), tracker.attacks());
        assertEquals(List.of("KH"), tracker.defenses());
    }

    @Test
    public void repairSelectsActualBeatingCardBeforeSameRankThrowIn() {
        GameStateTracker tracker = initializedTracker();
        tracker.setTrumpSuit("C");
        String[] hand = {"6H", "7C", "8D", "9S", "AC"};
        tracker.observe(frame(hand, new String[]{"QH"}));
        // KD is deliberately processed first. It cannot beat QH, while KH
        // must be the defense that made a later rank-K throw-in legal.
        tracker.observe(frame(hand, new String[]{"QH", "KD", "KH"}));

        assertEquals(List.of("QH", "KD"), tracker.attacks());
        assertEquals(List.of("KH"), tracker.defenses());
    }

    private static GameStateTracker initializedTracker() {
        GameStateTracker tracker = new GameStateTracker();
        String[] initial = {"6H", "7C", "8D", "9S", "10H", "AC"};
        tracker.observe(frame(initial, new String[]{}));
        tracker.observe(frame(initial, new String[]{}));
        assertTrue(tracker.initialized());
        return tracker;
    }

    private static List<GameStateTracker.ObservedCard> frame(
            String[] hand, String[] table) {
        List<GameStateTracker.ObservedCard> result = new ArrayList<>();
        for (int index = 0; index < hand.length; index++) {
            result.add(card(hand[index], GameStateTracker.Zone.HAND, index));
        }
        for (int index = 0; index < table.length; index++) {
            result.add(card(table[index], GameStateTracker.Zone.TABLE, index));
        }
        return result;
    }

    private static GameStateTracker.ObservedCard card(
            String name, GameStateTracker.Zone zone, float x) {
        return new GameStateTracker.ObservedCard(
                name, zone, 0.99f, 0.99f, x, 500.0f);
    }
}
