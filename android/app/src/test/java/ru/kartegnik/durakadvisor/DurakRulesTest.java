package ru.kartegnik.durakadvisor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Set;

import org.junit.Test;

public final class DurakRulesTest {
    @Test
    public void modelCardOrderMatchesPythonTrainingOrder() {
        assertEquals(0, DurakRules.cardIndex("6H"));
        assertEquals(8, DurakRules.cardIndex("AH"));
        assertEquals(9, DurakRules.cardIndex("6C"));
        assertEquals(18, DurakRules.cardIndex("6S"));
        assertEquals(27, DurakRules.cardIndex("6D"));
        assertEquals(35, DurakRules.cardIndex("AD"));
    }

    @Test
    public void trumpBeatsNonTrumpButNotHigherTrump() {
        assertTrue(DurakRules.beats("6S", "AH", "S"));
        assertFalse(DurakRules.beats("6S", "7S", "S"));
        assertTrue(DurakRules.beats("AS", "7S", "S"));
    }

    @Test
    public void defenseIncludesTransferBeforeFirstDefense() {
        DurakRules.Advice advice = DurakRules.legalActions(
                Set.of("7C", "8H", "9S"), List.of("7D"), List.of(), false, "S");
        assertTrue(advice.options.stream().anyMatch(option -> option.type == 6
                && "7C".equals(option.card)));
        assertTrue(advice.options.stream().anyMatch(option -> option.type == 2
                && "9S".equals(option.card)));
        assertTrue(advice.options.stream().anyMatch(option -> option.type == 3));
    }

    @Test
    public void attackerCanPassOnlyAfterAllCardsAreBeaten() {
        DurakRules.Advice waiting = DurakRules.legalActions(
                Set.of("8C"), List.of("7D"), List.of(), true, "S");
        assertTrue(waiting.options.isEmpty());

        DurakRules.Advice complete = DurakRules.legalActions(
                Set.of("8C"), List.of("7D"), List.of("9D"), true, "S");
        assertTrue(complete.options.stream().anyMatch(option -> option.type == 4));
    }
}
