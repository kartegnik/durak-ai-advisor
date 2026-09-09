package ru.kartegnik.durakadvisor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Pure Java rules and model option encoding shared by the live advisor and tests. */
final class DurakRules {
    static final int CARD_COUNT = 36;
    static final int OPTION_SIZE = 43;
    private static final List<String> RANKS = List.of(
            "6", "7", "8", "9", "10", "J", "Q", "K", "A");
    // This order must stay identical to state.py, used while training PPO v3.
    private static final List<String> MODEL_SUITS = List.of("H", "C", "S", "D");

    static final class MoveOption {
        final int type;
        final String card;
        final String display;

        MoveOption(int type, String card, String display) {
            this.type = type;
            this.card = card;
            this.display = display;
        }

        float[] encode() {
            float[] result = new float[OPTION_SIZE];
            result[type] = 1.0f;
            if (card != null) {
                result[7 + cardIndex(card)] = 1.0f;
            }
            return result;
        }
    }

    static final class Advice {
        final List<MoveOption> options;
        final String note;

        Advice(List<MoveOption> options, String note) {
            this.options = List.copyOf(options);
            this.note = note;
        }
    }

    private DurakRules() {}

    static Advice legalActions(
            Collection<String> hand, List<String> attacks, List<String> defenses,
            Boolean playerAttacker, String trump) {
        validateSuit(trump);
        List<String> sortedHand = new ArrayList<>(hand);
        sortedHand.sort(Comparator
                .comparingInt(DurakRules::rankIndex)
                .thenComparing(card -> card.substring(card.length() - 1)));

        if (attacks.isEmpty()) {
            if (Boolean.FALSE.equals(playerAttacker)) {
                return new Advice(List.of(), "ожидаем ход соперника");
            }
            List<MoveOption> openings = new ArrayList<>();
            for (String card : sortedHand) {
                openings.add(new MoveOption(0, card, "ход " + displayCard(card)));
            }
            return new Advice(
                    openings,
                    playerAttacker == null ? "если первый ход ваш" : "ваша атака");
        }

        Set<String> ranksOnTable = new HashSet<>();
        for (String card : attacks) {
            ranksOnTable.add(rank(card));
        }
        for (String card : defenses) {
            ranksOnTable.add(rank(card));
        }

        if (Boolean.TRUE.equals(playerAttacker)) {
            List<MoveOption> options = new ArrayList<>();
            for (String card : sortedHand) {
                if (ranksOnTable.contains(rank(card))) {
                    options.add(new MoveOption(
                            1, card, "подкинуть " + displayCard(card)));
                }
            }
            if (defenses.size() >= attacks.size()) {
                options.add(new MoveOption(4, null, "бито / пас"));
            }
            String note = options.isEmpty()
                    ? "ожидаем, пока соперник отобьёт или возьмёт" : "";
            return new Advice(options, note);
        }

        if (attacks.size() <= defenses.size()) {
            return new Advice(List.of(), "ожидаем следующую карту соперника");
        }
        String attack = attacks.get(defenses.size());
        List<MoveOption> options = new ArrayList<>();
        for (String card : sortedHand) {
            if (beats(card, attack, trump)) {
                options.add(new MoveOption(
                        2, card,
                        "отбить " + displayCard(attack) + " картой " + displayCard(card)));
            }
        }
        if (defenses.isEmpty() && allAttackRanksEqual(attacks)) {
            for (String card : sortedHand) {
                if (rank(card).equals(rank(attack))) {
                    options.add(new MoveOption(
                            6, card, "перевести картой " + displayCard(card)));
                }
            }
        }
        options.add(new MoveOption(3, null, "взять"));
        return new Advice(options, "");
    }

    private static boolean allAttackRanksEqual(List<String> attacks) {
        String first = rank(attacks.get(0));
        for (String attack : attacks) {
            if (!rank(attack).equals(first)) {
                return false;
            }
        }
        return true;
    }

    static boolean beats(String card, String attack, String trump) {
        validateCard(card);
        validateCard(attack);
        validateSuit(trump);
        String suit = suit(card);
        String attackSuit = suit(attack);
        if (suit.equals(attackSuit)) {
            return rankIndex(card) > rankIndex(attack);
        }
        return suit.equals(trump) && !attackSuit.equals(trump);
    }

    static int cardIndex(String card) {
        validateCard(card);
        return MODEL_SUITS.indexOf(suit(card)) * RANKS.size() + rankIndex(card);
    }

    static int suitIndex(String suit) {
        validateSuit(suit);
        return MODEL_SUITS.indexOf(suit);
    }

    static String displayCard(String card) {
        validateCard(card);
        String shownRank = switch (rank(card)) {
            case "J" -> "В";
            case "Q" -> "Д";
            case "K" -> "К";
            case "A" -> "Т";
            default -> rank(card);
        };
        String shownSuit = switch (suit(card)) {
            case "H" -> "♥";
            case "D" -> "♦";
            case "C" -> "♣";
            case "S" -> "♠";
            default -> throw new IllegalArgumentException(card);
        };
        return shownRank + shownSuit;
    }

    private static int rankIndex(String card) {
        validateCard(card);
        return RANKS.indexOf(rank(card));
    }

    private static String rank(String card) {
        return card.substring(0, card.length() - 1);
    }

    private static String suit(String card) {
        return card.substring(card.length() - 1);
    }

    private static void validateCard(String card) {
        if (card == null || card.length() < 2
                || !RANKS.contains(rank(card)) || !MODEL_SUITS.contains(suit(card))) {
            throw new IllegalArgumentException("Unknown card: " + card);
        }
    }

    private static void validateSuit(String suit) {
        if (!MODEL_SUITS.contains(suit)) {
            throw new IllegalArgumentException("Unknown suit: " + suit);
        }
    }
}
