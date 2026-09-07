package ru.kartegnik.durakadvisor;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.StringJoiner;

import ai.onnxruntime.OrtException;

final class FrameAnalyzer {
    private static final int MAX_ANALYSIS_WIDTH = 576;
    private static final double MIN_LOCK_CONFIDENCE = 0.65;
    private static final long CARD_ANALYSIS_INTERVAL_MS = 700L;

    interface Listener {
        void onText(String text);
    }

    private final TrumpSuitMatcher matcher;
    private final CardRecognizer cardRecognizer;
    private final Listener listener;
    private String candidate;
    private int candidateFrames;
    private String lockedSuit;
    private List<CardRecognizer.DetectedCard> detectedCards = List.of();
    private long lastCardAnalysisAt;
    private boolean cardRecognizerFailed;

    FrameAnalyzer(Context context, Listener listener) throws IOException {
        this.listener = listener;
        try (InputStream stream = context.getAssets().open("trump_suit_templates.bin")) {
            matcher = new TrumpSuitMatcher(stream);
        }
        cardRecognizer = new CardRecognizer(context);
    }

    void analyze(Bitmap bitmap) {
        Bitmap analysis = bitmap;
        if (bitmap.getWidth() > MAX_ANALYSIS_WIDTH) {
            int scaledHeight = Math.round(
                    bitmap.getHeight() * MAX_ANALYSIS_WIDTH / (float) bitmap.getWidth());
            analysis = Bitmap.createScaledBitmap(
                    bitmap, MAX_ANALYSIS_WIDTH, scaledHeight, true);
        }
        int width = analysis.getWidth();
        int height = analysis.getHeight();
        int[] pixels = new int[width * height];
        analysis.getPixels(pixels, 0, width, 0, 0, width, height);
        if (analysis != bitmap) {
            analysis.recycle();
        }
        TrumpSuitMatcher.Detection trumpDetection = null;
        if (lockedSuit == null) {
            trumpDetection = matcher.detect(pixels, width, height);
            updateTrump(trumpDetection);
        }

        long now = SystemClock.elapsedRealtime();
        if (!cardRecognizerFailed
                && now - lastCardAnalysisAt >= CARD_ANALYSIS_INTERVAL_MS) {
            lastCardAnalysisAt = now;
            try {
                detectedCards = cardRecognizer.detect(pixels, width, height);
            } catch (OrtException | RuntimeException error) {
                cardRecognizerFailed = true;
                detectedCards = List.of();
            }
        }
        listener.onText(displayText(trumpDetection));
    }

    private void updateTrump(TrumpSuitMatcher.Detection detection) {
        boolean reliable = detection.cardPresent
                && detection.suit != null
                && detection.confidence >= MIN_LOCK_CONFIDENCE;
        if (reliable) {
            if (detection.suit.equals(candidate)) {
                candidateFrames++;
            } else {
                candidate = detection.suit;
                candidateFrames = 1;
            }
            if (candidateFrames >= 2) {
                lockedSuit = candidate;
            }
        } else {
            // Matching observations must be consecutive. A missing or weak
            // card between them means that dealing has not settled yet.
            candidate = null;
            candidateFrames = 0;
        }
    }

    void reset() {
        candidate = null;
        candidateFrames = 0;
        lockedSuit = null;
    }

    void lockSuit(String suit) {
        if (!List.of("H", "D", "C", "S").contains(suit)) {
            throw new IllegalArgumentException("Unsupported suit: " + suit);
        }
        candidate = null;
        candidateFrames = 0;
        lockedSuit = suit;
        listener.onText(displayText(null));
    }

    void close() {
        cardRecognizer.close();
    }

    private String displayText(TrumpSuitMatcher.Detection detection) {
        String trumpText;
        if (lockedSuit != null) {
            trumpText = "Козырь: " + suitName(lockedSuit);
        } else if (detection != null && detection.suit != null) {
            trumpText = String.format(
                    Locale.US, "Проверяю: %s %.0f%%",
                    suitName(detection.suit), detection.confidence * 100.0);
        } else {
            trumpText = "Ищу козырь…";
        }
        if (cardRecognizerFailed) {
            return trumpText + "\nКарты: ошибка модели";
        }

        List<CardRecognizer.DetectedCard> hand = cardsIn(CardRecognizer.Zone.HAND);
        List<CardRecognizer.DetectedCard> table = cardsIn(CardRecognizer.Zone.TABLE);
        return trumpText
                + "\nРука: " + cardList(hand)
                + "\nСтол: " + cardList(table);
    }

    private List<CardRecognizer.DetectedCard> cardsIn(CardRecognizer.Zone zone) {
        List<CardRecognizer.DetectedCard> result = new ArrayList<>();
        for (CardRecognizer.DetectedCard card : detectedCards) {
            if (card.zone == zone) {
                result.add(card);
            }
        }
        result.sort(Comparator
                .comparingDouble((CardRecognizer.DetectedCard card) -> card.centerY)
                .thenComparingDouble(card -> card.centerX));
        return result;
    }

    private static String cardList(List<CardRecognizer.DetectedCard> cards) {
        if (cards.isEmpty()) {
            return "—";
        }
        StringJoiner result = new StringJoiner(" ");
        for (CardRecognizer.DetectedCard card : cards) {
            boolean uncertain = card.classConfidence < 0.65 || card.boxConfidence < 0.35;
            result.add(displayCard(card.card) + (uncertain ? "?" : ""));
        }
        return result.toString();
    }

    private static String displayCard(String card) {
        String rank = card.substring(0, card.length() - 1);
        String suit = card.substring(card.length() - 1);
        rank = switch (rank) {
            case "J" -> "В";
            case "Q" -> "Д";
            case "K" -> "К";
            case "A" -> "Т";
            default -> rank;
        };
        String symbol = switch (suit) {
            case "H" -> "♥";
            case "D" -> "♦";
            case "C" -> "♣";
            case "S" -> "♠";
            default -> suit;
        };
        return rank + symbol;
    }

    private static String suitName(String suit) {
        return switch (suit) {
            case "H" -> "черви";
            case "D" -> "бубны";
            case "C" -> "крести";
            case "S" -> "пики";
            default -> suit;
        };
    }
}
