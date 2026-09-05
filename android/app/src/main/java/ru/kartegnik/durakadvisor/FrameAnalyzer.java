package ru.kartegnik.durakadvisor;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

final class FrameAnalyzer {
    private static final int MAX_ANALYSIS_WIDTH = 540;

    interface Listener {
        void onText(String text);
    }

    private final TrumpSuitMatcher matcher;
    private final Listener listener;
    private String candidate;
    private int candidateFrames;
    private String lockedSuit;

    FrameAnalyzer(Context context, Listener listener) throws IOException {
        this.listener = listener;
        try (InputStream stream = context.getAssets().open("trump_suit_templates.bin")) {
            matcher = new TrumpSuitMatcher(stream);
        }
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
        TrumpSuitMatcher.Detection detection = matcher.detect(pixels, width, height);

        if (lockedSuit == null && detection.cardPresent && detection.suit != null) {
            if (detection.suit.equals(candidate)) {
                candidateFrames++;
            } else {
                candidate = detection.suit;
                candidateFrames = 1;
            }
            if (candidateFrames >= 2) {
                lockedSuit = candidate;
            }
        }

        if (lockedSuit != null) {
            listener.onText("Козырь: " + suitName(lockedSuit));
        } else if (detection.suit != null) {
            listener.onText(String.format(
                    Locale.US, "Проверяю козырь: %s (%.0f%%)",
                    suitName(detection.suit), detection.confidence * 100.0));
        } else {
            listener.onText("Ищу козырную карту…");
        }
    }

    private static String suitName(String suit) {
        return switch (suit) {
            case "H" -> "черви";
            case "D" -> "бубны";
            case "C" -> "трефы";
            case "S" -> "пики";
            default -> suit;
        };
    }
}
