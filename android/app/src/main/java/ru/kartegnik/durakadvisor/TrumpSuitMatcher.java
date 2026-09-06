package ru.kartegnik.durakadvisor;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class TrumpSuitMatcher {
    private static final int CROP_WIDTH = 64;
    private static final int CROP_HEIGHT = 96;
    private static final int GLYPH_WIDTH = 32;
    private static final int GLYPH_HEIGHT = 44;
    // The desktop detector was calibrated on a padded 462x806 phone viewport.
    // Modern phones are taller, so the game area normally occupies a
    // top-aligned slice while controls/navigation use the remaining height.
    private static final double PADDED_VIEWPORT_ASPECT = 462.0 / 806.0;
    private static final double PHONE_SCREEN_ASPECT = 428.0 / 932.0;
    private static final double[] PADDED_VIEWPORT_TRANSFORM = homography(new double[][]{
            {0.090, 0.430}, {0.235, 0.455},
            {0.195, 0.600}, {0.025, 0.565},
    });
    private static final double[] PHONE_SCREEN_TRANSFORM = homography(new double[][]{
            {0.055093, 0.378305}, {0.211612, 0.399925},
            {0.168435, 0.525322}, {-0.015070, 0.495054},
    });
    // Depending on which player starts, the game fans the face-up trump card
    // in the opposite direction. On a tall edge-to-edge phone screen this
    // version is also clipped by the left display edge, so it needs its own
    // calibrated crop instead of mirroring the older transform.
    private static final double[] REVERSED_PHONE_SCREEN_TRANSFORM = homography(new double[][]{
            {-0.005300, 0.388040}, {0.137200, 0.366440},
            {0.199200, 0.488760}, {0.031700, 0.506360},
    });

    static final class Detection {
        final String suit;
        final double confidence;
        final boolean cardPresent;

        Detection(String suit, double confidence, boolean cardPresent) {
            this.suit = suit;
            this.confidence = confidence;
            this.cardPresent = cardPresent;
        }
    }

    private static final class Viewport {
        final double left;
        final double top;
        final double width;
        final double height;

        Viewport(double left, double top, double width, double height) {
            this.left = left;
            this.top = top;
            this.width = width;
            this.height = height;
        }
    }

    private static final class Candidate {
        final String suit;
        final double confidence;
        final boolean present;

        Candidate(String suit, double confidence, boolean present) {
            this.suit = suit;
            this.confidence = confidence;
            this.present = present;
        }
    }

    private final Map<String, List<byte[]>> templates = new HashMap<>();

    TrumpSuitMatcher(InputStream input) throws IOException {
        DataInputStream stream = new DataInputStream(input);
        byte[] magic = new byte[4];
        stream.readFully(magic);
        if (!"DTS1".equals(new String(magic, StandardCharsets.US_ASCII))) {
            throw new IOException("Unsupported trump template asset");
        }
        int suitCount = stream.readUnsignedByte();
        for (int suitIndex = 0; suitIndex < suitCount; suitIndex++) {
            String suit = Character.toString((char) stream.readUnsignedByte());
            int variantCount = stream.readUnsignedByte();
            List<byte[]> variants = new ArrayList<>();
            for (int variant = 0; variant < variantCount; variant++) {
                byte[] glyph = new byte[GLYPH_WIDTH * GLYPH_HEIGHT];
                stream.readFully(glyph);
                variants.add(glyph);
            }
            templates.put(suit, variants);
        }
        if (!templates.keySet().containsAll(List.of("H", "D", "C", "S"))) {
            throw new IOException("Trump template asset does not contain all suits");
        }
    }

    Detection detect(int[] pixels, int screenWidth, int screenHeight) {
        Candidate best = null;
        for (Viewport viewport : candidateViewports(
                screenWidth, screenHeight, PHONE_SCREEN_ASPECT)) {
            Candidate candidate = detectInViewport(
                    pixels, screenWidth, screenHeight, viewport, PHONE_SCREEN_TRANSFORM);
            best = better(best, candidate);
            candidate = detectInViewport(
                    pixels, screenWidth, screenHeight, viewport,
                    REVERSED_PHONE_SCREEN_TRANSFORM);
            best = better(best, candidate);
        }
        for (Viewport viewport : candidateViewports(
                screenWidth, screenHeight, PADDED_VIEWPORT_ASPECT)) {
            Candidate candidate = detectInViewport(
                    pixels, screenWidth, screenHeight, viewport, PADDED_VIEWPORT_TRANSFORM);
            best = better(best, candidate);
        }
        if (best == null || !best.present) {
            return new Detection(null, best == null ? 0.0 : best.confidence, false);
        }
        return new Detection(best.suit, best.confidence, true);
    }

    private static Candidate better(Candidate current, Candidate candidate) {
        if (current == null
                || (candidate.present && !current.present)
                || (candidate.present == current.present
                && candidate.confidence > current.confidence)) {
            return candidate;
        }
        return current;
    }

    private Candidate detectInViewport(
            int[] pixels, int screenWidth, int screenHeight, Viewport viewport,
            double[] transform) {
        int[] crop = extractTrump(
                pixels, screenWidth, screenHeight, viewport, transform);
        byte[] glyph = suitGlyph(crop);

        String bestSuit = null;
        double bestDistance = Double.POSITIVE_INFINITY;
        double secondDistance = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, List<byte[]>> entry : templates.entrySet()) {
            double suitDistance = Double.POSITIVE_INFINITY;
            for (byte[] template : entry.getValue()) {
                suitDistance = Math.min(suitDistance, meanAbsoluteDifference(glyph, template));
            }
            if (suitDistance < bestDistance) {
                secondDistance = bestDistance;
                bestDistance = suitDistance;
                bestSuit = entry.getKey();
            } else if (suitDistance < secondDistance) {
                secondDistance = suitDistance;
            }
        }

        double margin = secondDistance - bestDistance;
        double confidence = clamp(0.55 + margin / 100.0 - bestDistance / 500.0);
        double whiteFraction = whiteFraction(crop);
        double inkFraction = inkFraction(glyph);
        boolean present = whiteFraction >= 0.18
                && inkFraction >= 0.03
                && confidence >= 0.55;
        return new Candidate(bestSuit, confidence, present);
    }

    private static List<Viewport> candidateViewports(
            int width, int height, double referenceAspect) {
        List<Viewport> result = new ArrayList<>();
        result.add(new Viewport(0.0, 0.0, width, height));
        double screenAspect = (double) width / height;
        if (Math.abs(screenAspect - referenceAspect) > 0.01) {
            if (screenAspect > referenceAspect) {
                double viewportWidth = height * referenceAspect;
                result.add(new Viewport(
                        (width - viewportWidth) / 2.0, 0.0, viewportWidth, height));
            } else {
                double viewportHeight = width / referenceAspect;
                result.add(new Viewport(0.0, 0.0, width, viewportHeight));
                result.add(new Viewport(
                        0.0, (height - viewportHeight) / 2.0, width, viewportHeight));
            }
        }
        return result;
    }

    private static int[] extractTrump(
            int[] pixels, int screenWidth, int screenHeight, Viewport viewport,
            double[] transform) {
        int[] crop = new int[CROP_WIDTH * CROP_HEIGHT];
        double[] h = transform;
        for (int y = 0; y < CROP_HEIGHT; y++) {
            double v = y / (double) (CROP_HEIGHT - 1);
            for (int x = 0; x < CROP_WIDTH; x++) {
                double u = x / (double) (CROP_WIDTH - 1);
                double denominator = h[6] * u + h[7] * v + 1.0;
                double normalizedX = (h[0] * u + h[1] * v + h[2]) / denominator;
                double normalizedY = (h[3] * u + h[4] * v + h[5]) / denominator;
                double sourceX = viewport.left + normalizedX * viewport.width;
                double sourceY = viewport.top + normalizedY * viewport.height;
                crop[y * CROP_WIDTH + x] = bilinear(
                        pixels, screenWidth, screenHeight, sourceX, sourceY);
            }
        }
        return crop;
    }

    private static byte[] suitGlyph(int[] crop) {
        byte[] glyph = new byte[GLYPH_WIDTH * GLYPH_HEIGHT];
        for (int y = 0; y < GLYPH_HEIGHT; y++) {
            int sourceY = 14 + y * 18 / GLYPH_HEIGHT;
            for (int x = 0; x < GLYPH_WIDTH; x++) {
                int sourceX = x * 18 / GLYPH_WIDTH;
                int color = crop[sourceY * CROP_WIDTH + sourceX];
                Hsv hsv = hsv(color);
                boolean red = hsv.saturation > 95.0
                        && (hsv.hue < 15.0 || hsv.hue > 165.0);
                boolean black = hsv.value < 105.0;
                glyph[y * GLYPH_WIDTH + x] = (byte) ((red || black) ? 255 : 0);
            }
        }
        return glyph;
    }

    private static double whiteFraction(int[] crop) {
        int white = 0;
        for (int color : crop) {
            Hsv hsv = hsv(color);
            if (hsv.saturation < 80.0 && hsv.value > 155.0) {
                white++;
            }
        }
        return white / (double) crop.length;
    }

    private static double inkFraction(byte[] glyph) {
        int ink = 0;
        for (byte value : glyph) {
            if ((value & 0xff) != 0) {
                ink++;
            }
        }
        return ink / (double) glyph.length;
    }

    private static double meanAbsoluteDifference(byte[] first, byte[] second) {
        long total = 0;
        for (int index = 0; index < first.length; index++) {
            total += Math.abs((first[index] & 0xff) - (second[index] & 0xff));
        }
        return total / (double) first.length;
    }

    private static int bilinear(
            int[] pixels, int width, int height, double sourceX, double sourceY) {
        double clampedX = Math.max(0.0, Math.min(width - 1.0, sourceX));
        double clampedY = Math.max(0.0, Math.min(height - 1.0, sourceY));
        int x0 = (int) Math.floor(clampedX);
        int y0 = (int) Math.floor(clampedY);
        int x1 = Math.min(width - 1, x0 + 1);
        int y1 = Math.min(height - 1, y0 + 1);
        double dx = clampedX - x0;
        double dy = clampedY - y0;

        int c00 = pixels[y0 * width + x0];
        int c10 = pixels[y0 * width + x1];
        int c01 = pixels[y1 * width + x0];
        int c11 = pixels[y1 * width + x1];
        int red = interpolate(c00 >> 16, c10 >> 16, c01 >> 16, c11 >> 16, dx, dy);
        int green = interpolate(c00 >> 8, c10 >> 8, c01 >> 8, c11 >> 8, dx, dy);
        int blue = interpolate(c00, c10, c01, c11, dx, dy);
        return 0xff000000 | (red << 16) | (green << 8) | blue;
    }

    private static int interpolate(
            int c00, int c10, int c01, int c11, double dx, double dy) {
        c00 &= 0xff;
        c10 &= 0xff;
        c01 &= 0xff;
        c11 &= 0xff;
        double top = c00 * (1.0 - dx) + c10 * dx;
        double bottom = c01 * (1.0 - dx) + c11 * dx;
        return (int) Math.round(top * (1.0 - dy) + bottom * dy);
    }

    private static final class Hsv {
        final double hue;
        final double saturation;
        final double value;

        Hsv(double hue, double saturation, double value) {
            this.hue = hue;
            this.saturation = saturation;
            this.value = value;
        }
    }

    private static Hsv hsv(int color) {
        double red = ((color >> 16) & 0xff) / 255.0;
        double green = ((color >> 8) & 0xff) / 255.0;
        double blue = (color & 0xff) / 255.0;
        double max = Math.max(red, Math.max(green, blue));
        double min = Math.min(red, Math.min(green, blue));
        double delta = max - min;
        double hueDegrees;
        if (delta == 0.0) {
            hueDegrees = 0.0;
        } else if (max == red) {
            hueDegrees = 60.0 * (((green - blue) / delta) % 6.0);
        } else if (max == green) {
            hueDegrees = 60.0 * (((blue - red) / delta) + 2.0);
        } else {
            hueDegrees = 60.0 * (((red - green) / delta) + 4.0);
        }
        if (hueDegrees < 0.0) {
            hueDegrees += 360.0;
        }
        double saturation = max == 0.0 ? 0.0 : delta / max;
        return new Hsv(hueDegrees / 2.0, saturation * 255.0, max * 255.0);
    }

    private static double[] homography(double[][] sources) {
        double[][] targets = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        double[][] matrix = new double[8][8];
        double[] values = new double[8];
        for (int index = 0; index < 4; index++) {
            double u = targets[index][0];
            double v = targets[index][1];
            double x = sources[index][0];
            double y = sources[index][1];
            matrix[index * 2] = new double[]{u, v, 1, 0, 0, 0, -x * u, -x * v};
            values[index * 2] = x;
            matrix[index * 2 + 1] = new double[]{0, 0, 0, u, v, 1, -y * u, -y * v};
            values[index * 2 + 1] = y;
        }
        return solve(matrix, values);
    }

    private static double[] solve(double[][] matrix, double[] values) {
        int size = values.length;
        for (int pivot = 0; pivot < size; pivot++) {
            int best = pivot;
            for (int row = pivot + 1; row < size; row++) {
                if (Math.abs(matrix[row][pivot]) > Math.abs(matrix[best][pivot])) {
                    best = row;
                }
            }
            double[] temporaryRow = matrix[pivot];
            matrix[pivot] = matrix[best];
            matrix[best] = temporaryRow;
            double temporaryValue = values[pivot];
            values[pivot] = values[best];
            values[best] = temporaryValue;

            double divisor = matrix[pivot][pivot];
            for (int column = pivot; column < size; column++) {
                matrix[pivot][column] /= divisor;
            }
            values[pivot] /= divisor;
            for (int row = 0; row < size; row++) {
                if (row == pivot) {
                    continue;
                }
                double factor = matrix[row][pivot];
                for (int column = pivot; column < size; column++) {
                    matrix[row][column] -= factor * matrix[pivot][column];
                }
                values[row] -= factor * values[pivot];
            }
        }
        return values;
    }

    private static double clamp(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}
