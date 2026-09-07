package ru.kartegnik.durakadvisor;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/** Runs the published card localizer and classifier entirely on the device. */
final class CardRecognizer implements AutoCloseable {
    private static final int LOCALIZER_SIZE = 416;
    private static final int CROP_WIDTH = 64;
    private static final int CROP_HEIGHT = 96;
    private static final int MAX_DETECTIONS = 24;
    private static final float BOX_THRESHOLD = 0.25f;
    private static final float NMS_THRESHOLD = 0.50f;
    private static final double GAMEPLAY_HEIGHT_FRACTION = 0.85;

    private static final String[] CLASS_NAMES = buildClassNames();

    enum Zone {
        OPPONENT,
        TABLE,
        HAND,
    }

    static final class DetectedCard {
        final String card;
        final Zone zone;
        final float classConfidence;
        final float boxConfidence;
        final float centerX;
        final float centerY;

        DetectedCard(
                String card, Zone zone, float classConfidence, float boxConfidence,
                float centerX, float centerY) {
            this.card = card;
            this.zone = zone;
            this.classConfidence = classConfidence;
            this.boxConfidence = boxConfidence;
            this.centerX = centerX;
            this.centerY = centerY;
        }
    }

    private static final class Letterbox {
        final float[] input;
        final double scale;
        final int padLeft;
        final int padTop;
        final int viewportHeight;

        Letterbox(float[] input, double scale, int padLeft, int padTop, int viewportHeight) {
            this.input = input;
            this.scale = scale;
            this.padLeft = padLeft;
            this.padTop = padTop;
            this.viewportHeight = viewportHeight;
        }
    }

    private static final class Box {
        final float x1;
        final float y1;
        final float x2;
        final float y2;
        final float confidence;

        Box(float x1, float y1, float x2, float y2, float confidence) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.confidence = confidence;
        }
    }

    private final OrtEnvironment environment;
    private final OrtSession localizer;
    private final OrtSession classifier;

    CardRecognizer(Context context) throws IOException {
        environment = OrtEnvironment.getEnvironment();
        try {
            localizer = createSession(context, "models/card_localizer.onnx");
            classifier = createSession(context, "models/card_classifier.onnx");
        } catch (OrtException error) {
            throw new IOException("Could not initialize card models", error);
        }
    }

    List<DetectedCard> detect(int[] pixels, int width, int height) throws OrtException {
        Letterbox letterbox = letterbox(pixels, width, height);
        List<Box> boxes;
        long[] localizerShape = {1, 3, LOCALIZER_SIZE, LOCALIZER_SIZE};
        try (OnnxTensor input = OnnxTensor.createTensor(
                environment, FloatBuffer.wrap(letterbox.input), localizerShape);
             OrtSession.Result result = localizer.run(
                     Collections.singletonMap("images", input))) {
            float[][][] output = (float[][][]) result.get(0).getValue();
            boxes = nonMaximumSuppression(
                    output[0], width, letterbox.viewportHeight,
                    letterbox.scale, letterbox.padLeft, letterbox.padTop);
        }
        if (boxes.isEmpty()) {
            return List.of();
        }

        float[] cardInputs = cardInputs(pixels, width, letterbox.viewportHeight, boxes);
        long[] classifierShape = {boxes.size(), 3, CROP_HEIGHT, CROP_WIDTH};
        float[][] logits;
        try (OnnxTensor input = OnnxTensor.createTensor(
                environment, FloatBuffer.wrap(cardInputs), classifierShape);
             OrtSession.Result result = classifier.run(
                     Collections.singletonMap("cards", input))) {
            logits = (float[][]) result.get(0).getValue();
        }

        Map<String, DetectedCard> unique = new HashMap<>();
        for (int index = 0; index < boxes.size(); index++) {
            Box box = boxes.get(index);
            int classIndex = maximumIndex(logits[index]);
            float classConfidence = softmaxProbability(logits[index], classIndex);
            float centerX = (box.x1 + box.x2) / 2.0f;
            float centerY = (box.y1 + box.y2) / 2.0f;
            double verticalRatio = centerY / letterbox.viewportHeight;
            Zone zone = verticalRatio < 0.22
                    ? Zone.OPPONENT
                    : verticalRatio >= 0.68 ? Zone.HAND : Zone.TABLE;
            DetectedCard detected = new DetectedCard(
                    CLASS_NAMES[classIndex], zone, classConfidence, box.confidence,
                    centerX, centerY);
            DetectedCard previous = unique.get(detected.card);
            if (previous == null || combinedConfidence(detected) > combinedConfidence(previous)) {
                unique.put(detected.card, detected);
            }
        }

        List<DetectedCard> result = new ArrayList<>(unique.values());
        result.sort(Comparator
                .comparing((DetectedCard card) -> card.zone)
                .thenComparingDouble(card -> card.centerY)
                .thenComparingDouble(card -> card.centerX));
        return result;
    }

    @Override
    public void close() {
        try {
            classifier.close();
        } catch (OrtException ignored) {
            // The service is already shutting down.
        }
        try {
            localizer.close();
        } catch (OrtException ignored) {
            // The service is already shutting down.
        }
    }

    private OrtSession createSession(Context context, String asset) throws IOException, OrtException {
        byte[] model;
        try (InputStream stream = context.getAssets().open(asset);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int read;
            while ((read = stream.read(buffer)) >= 0) {
                output.write(buffer, 0, read);
            }
            model = output.toByteArray();
        }
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(Math.max(
                    1, Math.min(4, Runtime.getRuntime().availableProcessors() / 2)));
            return environment.createSession(model, options);
        }
    }

    private static Letterbox letterbox(int[] pixels, int width, int height) {
        int viewportHeight = Math.min(height, Math.max(
                1, (int) Math.round(height * GAMEPLAY_HEIGHT_FRACTION)));
        double scale = Math.min(
                LOCALIZER_SIZE / (double) width,
                LOCALIZER_SIZE / (double) viewportHeight);
        int resizedWidth = Math.max(1, (int) Math.round(width * scale));
        int resizedHeight = Math.max(1, (int) Math.round(viewportHeight * scale));
        int padLeft = (LOCALIZER_SIZE - resizedWidth) / 2;
        int padTop = (LOCALIZER_SIZE - resizedHeight) / 2;
        int plane = LOCALIZER_SIZE * LOCALIZER_SIZE;
        float[] input = new float[plane * 3];
        float padding = 114.0f / 255.0f;
        for (int index = 0; index < input.length; index++) {
            input[index] = padding;
        }
        for (int y = 0; y < resizedHeight; y++) {
            double sourceY = (y + 0.5) / scale - 0.5;
            for (int x = 0; x < resizedWidth; x++) {
                double sourceX = (x + 0.5) / scale - 0.5;
                int color = bilinear(pixels, width, viewportHeight, sourceX, sourceY);
                int destination = (y + padTop) * LOCALIZER_SIZE + x + padLeft;
                input[destination] = ((color >> 16) & 0xff) / 255.0f;
                input[plane + destination] = ((color >> 8) & 0xff) / 255.0f;
                input[plane * 2 + destination] = (color & 0xff) / 255.0f;
            }
        }
        return new Letterbox(input, scale, padLeft, padTop, viewportHeight);
    }

    private static List<Box> nonMaximumSuppression(
            float[][] output, int width, int height,
            double scale, int padLeft, int padTop) {
        List<Box> candidates = new ArrayList<>();
        int anchors = output[0].length;
        for (int index = 0; index < anchors; index++) {
            float confidence = output[4][index];
            if (confidence < BOX_THRESHOLD) {
                continue;
            }
            float centerX = output[0][index];
            float centerY = output[1][index];
            float boxWidth = output[2][index];
            float boxHeight = output[3][index];
            float x1 = clamp((float) ((centerX - boxWidth / 2.0 - padLeft) / scale), 0, width);
            float y1 = clamp((float) ((centerY - boxHeight / 2.0 - padTop) / scale), 0, height);
            float x2 = clamp((float) ((centerX + boxWidth / 2.0 - padLeft) / scale), 0, width);
            float y2 = clamp((float) ((centerY + boxHeight / 2.0 - padTop) / scale), 0, height);
            if (x2 - x1 >= 4.0f && y2 - y1 >= 4.0f) {
                candidates.add(new Box(x1, y1, x2, y2, confidence));
            }
        }
        candidates.sort(Comparator.comparingDouble((Box box) -> box.confidence).reversed());
        List<Box> selected = new ArrayList<>();
        for (Box candidate : candidates) {
            boolean overlaps = false;
            for (Box previous : selected) {
                if (intersectionOverUnion(candidate, previous) > NMS_THRESHOLD) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                selected.add(candidate);
                if (selected.size() >= MAX_DETECTIONS) {
                    break;
                }
            }
        }
        return selected;
    }

    private static float[] cardInputs(
            int[] pixels, int width, int height, List<Box> boxes) {
        int cropPlane = CROP_WIDTH * CROP_HEIGHT;
        float[] inputs = new float[boxes.size() * cropPlane * 3];
        for (int cardIndex = 0; cardIndex < boxes.size(); cardIndex++) {
            Box box = boxes.get(cardIndex);
            int base = cardIndex * cropPlane * 3;
            double boxWidth = Math.max(1.0, box.x2 - box.x1);
            double boxHeight = Math.max(1.0, box.y2 - box.y1);
            for (int y = 0; y < CROP_HEIGHT; y++) {
                double sourceY = box.y1 + (y + 0.5) * boxHeight / CROP_HEIGHT - 0.5;
                for (int x = 0; x < CROP_WIDTH; x++) {
                    double sourceX = box.x1 + (x + 0.5) * boxWidth / CROP_WIDTH - 0.5;
                    int color = bilinear(pixels, width, height, sourceX, sourceY);
                    int destination = y * CROP_WIDTH + x;
                    inputs[base + destination] = ((color >> 16) & 0xff) / 255.0f;
                    inputs[base + cropPlane + destination] = ((color >> 8) & 0xff) / 255.0f;
                    inputs[base + cropPlane * 2 + destination] = (color & 0xff) / 255.0f;
                }
            }
        }
        return inputs;
    }

    private static int bilinear(
            int[] pixels, int width, int height, double sourceX, double sourceY) {
        double x = Math.max(0.0, Math.min(width - 1.0, sourceX));
        double y = Math.max(0.0, Math.min(height - 1.0, sourceY));
        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);
        int x1 = Math.min(width - 1, x0 + 1);
        int y1 = Math.min(height - 1, y0 + 1);
        double dx = x - x0;
        double dy = y - y0;
        int topLeft = pixels[y0 * width + x0];
        int topRight = pixels[y0 * width + x1];
        int bottomLeft = pixels[y1 * width + x0];
        int bottomRight = pixels[y1 * width + x1];
        int red = interpolate(
                topLeft >> 16, topRight >> 16, bottomLeft >> 16, bottomRight >> 16,
                dx, dy);
        int green = interpolate(
                topLeft >> 8, topRight >> 8, bottomLeft >> 8, bottomRight >> 8,
                dx, dy);
        int blue = interpolate(
                topLeft, topRight, bottomLeft, bottomRight, dx, dy);
        return 0xff000000 | (red << 16) | (green << 8) | blue;
    }

    private static int interpolate(
            int topLeft, int topRight, int bottomLeft, int bottomRight,
            double dx, double dy) {
        topLeft &= 0xff;
        topRight &= 0xff;
        bottomLeft &= 0xff;
        bottomRight &= 0xff;
        double top = topLeft * (1.0 - dx) + topRight * dx;
        double bottom = bottomLeft * (1.0 - dx) + bottomRight * dx;
        return (int) Math.round(top * (1.0 - dy) + bottom * dy);
    }

    private static float intersectionOverUnion(Box first, Box second) {
        float x1 = Math.max(first.x1, second.x1);
        float y1 = Math.max(first.y1, second.y1);
        float x2 = Math.min(first.x2, second.x2);
        float y2 = Math.min(first.y2, second.y2);
        float intersection = Math.max(0.0f, x2 - x1) * Math.max(0.0f, y2 - y1);
        float firstArea = (first.x2 - first.x1) * (first.y2 - first.y1);
        float secondArea = (second.x2 - second.x1) * (second.y2 - second.y1);
        return intersection / Math.max(1.0e-6f, firstArea + secondArea - intersection);
    }

    private static int maximumIndex(float[] values) {
        int best = 0;
        for (int index = 1; index < values.length; index++) {
            if (values[index] > values[best]) {
                best = index;
            }
        }
        return best;
    }

    private static float softmaxProbability(float[] logits, int selected) {
        float maximum = logits[maximumIndex(logits)];
        double total = 0.0;
        for (float logit : logits) {
            total += Math.exp(logit - maximum);
        }
        return (float) (Math.exp(logits[selected] - maximum) / total);
    }

    private static float combinedConfidence(DetectedCard card) {
        return card.classConfidence * card.boxConfidence;
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static String[] buildClassNames() {
        String[] ranks = {"6", "7", "8", "9", "10", "J", "Q", "K", "A"};
        String[] suits = {"H", "D", "C", "S"};
        String[] result = new String[ranks.length * suits.length];
        int index = 0;
        for (String rank : ranks) {
            for (String suit : suits) {
                result[index++] = rank + suit;
            }
        }
        return result;
    }
}
