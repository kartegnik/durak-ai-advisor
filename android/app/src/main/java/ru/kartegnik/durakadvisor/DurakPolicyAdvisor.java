package ru.kartegnik.durakadvisor;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/** Runs the original v1 move scorer exported from durak_model.pt. */
final class DurakPolicyAdvisor implements AutoCloseable {
    private static final int STATE_SIZE = 150;
    private static final int COMBINED_SIZE = STATE_SIZE + DurakRules.OPTION_SIZE;

    static final class Recommendation {
        final String action;
        final float score;

        Recommendation(String action, float score) {
            this.action = action;
            this.score = score;
        }
    }

    private final OrtEnvironment environment;
    private final OrtSession session;
    private String lastSignature;
    private Recommendation lastRecommendation;

    DurakPolicyAdvisor(Context context) throws IOException {
        environment = OrtEnvironment.getEnvironment();
        try {
            session = createSession(context, "models/durak_policy_v1.onnx");
        } catch (OrtException error) {
            throw new IOException("Could not initialize Durak policy", error);
        }
    }

    Recommendation recommend(
            GameStateTracker tracker, String trump, List<DurakRules.MoveOption> options)
            throws OrtException {
        if (options.isEmpty()) {
            return null;
        }
        String signature = signature(tracker, trump, options);
        if (signature.equals(lastSignature) && lastRecommendation != null) {
            return lastRecommendation;
        }

        float[] state = encodeState(tracker, trump);
        float[] combined = new float[options.size() * COMBINED_SIZE];
        for (int index = 0; index < options.size(); index++) {
            int row = index * COMBINED_SIZE;
            System.arraycopy(state, 0, combined, row, STATE_SIZE);
            float[] encoded = options.get(index).encode();
            System.arraycopy(
                    encoded, 0, combined, row + STATE_SIZE, DurakRules.OPTION_SIZE);
        }

        long[] shape = {options.size(), COMBINED_SIZE};
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                    environment, FloatBuffer.wrap(combined), shape);
             OrtSession.Result result = session.run(
                     Collections.singletonMap("combined", tensor))) {
            float[] scores = (float[]) result.get(0).getValue();
            int selected = maximumIndex(scores);
            lastSignature = signature;
            lastRecommendation = new Recommendation(
                    options.get(selected).display, scores[selected]);
            return lastRecommendation;
        }
    }

    void reset() {
        lastSignature = null;
        lastRecommendation = null;
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException ignored) {
            // The capture service is already shutting down.
        }
    }

    private static float[] encodeState(GameStateTracker tracker, String trump) {
        float[] result = new float[STATE_SIZE];
        int offset = 0;
        offset = putCards(result, offset, tracker.hand());
        offset = putCards(result, offset, tracker.attacks());
        offset = putCards(result, offset, tracker.defenses());
        offset = putCards(result, offset, tracker.discard());

        result[offset + DurakRules.suitIndex(trump)] = 1.0f;
        offset += 4;
        result[offset++] = tracker.deckCount() / 36.0f;
        // The live tracker never recommends after the defender has forfeited.
        result[offset++] = 0.0f;
        if (offset != STATE_SIZE) {
            throw new IllegalStateException("State size mismatch: " + offset);
        }
        return result;
    }

    private static int putCards(float[] target, int offset, Iterable<String> cards) {
        for (String card : cards) {
            target[offset + DurakRules.cardIndex(card)] = 1.0f;
        }
        return offset + DurakRules.CARD_COUNT;
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

    private static String signature(
            GameStateTracker tracker, String trump, List<DurakRules.MoveOption> options) {
        List<String> hand = new ArrayList<>(tracker.hand());
        List<String> discard = new ArrayList<>(tracker.discard());
        List<String> opponent = new ArrayList<>(tracker.knownOpponent());
        Collections.sort(hand);
        Collections.sort(discard);
        Collections.sort(opponent);
        StringBuilder result = new StringBuilder();
        result.append(hand).append('|').append(tracker.attacks()).append('|')
                .append(tracker.defenses()).append('|').append(discard).append('|')
                .append(opponent).append('|').append(trump).append('|')
                .append(tracker.deckCount()).append('|').append(tracker.opponentCount())
                .append('|').append(tracker.playerAttacker()).append('|')
                .append(tracker.lastEventKind()).append('|')
                .append(tracker.lastEventCards()).append('|')
                .append(tracker.lastEventActorSelf());
        for (DurakRules.MoveOption option : options) {
            result.append('|').append(option.type).append(':').append(option.card);
        }
        return result.toString();
    }

    private OrtSession createSession(Context context, String asset)
            throws IOException, OrtException {
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
}
