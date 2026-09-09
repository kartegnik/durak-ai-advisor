package ru.kartegnik.durakadvisor;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/** Runs the recurrent PPO v4 policy with its separate hidden-card memory. */
final class DurakPolicyAdvisor implements AutoCloseable {
    private static final int OBSERVATION_SIZE = 284;
    private static final int HIDDEN_SIZE = 192;
    private static final int EVENT_SIZE = 45;

    static final class Recommendation {
        final String action;
        final float score;
        final int type;
        final String card;

        Recommendation(String action, float score, int type, String card) {
            this.action = action;
            this.score = score;
            this.type = type;
            this.card = card;
        }
    }

    private final OrtEnvironment environment;
    private final OrtSession session;
    private float[] policyHidden = new float[HIDDEN_SIZE];
    private float[] beliefHidden = new float[HIDDEN_SIZE];
    private String lastSignature;
    private Recommendation lastRecommendation;

    DurakPolicyAdvisor(Context context) throws IOException {
        environment = OrtEnvironment.getEnvironment();
        try {
            session = createSession(context, "models/durak_policy_v4.onnx");
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

        float[] observation = encodeObservation(tracker, trump);
        float[] encodedOptions = new float[options.size() * DurakRules.OPTION_SIZE];
        for (int index = 0; index < options.size(); index++) {
            float[] encoded = options.get(index).encode();
            System.arraycopy(
                    encoded, 0, encodedOptions,
                    index * DurakRules.OPTION_SIZE, DurakRules.OPTION_SIZE);
        }

        try (OnnxTensor observationTensor = OnnxTensor.createTensor(
                    environment, FloatBuffer.wrap(observation),
                    new long[]{OBSERVATION_SIZE});
             OnnxTensor optionsTensor = OnnxTensor.createTensor(
                    environment, FloatBuffer.wrap(encodedOptions),
                    new long[]{options.size(), DurakRules.OPTION_SIZE});
             OnnxTensor policyHiddenTensor = OnnxTensor.createTensor(
                    environment, FloatBuffer.wrap(policyHidden),
                    new long[]{HIDDEN_SIZE});
             OnnxTensor beliefHiddenTensor = OnnxTensor.createTensor(
                    environment, FloatBuffer.wrap(beliefHidden),
                    new long[]{HIDDEN_SIZE})) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("observation", observationTensor);
            inputs.put("options", optionsTensor);
            inputs.put("policy_hidden", policyHiddenTensor);
            inputs.put("belief_hidden", beliefHiddenTensor);
            try (OrtSession.Result result = session.run(inputs)) {
                float[] scores = (float[]) result.get(0).getValue();
                float[] nextPolicyHidden = (float[]) result.get(1).getValue();
                float[] nextBeliefHidden = (float[]) result.get(2).getValue();
                int selected = maximumIndex(scores);
                policyHidden = nextPolicyHidden.clone();
                beliefHidden = nextBeliefHidden.clone();
                lastSignature = signature;
                DurakRules.MoveOption choice = options.get(selected);
                lastRecommendation = new Recommendation(
                        choice.display, scores[selected], choice.type, choice.card);
                return lastRecommendation;
            }
        }
    }

    void reset() {
        policyHidden = new float[HIDDEN_SIZE];
        beliefHidden = new float[HIDDEN_SIZE];
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

    private static float[] encodeObservation(GameStateTracker tracker, String trump) {
        float[] result = new float[OBSERVATION_SIZE];
        int offset = 0;
        offset = putCards(result, offset, tracker.hand());
        offset = putCards(result, offset, tracker.attacks());
        offset = putCards(result, offset, tracker.defenses());
        offset = putCards(result, offset, tracker.discard());
        offset = putCards(result, offset, tracker.knownOpponent());

        boolean[] visible = new boolean[DurakRules.CARD_COUNT];
        mark(visible, tracker.hand());
        mark(visible, tracker.attacks());
        mark(visible, tracker.defenses());
        mark(visible, tracker.discard());
        mark(visible, tracker.knownOpponent());
        for (int index = 0; index < visible.length; index++) {
            result[offset + index] = visible[index] ? 0.0f : 1.0f;
        }
        offset += DurakRules.CARD_COUNT;

        result[offset + DurakRules.suitIndex(trump)] = 1.0f;
        offset += 4;
        // The face-up trump rank is not read on Android: nine zeros + known flag.
        offset += 9;
        result[offset++] = 0.0f;

        result[offset++] = Math.min(24, tracker.deckCount()) / 24.0f;
        result[offset++] = tracker.hand().size() / 36.0f;
        result[offset++] = tracker.opponentCount() / 36.0f;
        result[offset++] = tracker.attacks().size() / 6.0f;
        result[offset++] = tracker.defenses().size() / 6.0f;
        // When the opening role is unknown, the displayed recommendation is
        // explicitly conditional on it being our attack, so encode that case.
        boolean isAttacker = !Boolean.FALSE.equals(tracker.playerAttacker());
        result[offset++] = isAttacker ? 1.0f : 0.0f;
        result[offset++] = isAttacker ? 0.0f : 1.0f;
        result[offset++] = 0.0f;
        result[offset++] = tracker.defenses().isEmpty() ? 0.0f : 1.0f;

        Integer eventKind = tracker.lastEventKind();
        if (eventKind != null) {
            result[offset + eventKind] = 1.0f;
            for (String card : tracker.lastEventCards()) {
                result[offset + 7 + DurakRules.cardIndex(card)] = 1.0f;
            }
            Boolean actor = tracker.lastEventActorSelf();
            if (actor != null) {
                result[offset + 7 + DurakRules.CARD_COUNT + (actor ? 0 : 1)] = 1.0f;
            }
        }
        offset += EVENT_SIZE;
        if (offset != OBSERVATION_SIZE) {
            throw new IllegalStateException("Observation size mismatch: " + offset);
        }
        return result;
    }

    private static int putCards(float[] target, int offset, Iterable<String> cards) {
        for (String card : cards) {
            target[offset + DurakRules.cardIndex(card)] = 1.0f;
        }
        return offset + DurakRules.CARD_COUNT;
    }

    private static void mark(boolean[] target, Iterable<String> cards) {
        for (String card : cards) {
            target[DurakRules.cardIndex(card)] = true;
        }
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
