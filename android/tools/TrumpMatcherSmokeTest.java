package ru.kartegnik.durakadvisor;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.nio.file.Path;

import javax.imageio.ImageIO;

/** Desktop-only parity check for the pure-Java Android trump matcher. */
public final class TrumpMatcherSmokeTest {
    private TrumpMatcherSmokeTest() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 7) {
            throw new IllegalArgumentException(
                    "template.bin screenshot.png expected x y width height");
        }
        BufferedImage image = ImageIO.read(Path.of(args[1]).toFile());
        int x = Integer.parseInt(args[3]);
        int y = Integer.parseInt(args[4]);
        int width = Integer.parseInt(args[5]);
        int height = Integer.parseInt(args[6]);
        int[] pixels = image.getRGB(x, y, width, height, null, 0, width);

        TrumpSuitMatcher matcher;
        try (FileInputStream stream = new FileInputStream(args[0])) {
            matcher = new TrumpSuitMatcher(stream);
        }
        TrumpSuitMatcher.Detection result = matcher.detect(pixels, width, height);
        System.out.printf(
                "%s confidence=%.3f present=%s%n",
                result.suit, result.confidence, result.cardPresent);
        boolean expectedAbsent = "-".equals(args[2]);
        if ((expectedAbsent && (result.suit != null || result.cardPresent))
                || (!expectedAbsent && (!args[2].equals(result.suit) || !result.cardPresent))) {
            throw new AssertionError("expected " + args[2]);
        }
    }
}
