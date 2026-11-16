package com.example.facescorelab;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Rect;

/** Synthetic coordinate/mask contracts; no face photos, model runs or user history. */
public final class SkinUniformityContractChecks {
    private static final int WIDTH = 120, HEIGHT = 100;
    private static final int MASK_WIDTH = 60, MASK_HEIGHT = 50;
    private static final float[] LOCAL_FACE_BOX = {20, 16, 100, 88};

    private SkinUniformityContractChecks() {}

    /** The same face patch must produce the same statistic at different original-image positions. */
    public static void verifyTranslatedFaceBox() {
        Bitmap patch = facePatch();
        byte[] mask = skinMask();
        int[][] placements = {
                {160, 140, 20, 20, Color.RED},
                {400, 260, 8, 7, Color.GREEN},
                {1200, 900, 1030, 770, Color.BLUE}
        };
        try {
            for (int[] placement : placements) {
                Bitmap original = Bitmap.createBitmap(placement[0], placement[1], Bitmap.Config.ARGB_8888);
                Bitmap crop = null;
                try {
                    original.eraseColor(placement[4]);
                    new Canvas(original).drawBitmap(patch, placement[2], placement[3], null);
                    Rect sourceCrop = new Rect(placement[2], placement[3],
                            placement[2] + WIDTH, placement[3] + HEIGHT);
                    crop = Bitmap.createBitmap(original, sourceCrop.left, sourceCrop.top, WIDTH, HEIGHT);
                    float[] sourceBox = translatedBox(sourceCrop.left, sourceCrop.top);
                    ScoringBasis.SkinUniformity measured = ScoringBasis.skinUniformityForCrop(
                            crop, mask, MASK_WIDTH, MASK_HEIGHT, sourceCrop, sourceBox);
                    assertKnownStatistic(measured);
                } finally {
                    if (crop != null && crop != original) crop.recycle();
                    original.recycle();
                }
            }
        } finally {
            patch.recycle();
        }
    }

    /** Non-skin pixels inside the sampled face rectangle must not affect the statistic. */
    public static void verifyNonSkinExclusion() {
        Bitmap patch = facePatch();
        try {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    if (x < 20 || x >= 100 || y < 16 || y >= 88) {
                        patch.setPixel(x, y, ((x + y) & 1) == 0 ? Color.BLACK : Color.WHITE);
                    }
                }
            }
            ScoringBasis.SkinUniformity measured = ScoringBasis.skinUniformityForCrop(
                    patch, skinMask(), MASK_WIDTH, MASK_HEIGHT,
                    new Rect(0, 0, WIDTH, HEIGHT), LOCAL_FACE_BOX);
            assertKnownStatistic(measured);
        } finally {
            patch.recycle();
        }
    }

    /** Below 64 skin samples is unavailable, rather than a fabricated perfect score. */
    public static void verifyInsufficientSkin() {
        Bitmap patch = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888);
        patch.eraseColor(Color.rgb(128, 128, 128));
        try {
            byte[] labels = new byte[32 * 32];
            for (int y = 0; y < 7; y++) for (int x = 0; x < 8; x++) labels[y * 32 + x] = 1;
            ScoringBasis.SkinUniformity insufficient = ScoringBasis.skinUniformityForCrop(
                    patch, labels, 32, 32, new Rect(0, 0, 32, 32), null);
            require(!insufficient.available && insufficient.note.contains("皮肤区域过小"),
                    "56 skin samples must be unavailable");
            for (int x = 0; x < 8; x++) labels[7 * 32 + x] = 1;
            ScoringBasis.SkinUniformity enough = ScoringBasis.skinUniformityForCrop(
                    patch, labels, 32, 32, new Rect(0, 0, 32, 32), null);
            require(enough.available, "64 skin samples must be available");
            // The existing sum-of-squares estimator may leave tiny floating-point roundoff.
            near(enough.luminanceStdDev, 0.0, 1e-7, "constant skin has zero dispersion");
            near(enough.uniformity, 100.0, 1e-4, "constant skin mapping");
            near(enough.coverage, 64.0 / (32 * 32), 1e-12, "coverage includes non-skin samples");
        } finally {
            patch.recycle();
        }
    }

    /** Never silently treat incompatible crop geometry or non-finite regions as aligned. */
    public static void verifyInvalidCoordinateSpace() {
        Bitmap patch = facePatch();
        try {
            ScoringBasis.SkinUniformity wrongSize = ScoringBasis.skinUniformityForCrop(
                    patch, skinMask(), MASK_WIDTH, MASK_HEIGHT,
                    new Rect(200, 100, 200 + WIDTH + 1, 100 + HEIGHT), translatedBox(200, 100));
            require(!wrongSize.available && wrongSize.note.contains("坐标范围不一致"),
                    "a crop rectangle with a different width must be rejected");
            ScoringBasis.SkinUniformity invalidBox = ScoringBasis.skinUniformityForCrop(
                    patch, skinMask(), MASK_WIDTH, MASK_HEIGHT,
                    new Rect(0, 0, WIDTH, HEIGHT), new float[]{Float.NaN, 16, 100, 88});
            require(!invalidBox.available && invalidBox.note.contains("坐标无效"),
                    "a non-finite sampling region must be rejected");
        } finally {
            patch.recycle();
        }
    }

    private static Bitmap facePatch() {
        Bitmap patch = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888);
        patch.eraseColor(Color.MAGENTA);
        for (int y = 16; y < 88; y++) {
            for (int x = 20; x < 100; x++) {
                int value = x < 60 ? 128 : 192;
                patch.setPixel(x, y, Color.rgb(value, value, value));
            }
        }
        return patch;
    }

    private static byte[] skinMask() {
        byte[] mask = new byte[MASK_WIDTH * MASK_HEIGHT];
        for (int y = 8; y < 44; y++) {
            for (int x = 10; x < 50; x++) mask[y * MASK_WIDTH + x] = 1;
        }
        return mask;
    }

    private static float[] translatedBox(int left, int top) {
        return new float[]{left + LOCAL_FACE_BOX[0], top + LOCAL_FACE_BOX[1],
                left + LOCAL_FACE_BOX[2], top + LOCAL_FACE_BOX[3]};
    }

    private static void assertKnownStatistic(ScoringBasis.SkinUniformity measured) {
        require(measured.available, "synthetic skin region must be measurable: " + measured.note);
        // 80*72 skin pixels, half gray128 and half gray192. The sampled rectangle is inclusive.
        double expectedStd = 32.0 / 255.0;
        near(measured.luminanceStdDev, expectedStd, 1e-9, "known two-tone skin dispersion");
        near(measured.coverage, (80.0 * 72) / (81 * 73), 1e-12, "known mask coverage");
        near(measured.uniformity, 100.0 * Math.exp(-expectedStd / 0.15), 1e-7, "known mapping");
    }

    private static void near(double actual, double expected, double tolerance, String label) {
        require(Double.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
                label + ": expected " + expected + ", actual " + actual);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
