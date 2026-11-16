package com.example.facescorelab;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

/** 对同一人脸裁剪进行图像扰动，统计评分模型的输出漂移。
 * Leuner (2019) 讨论形态与呈现条件的影响；此处不复现其分类实验。
 */
public final class RobustnessProbe {

    /** 一种扰动方式。 */
    public static final class Perturbation {
        public final String name;
        public final String description;

        Perturbation(String name, String description) {
            this.name = name;
            this.description = description;
        }
    }

    /** 一次扰动下的推理结果。 */
    public static final class Sample {
        public final String name;
        public final double value;
        public final long millis;
        public final boolean valid;

        Sample(String name, double value, long millis, boolean valid) {
            this.name = name;
            this.value = value;
            this.millis = millis;
            this.valid = valid;
        }
    }

    /** 完整演示结果。 */
    public static final class Result {
        public final List<Sample> samples = new ArrayList<>();
        public double baseline=Double.NaN;
        public double mean=Double.NaN, stdDev=Double.NaN, maxAbsDelta=Double.NaN, relativeDrift=Double.NaN;
        public int validCount, totalCount;
        public String summary = "";

        public JSONObject toJson() throws Exception {
            JSONObject json = new JSONObject();
            json.put("model", VERSION);
            json.put("baseline", finite(baseline));
            json.put("mean", finite(mean));
            json.put("stdDev", finite(stdDev));
            json.put("maxAbsDelta", finite(maxAbsDelta));
            json.put("relativeDrift", finite(relativeDrift));
            json.put("validCount", validCount);
            json.put("totalCount", totalCount);
            JSONArray array = new JSONArray();
            for (Sample sample : samples) {
                JSONObject item = new JSONObject();
                item.put("name", sample.name);
                item.put("value", finite(sample.value));
                item.put("millis", sample.millis);
                item.put("valid", sample.valid);
                array.put(item);
            }
            json.put("samples", array);
            json.put("scope",
                    "Robustness demonstration on one photo. Shows that model output is sensitive to capture conditions; "
                            + "it is not an evaluation of any person or group.");
            return json;
        }
        private static Object finite(double value){return Double.isFinite(value)?value:JSONObject.NULL;}
    }

    public static final String VERSION = "capture-condition-robustness-aligned-v2";

    /** 扰动列表：只做确定性的图像变换，不引入随机性。 */
    public static List<Perturbation> perturbations() {
        List<Perturbation> list = new ArrayList<>();
        list.add(new Perturbation("原图", "不扰动，作为基准。"));
        list.add(new Perturbation("亮度 +20%", "模拟更亮的光照。"));
        list.add(new Perturbation("亮度 -20%", "模拟更暗的光照。"));
        list.add(new Perturbation("对比度 +20%", "模拟不同的成像对比度。"));
        list.add(new Perturbation("旋转 +6°", "模拟轻微歪头。"));
        list.add(new Perturbation("旋转 -6°", "模拟另一方向歪头。"));
        list.add(new Perturbation("水平翻转", "模拟镜像自拍与真实方向的差异。"));
        list.add(new Perturbation("3×3均值模糊", "等权均值滤波的确定性图像处理。"));
        list.add(new Perturbation("中心遮挡（宽高各15%）", "覆盖面积约2.25%，模拟局部遮挡。"));
        return list;
    }

    /** 推理接口：给定图像返回 1～5 原始分。 */
    public interface Scorer {
        double score(Bitmap bitmap) throws Exception;
    }

    /**
     * 执行完整扰动演示。若人脸框不可用则只做整图变换。
     *
     * @param source  原图
     * @param crop    人脸框 {left, top, right, bottom}，可为 null
     * @param scorer  真实推理函数
     * @param progress 进度回调，可为 null
     */
    public static Result run(Bitmap source, float[] crop, Scorer scorer, OnProgress progress) {
        Result result = new Result();
        if (source == null || source.isRecycled() || scorer == null) {
            result.summary = "图片或推理引擎不可用，未执行鲁棒性演示。";
            return result;
        }
        int left = 0, top = 0, right = source.getWidth(), bottom = source.getHeight();
        if (crop != null && crop.length >= 4) {
            left = Math.max(0, Math.round(crop[0]));
            top = Math.max(0, Math.round(crop[1]));
            right = Math.min(source.getWidth(), Math.round(crop[2]));
            bottom = Math.min(source.getHeight(), Math.round(crop[3]));
        }
        if (right - left < 32 || bottom - top < 32) {
            left = 0; top = 0; right = source.getWidth(); bottom = source.getHeight();
        }
        Bitmap face;
        try {
            face = Bitmap.createBitmap(source, left, top, right - left, bottom - top);
        } catch (Exception error) {
            result.summary = "人脸区域裁剪失败，未执行鲁棒性演示。";
            return result;
        }
        List<Perturbation> steps = perturbations();
        for (int index = 0; index < steps.size(); index++) {
            Perturbation step = steps.get(index);
            if (progress != null) progress.update(step.name, index + 1, steps.size());
            Bitmap altered = null;
            long start = System.nanoTime();
            try {
                altered = transform(face, index);
                double value = scorer.score(altered);
                boolean valid = Double.isFinite(value);
                long millis = (System.nanoTime() - start) / 1000000;
                result.samples.add(new Sample(step.name, value, millis, valid));
                if (valid) result.validCount++;
            } catch (Exception error) {
                result.samples.add(new Sample(step.name, Double.NaN,
                        (System.nanoTime() - start) / 1000000, false));
            } finally {
                if (altered != null && altered != face && !altered.isRecycled()) altered.recycle();
                result.totalCount++;
            }
        }
        if (face!=source&&!face.isRecycled()) face.recycle();
        summarize(result);
        return result;
    }

    /** 汇总统计量。 */
    private static void summarize(Result result) {
        List<Double> values = new ArrayList<>();
        for (Sample sample : result.samples) if (sample.valid) values.add(sample.value);
        if (values.isEmpty()) {
            result.summary = "所有扰动下均未取得有效分数，未完成演示。";
            return;
        }
        // Missing baseline stays missing; a perturbed sample is not the original.
        result.baseline = result.samples.get(0).valid ? result.samples.get(0).value : Double.NaN;
        double sum = 0, sumSquares = 0, maxDelta = 0;
        for (double value : values) {
            sum += value;
            sumSquares += value * value;
            maxDelta = Math.max(maxDelta, Math.abs(value - result.baseline));
        }
        result.mean = sum / values.size();
        result.stdDev = Math.sqrt(Math.max(0, sumSquares / values.size() - result.mean * result.mean));
        result.maxAbsDelta = Double.isFinite(result.baseline)?maxDelta:Double.NaN;
        result.relativeDrift = Math.abs(result.baseline) > 1e-9 ? maxDelta / Math.abs(result.baseline) : Double.NaN;
        if(!Double.isFinite(result.baseline)){
            result.summary="原图未成功推理；已取得 "+values.size()+" 条扰动结果，不计算相对原图的偏移。";
            return;
        }
        result.summary = String.format(Locale.CHINA,
                "同一张脸在 %d 种拍摄条件下：基准分 %.3f，均值 %.3f，标准差 %.3f；"
                        + "最大偏移 %.3f 分（相对 %.1f%%）。",
                values.size(), result.baseline, result.mean, result.stdDev,
                result.maxAbsDelta, result.relativeDrift * 100);
    }

    /** 确定性图像变换。 */
    private static Bitmap transform(Bitmap face, int index) {
        int width = face.getWidth(), height = face.getHeight();
        switch (index) {
            case 0:
                return face.copy(Bitmap.Config.ARGB_8888, false);
            case 1:
                return adjust(face, 1.20f, 1.0f);
            case 2:
                return adjust(face, 0.80f, 1.0f);
            case 3:
                return adjust(face, 1.0f, 1.20f);
            case 4:
                return rotate(face, 6f);
            case 5:
                return rotate(face, -6f);
            case 6:
                return flip(face);
            case 7:
                return blur(face);
            case 8:
                return occlude(face, 0.15f);
            default:
                return face.copy(Bitmap.Config.ARGB_8888, false);
        }
    }

    /** 亮度与对比度调整；contrast 以 0.5 为中心缩放。 */
    private static Bitmap adjust(Bitmap source, float brightness, float contrast) {
        int width = source.getWidth(), height = source.getHeight();
        int[] pixels = new int[width * height];
        source.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int a = (pixel >>> 24) & 0xFF;
            int r = clampChannel(((pixel >> 16) & 0xFF) * brightness);
            int g = clampChannel(((pixel >> 8) & 0xFF) * brightness);
            int b = clampChannel((pixel & 0xFF) * brightness);
            if (contrast != 1.0f) {
                r = clampChannel((r - 128) * contrast + 128);
                g = clampChannel((g - 128) * contrast + 128);
                b = clampChannel((b - 128) * contrast + 128);
            }
            pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
        Bitmap output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        output.setPixels(pixels, 0, width, 0, 0, width, height);
        return output;
    }

    private static int clampChannel(float value) {
        return (int) Math.max(0, Math.min(255, Math.round(value)));
    }

    private static Bitmap rotate(Bitmap source, float degrees) {
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    private static Bitmap flip(Bitmap source) {
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postScale(-1, 1);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    /** 3×3 均值模糊，模拟失焦。 */
    private static Bitmap blur(Bitmap source) {
        int width = source.getWidth(), height = source.getHeight();
        int[] pixels = new int[width * height];
        source.getPixels(pixels, 0, width, 0, 0, width, height);
        int[] output = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = 0, g = 0, b = 0, count = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= width) continue;
                        int pixel = pixels[ny * width + nx];
                        r += (pixel >> 16) & 0xFF;
                        g += (pixel >> 8) & 0xFF;
                        b += pixel & 0xFF;
                        count++;
                    }
                }
                int a = (pixels[y * width + x] >>> 24) & 0xFF;
                output[y * width + x] = (a << 24) | ((r / count) << 16) | ((g / count) << 8) | (b / count);
            }
        }
        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);
        return result;
    }

    /** 中心区域替换为灰色，模拟局部遮挡。 */
    private static Bitmap occlude(Bitmap source, float fraction) {
        Bitmap output = source.copy(Bitmap.Config.ARGB_8888, true);
        if (output == null) return source;
        Canvas canvas = new Canvas(output);
        Paint paint = new Paint();
        paint.setColor(Color.rgb(128, 128, 128));
        float width = source.getWidth() * fraction, height = source.getHeight() * fraction;
        float left = (source.getWidth() - width) / 2f, top = (source.getHeight() - height) / 2f;
        canvas.drawRect(left, top, left + width, top + height, paint);
        return output;
    }

    public interface OnProgress {
        void update(String stepName, int index, int total);
    }

    private RobustnessProbe() {}
}
