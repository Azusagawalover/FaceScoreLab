package com.example.facescorelab;

import android.graphics.Bitmap;
import android.graphics.PointF;
import android.graphics.Rect;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

/** 依据用户选择的参照组汇总二维几何、皮肤亮度及眼口开合统计。 */
public final class ScoringBasis {

    public static final String MODE_GENERAL = "general";
    public static final String MODE_MALE = "male";
    public static final String MODE_FEMALE = "female";

    private ScoringBasis() {}

    // ---------------------------------------------------------------- 参照组

    public static String modeLabel(String mode) {
        if (MODE_MALE.equals(mode)) return "男性参考";
        if (MODE_FEMALE.equals(mode)) return "女性参考";
        return "通用参考";
    }

    /** 该参照组侧重的可见几何项，索引对应 GeometryAnalyzer.FEATURE_NAMES。 */
    public static int[] focusFeatureIndices(String mode) {
        if (MODE_MALE.equals(mode)) {
            // 轮廓、眉眼间距、下颌线。
            return new int[]{0, 4, 12, 13};
        }
        if (MODE_FEMALE.equals(mode)) {
            // 眼型、唇形、面部柔和度（用下庭与左右相似度近似）。
            return new int[]{3, 8, 10, 11};
        }
        return new int[]{0, 1, 4, 8, 11};
    }

    public static String focusRationale(String mode) {
        if (MODE_MALE.equals(mode)) {
            return "男性参考组更侧重轮廓、眉眼间距与下颌线：面部高宽比、眉眼距/脸宽、下颌宽/脸宽、下颌角度。";
        }
        if (MODE_FEMALE.equals(mode)) {
            return "女性参考组更侧重眼型、唇形与面部柔和度：眼睛开合比、外唇高度/嘴宽、下庭占比、左右几何相似度。";
        }
        return "通用参考组不区分性别，综合面部高宽比、眼间距、眉眼距、唇形与左右相似度。";
    }

    /** 一条可展示的依据。 */
    public static final class FocusItem {
        public final String name;
        public final double value;
        public final String unit;
        public final String description;
        /** “偏高/接近/偏低该组均值”；无组内统计时为 null。 */
        public final String estimate;

        FocusItem(String name, double value, String unit, String description, String estimate) {
            this.name = name;
            this.value = value;
            this.unit = unit;
            this.description = description;
            this.estimate = estimate;
        }

        public String line() {
            String base = String.format(Locale.CHINA, "%s  %.3f %s", name, value, unit).trim();
            return estimate == null ? base : base + "  ·  " + estimate;
        }
    }

    /**
     * 按参照组挑选侧重项。
     *
     * @param values    GeometryAnalyzer.featureVector68 的输出
     * @param reference 组内参考统计；为 null 时只给测量值
     */
    public static List<FocusItem> focusItems(double[] values, ScoringModel.Prediction reference, String mode) {
        List<FocusItem> items = new ArrayList<>();
        if (values == null || values.length != GeometryAnalyzer.FEATURE_NAMES.length) return items;
        for (int index : focusFeatureIndices(mode)) {
            if (index < 0 || index >= values.length) continue;
            double value = values[index];
            if (!Double.isFinite(value)) continue;
            items.add(new FocusItem(
                    featureName(index),
                    displayValue(index, value),
                    featureUnit(index),
                    featureDescription(index),
                    estimate(index, value, reference)));
        }
        return items;
    }

    private static String estimate(int index, double value, ScoringModel.Prediction reference) {
        if (reference == null || reference.referenceFeatureMean == null
                || reference.referenceFeatureStd == null) return null;
        if (index >= reference.referenceFeatureMean.length || index >= reference.referenceFeatureStd.length) return null;
        double std = reference.referenceFeatureStd[index];
        if (!(std > 1e-9) || !Double.isFinite(reference.referenceFeatureMean[index])) return null;
        double z = (value - reference.referenceFeatureMean[index]) / std;
        if (!Double.isFinite(z)) return null;
        return Math.abs(z) < 0.35 ? "接近该组均值" : z > 0 ? "高于该组均值" : "低于该组均值";
    }

    static String featureName(int index) {
        switch (index) {
            case 0: return "面部高宽比";
            case 1: return "眼间距/脸宽";
            case 3: return "眼睛开合比";
            case 4: return "眉眼距/脸宽";
            case 8: return "外唇高度/嘴宽";
            case 10: return "下庭占比";
            case 11: return "左右平均偏差";
            case 12: return "下颌宽/脸宽";
            case 13: return "下颌角度";
            default: return index < GeometryAnalyzer.FEATURE_NAMES.length
                    ? GeometryAnalyzer.FEATURE_NAMES[index] : "特征" + index;
        }
    }

    static String featureUnit(int index) {
        switch (index) {
            case 0: case 1: case 4: case 8: case 12: return "倍";
            case 10: case 11: return "%";
            case 13: return "π";
            default: return "";
        }
    }

    /** 比例类特征按百分数展示，其余原样返回。 */
    static double displayValue(int index, double value) {
        return index == 10 || index == 11 ? value * 100 : value;
    }

    static String featureDescription(int index) {
        switch (index) {
            case 0: return "眉部均值至下巴的投影高度除以面部宽度，反映可见轮廓的纵向长度。";
            case 1: return "双眼内眼角距离除以面部宽度，属五眼比例的组成部分。";
            case 3: return "双眼标准眼睛纵横比的平均值，只描述可见开合，不代表情绪。";
            case 4: return "同侧眉毛与眼睛中心距离的平均值除以脸宽。";
            case 8: return "上下外唇中点距离除以嘴宽，受表情与拍摄角度影响。";
            case 10: return "鼻底以下的下庭投影长度占三庭的比例；发际线未手动校准时为估算值。";
            case 11: return "以眼睛轴校正倾斜后比较左右对应点，数值为平均归一化偏差，越小越对称。";
            case 12: return "左右下颌点间距除以面部宽度，近似轮廓宽度。";
            case 13: return "左右下颌点与下巴构成的夹角，近似下颌形状，不等于三维骨骼角度。";
            default: return "";
        }
    }

    // ------------------------------------------------------------ 肤色均匀度

    /** 肤色均匀度：仅基于分割出的皮肤类像素，不判断肤色深浅的“好坏”。 */
    public static final class SkinUniformity {
        public boolean available;
        /** 皮肤像素占人脸框采样点比例。 */
        public double coverage;
        /** 相对亮度的标准差；越小越均匀。 */
        public double luminanceStdDev;
        /** 0–100 的经验映射分数。 */
        public double uniformity;
        public String note = "";

        public String line() {
            if (!available) return note == null || note.isEmpty() ? "本次未计算肤色均匀度。" : note;
            return String.format(Locale.CHINA,
                    "肤色均匀度  %.1f / 100（皮肤区亮度标准差 %.3f，皮肤占比 %.1f%%）",
                    uniformity, luminanceStdDev, coverage * 100);
        }

        JSONObject toJson() throws Exception {
            JSONObject json = new JSONObject();
            json.put("available", available);
            if (available) {
                json.put("coverage", coverage);
                json.put("luminanceStdDev", luminanceStdDev);
                json.put("uniformity", uniformity);
                json.put("method", "Luminance dispersion inside BiSeNet skin class, mapped as 100*exp(-std/0.15); image statistic, not a health or skin-tone judgement.");
                json.put("methodVersion", "skin-luminance-mask-aligned-v2");
            } else {
                json.put("note", note == null ? "" : note);
            }
            return json;
        }
    }

    /**
     * 在分割裁剪图上统计肤色均匀度，并把原图中的统计区域转换到裁剪图坐标。
     * mask 必须覆盖整个 crop；sourceCrop 描述该位图在原图中的实际裁剪范围。
     */
    public static SkinUniformity skinUniformityForCrop(Bitmap crop, byte[] labels,
                                                       int labelWidth, int labelHeight,
                                                       Rect sourceCrop, float[] sourceFaceBox) {
        if (crop == null || crop.isRecycled() || sourceCrop == null
                || sourceCrop.width() != crop.getWidth() || sourceCrop.height() != crop.getHeight()) {
            SkinUniformity result = new SkinUniformity();
            result.note = "分割裁剪图与坐标范围不一致，未计算肤色均匀度。";
            return result;
        }
        float[] cropFaceBox = null;
        if (sourceFaceBox != null) {
            if (sourceFaceBox.length < 4 || !Float.isFinite(sourceFaceBox[0])
                    || !Float.isFinite(sourceFaceBox[1]) || !Float.isFinite(sourceFaceBox[2])
                    || !Float.isFinite(sourceFaceBox[3])) {
                SkinUniformity result = new SkinUniformity();
                result.note = "人脸统计区域坐标无效，未计算肤色均匀度。";
                return result;
            }
            cropFaceBox = new float[]{
                    sourceFaceBox[0] - sourceCrop.left, sourceFaceBox[1] - sourceCrop.top,
                    sourceFaceBox[2] - sourceCrop.left, sourceFaceBox[3] - sourceCrop.top
            };
        }
        return skinUniformity(crop, labels, labelWidth, labelHeight, cropFaceBox);
    }

    /**
     * 用 BiSeNet 分割的皮肤类（标签 1）估计肤色均匀度。
     *
     * @param bitmap     与 labels 对应的同一张图，不能传未裁剪的原图配裁剪图的 mask
     * @param labels     分割标签，长度为 labelWidth*labelHeight，或 null
     * @param labelWidth 分割输出宽
     * @param labelHeight 分割输出高
     * @param faceBox    人脸框 {left, top, right, bottom}，bitmap 像素坐标；null 表示整图
     */
    public static SkinUniformity skinUniformity(Bitmap bitmap, byte[] labels,
                                                int labelWidth, int labelHeight, float[] faceBox) {
        SkinUniformity result = new SkinUniformity();
        // 首页不传位图，统一提示为先运行分割，避免“图片不可用”的误导。
        if (bitmap == null || bitmap.isRecycled()) {
            result.note = "尚未运行五官分割，暂无肤色均匀度；在深度分析页运行后可查看。";
            return result;
        }
        if (labels == null || labelWidth <= 0 || labelHeight <= 0
                || labels.length < labelWidth * labelHeight) {
            result.note = "尚未运行五官分割，暂无肤色均匀度；在深度分析页运行后可查看。";
            return result;
        }
        int width = bitmap.getWidth(), height = bitmap.getHeight();
        int left = 0, top = 0, right = width - 1, bottom = height - 1;
        if (faceBox != null && faceBox.length >= 4) {
            left = Math.max(0, Math.round(faceBox[0]));
            top = Math.max(0, Math.round(faceBox[1]));
            right = Math.min(width - 1, Math.round(faceBox[2]));
            bottom = Math.min(height - 1, Math.round(faceBox[3]));
        }
        if (right <= left || bottom <= top) {
            result.note = "人脸框无效，未计算肤色均匀度。";
            return result;
        }
        // 限制采样次数，避免大图上逐像素读取带来的开销。
        long area = (long) (right - left + 1) * (bottom - top + 1);
        int step = (int) Math.max(1, Math.round(Math.sqrt(area / 40000.0)));
        int[] row = new int[width];
        double sum = 0, sumSquares = 0;
        int skin = 0, sampled = 0;
        for (int y = top; y <= bottom; y += step) {
            int labelY = Math.min(labelHeight - 1, (int) ((long) y * labelHeight / height));
            bitmap.getPixels(row, 0, width, 0, y, width, 1);
            for (int x = left; x <= right; x += step) {
                int labelX = Math.min(labelWidth - 1, (int) ((long) x * labelWidth / width));
                sampled++;
                if ((labels[labelY * labelWidth + labelX] & 0xFF) != 1) continue;  // 1 = 皮肤
                double luminance = relativeLuminance(row[x]);
                sum += luminance;
                sumSquares += luminance * luminance;
                skin++;
            }
        }
        if (skin < 64) {
            result.note = "皮肤区域过小，未计算肤色均匀度（需要更清晰的正脸照片）。";
            return result;
        }
        double mean = sum / skin;
        double variance = Math.max(0, sumSquares / skin - mean * mean);
        result.available = true;
        result.luminanceStdDev = Math.sqrt(variance);
        result.coverage = sampled <= 0 ? 0 : (double) skin / sampled;
        // 经验映射：uniformity = 100 * exp(-std / 0.15)。
        // 平滑衰减，不会因为常见的阴影或发际线而直接压到 0 分。
        // 这是课程内的相对比较尺度，不是临床或美学标准。
        result.uniformity = Math.max(0, Math.min(100, 100 * Math.exp(-result.luminanceStdDev / 0.15)));
        return result;
    }

    /** sRGB → 相对亮度。只用于均匀度统计，不做白平衡或肤色分类。 */
    private static double relativeLuminance(int color) {
        double r = ((color >> 16) & 0xFF) / 255.0;
        double g = ((color >> 8) & 0xFF) / 255.0;
        double b = (color & 0xFF) / 255.0;
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    // ------------------------------------------------------------ 表情自然度

    /** 表情自然度：判断是否接近中性放松表情，不评判好看与否。 */
    public static String expressionNaturalness(double eyeAspect, double mouthAspect) {
        if (!Double.isFinite(eyeAspect) || !Double.isFinite(mouthAspect)) {
            return "表情自然度：本次未取得眼部与嘴部开合测量。";
        }
        boolean eyesNatural = eyeAspect >= 0.12 && eyeAspect <= 0.42;
        boolean mouthNatural = mouthAspect >= 0.0 && mouthAspect <= 0.30;
        String eyeText = eyesNatural ? "眼睛开合处于常见范围"
                : eyeAspect < 0.12 ? "眼睛接近闭合" : "眼睛开合偏大";
        String mouthText = mouthNatural ? "嘴部接近放松" : "嘴部有明显张开";
        String overall = eyesNatural && mouthNatural
                ? "整体接近中性放松表情，适合作为评分输入。"
                : "表情偏离中性放松，评分会同时反映表情而非仅面部结构。";
        return String.format(Locale.CHINA, "表情自然度：%s，%s。%s", eyeText, mouthText, overall);
    }

    // ---------------------------------------------------------------- 汇总

    /**
     * 把三类依据汇总为可序列化的 JSON，供历史记录与深度页复用。
     *
     * @param values       几何特征向量
     * @param reference    几何模型的组内参考统计，可为 null
     * @param mode         用户选择的参照组
     * @param eyeAspect    眼睛开合比
     * @param mouthAspect  嘴部开合比
     * @param bitmap       原图，可为 null（则不计算肤色）
     * @param labels       分割标签，可为 null
     * @param labelWidth   分割宽
     * @param labelHeight  分割高
     * @param faceBox      人脸框
     */
    public static JSONObject toJson(double[] values, ScoringModel.Prediction reference, String mode,
                                    double eyeAspect, double mouthAspect,
                                    Bitmap bitmap, byte[] labels, int labelWidth, int labelHeight,
                                    float[] faceBox) {
        try {
            JSONObject json = new JSONObject();
            json.put("referenceMode", mode);
            json.put("referenceModeLabel", modeLabel(mode));
            json.put("focusRationale", focusRationale(mode));
            JSONArray focus = new JSONArray();
            for (FocusItem item : focusItems(values, reference, mode)) {
                JSONObject entry = new JSONObject();
                entry.put("name", item.name);
                entry.put("value", item.value);
                entry.put("unit", item.unit);
                entry.put("description", item.description);
                if (item.estimate != null) entry.put("estimate", item.estimate);
                focus.put(entry);
            }
            json.put("focus", focus);
            SkinUniformity skin = skinUniformity(bitmap, labels, labelWidth, labelHeight, faceBox);
            json.put("skinUniformity", skin.toJson());
            json.put("skinUniformityText", skin.line());
            json.put("expressionNaturalness", expressionNaturalness(eyeAspect, mouthAspect));
            json.put("disclaimer",
                    "分性别依据按用户手动选择的参照组展示，来自数据集组内统计；不从照片推断性别，"
                            + "不代表客观美丑、健康状态或对人的价值判断。");
            return json;
        } catch (Exception error) {
            return null;
        }
    }

    /** 关键点包围盒，略微外扩以覆盖额头与下巴。 */
    public static float[] faceBox(PointF[] points, int width, int height) {
        if (points == null || points.length == 0) return null;
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE, maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (PointF point : points) {
            if (point == null) continue;
            if (!Float.isFinite(point.x) || !Float.isFinite(point.y)) continue;
            minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x);
            minY = Math.min(minY, point.y); maxY = Math.max(maxY, point.y);
        }
        if (!(minX <= maxX) || !(minY <= maxY)) return null;
        float w = maxX - minX, h = maxY - minY;
        minX -= w * 0.08f; maxX += w * 0.08f;
        minY -= h * 0.10f; maxY += h * 0.06f;
        return new float[]{
                Math.max(0, minX), Math.max(0, minY),
                Math.min(width - 1, maxX), Math.min(height - 1, maxY)
        };
    }
}
