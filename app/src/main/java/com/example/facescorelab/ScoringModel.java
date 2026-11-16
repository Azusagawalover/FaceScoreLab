package com.example.facescorelab;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** Read-only inference for the actual trained MEBeauty geometric ridge model. */
public final class ScoringModel {
    public final String version;
    public final String[] names;
    public final String[] labels;
    public final double intercept;
    private final double[] lower, upper, mean, std, coefficients;
    private final JSONObject references;

    public ScoringModel(Context context) {
        try (InputStream input = context.getAssets().open("score_model.json");
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int count;
            while ((count = input.read(buffer)) != -1) bytes.write(buffer, 0, count);
            JSONObject root = new JSONObject(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
            version = root.getString("version");
            names = strings(root.getJSONArray("features"));
            labels = strings(root.getJSONArray("feature_labels"));
            lower = numbers(root.getJSONArray("lower"));
            upper = numbers(root.getJSONArray("upper"));
            mean = numbers(root.getJSONArray("mean"));
            std = numbers(root.getJSONArray("std"));
            coefficients = numbers(root.getJSONArray("coefficients"));
            intercept = root.getDouble("intercept");
            references = root.getJSONObject("references");
            if (names.length != GeometryAnalyzer.FEATURE_NAMES.length || labels.length != names.length
                    || lower.length != names.length || upper.length != names.length
                    || mean.length != names.length || std.length != names.length || coefficients.length != names.length) {
                throw new IllegalArgumentException("评分模型的特征维度不匹配");
            }
            for (int i=0; i<names.length; i++) {
                if (!names[i].equals(GeometryAnalyzer.FEATURE_NAMES[i]) || std[i] <= 0
                        || !Double.isFinite(std[i]) || lower[i] > upper[i]) {
                    throw new IllegalArgumentException("评分模型的参数或特征版本不匹配");
                }
            }
        } catch (Exception error) {
            throw new IllegalStateException("本地评分模型无法加载，请重新安装完整应用。", error);
        }
    }

    public Prediction predict(double[] values, String referenceMode) {
        if (values == null || values.length != names.length) throw new IllegalArgumentException("几何特征不完整");
        String mode = "male".equals(referenceMode) || "female".equals(referenceMode) ? referenceMode : "general";
        try {
            JSONObject reference = references.getJSONObject(mode);
            Prediction result = new Prediction();
            result.referenceMode = mode;
            result.referenceMean = reference.getDouble("score_mean");
            result.referenceCount = reference.getInt("n");
            result.referenceFeatureMean = numbers(reference.getJSONArray("feature_mean"));
            result.referenceFeatureStd = numbers(reference.getJSONArray("feature_std"));
            result.contributions = new double[values.length];
            result.standardized = new double[values.length];
            double total = intercept;
            for (int i=0; i<values.length; i++) {
                if (!Double.isFinite(values[i])) throw new IllegalArgumentException("面部特征包含无效数值");
                if (values[i] < lower[i] || values[i] > upper[i]) result.clippedFeatures++;
                double clipped = Math.max(lower[i], Math.min(upper[i], values[i]));
                result.standardized[i] = (clipped-mean[i])/std[i];
                result.contributions[i] = result.standardized[i]*coefficients[i];
                total += result.contributions[i];
            }
            result.rawScore = total;
            result.score = Math.max(1, Math.min(10, total));
            return result;
        } catch (JSONException error) {
            throw new IllegalStateException("本地评分模型的参考统计不完整。", error);
        }
    }

    private static String[] strings(JSONArray array) throws JSONException {
        String[] values = new String[array.length()];
        for(int i=0; i<values.length; i++) values[i] = array.getString(i);
        return values;
    }

    private static double[] numbers(JSONArray array) throws JSONException {
        double[] values = new double[array.length()];
        for(int i=0; i<values.length; i++) values[i] = array.getDouble(i);
        return values;
    }

    public static final class Prediction {
        public double score, rawScore, referenceMean;
        public int referenceCount, clippedFeatures;
        public String referenceMode;
        public double[] contributions, standardized, referenceFeatureMean, referenceFeatureStd;
    }
}
