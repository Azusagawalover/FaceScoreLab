package com.example.facescorelab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.SystemClock;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Collections;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

/** 基于 CelebA Male 标签训练的二分类参考模型。
 * 支持按元数据应用 Platt 校准；参照组仍由用户确认。
 * 二元标签不覆盖完整性别分布，不用于个人身份判定。
 */
public final class GenderClassifier implements AutoCloseable {
    public static final String MODEL_ASSET = "gender_classifier.onnx";
    public static final String METADATA_ASSET = "gender_model.json";
    public static final String FORMAT = "gender-classifier-mnv3-v1";
    public static final String PREPROCESS_VERSION = "mediapipe-largest-face-facecrop-v1-rgb224";
    public static final String FACE_TASK_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff";
    private static final String CROP_IMPLEMENTATION_SHA256 = "8205571ffe7e2e36511b27ff7e40fc4f104509f2fb2432660f38b3afd120a256";
    private static final Object MODEL_LOCK = new Object();
    private final Context context;
    private OrtSession session;
    private String modelHash;
    private double threshold;
    /** Platt 校准参数：p = sigmoid(a × logit + b)。 */
    private double calibrationA = 1.0, calibrationB = 0.0;
    private boolean calibrated;
    private String initializationFailure;
    private boolean closed;

    public GenderClassifier(Context context) {
        this.context = context.getApplicationContext();
    }

    /** 一次性别判断的结果。 */
    public static final class Result {
        public final boolean available;
        public final String status, modelSha256, inputTensorSha256;
        public final double logit, score, threshold;
        public final long durationMillis;
        public final Rect crop;

        private Result(String status, String hash, String inputHash, double logit, double probability,
                       double threshold, long elapsed, Rect crop) {
            this.status = status;
            available = "success".equals(status);
            modelSha256 = hash;
            inputTensorSha256 = inputHash;
            this.logit = logit;
            this.threshold = threshold;
            durationMillis = elapsed;
            this.crop = crop == null ? null : new Rect(crop);
            score = available ? probability : Double.NaN;
            if (available) {
                if (!validHash(hash) || !validHash(inputHash) || !Double.isFinite(logit)
                        || !Double.isFinite(probability) || probability < 0 || probability > 1
                        || !Double.isFinite(threshold) || threshold <= 0 || threshold >= 1
                        || crop == null || crop.width() < 32 || crop.height() < 32 || elapsed < 0) {
                    throw new IllegalArgumentException("Invalid gender result");
                }
            }
        }

        public static Result unavailable(String status) {
            if ("success".equals(status)) throw new IllegalArgumentException("Unavailable result cannot be successful");
            return new Result(status, null, null, Double.NaN, Double.NaN, Double.NaN, 0, null);
        }

        /** 模型倾向为 Male 时为 true；仅代表该二元标注下的模型输出。 */
        public boolean predictedMale() {
            return available && score >= threshold;
        }

        /** 距阈值不足 0.10 时认为判断不稳定，界面应提示不要据此下结论。 */
        public boolean uncertain() {
            return available && Math.abs(score - threshold) < 0.10;
        }

        /** 面向界面的中文描述。 */
        public String describe() {
            if (!available) {
                if ("no_face".equals(status)) return "未检测到可用人脸，未给出性别参考。";
                if ("not_installed".equals(status)) return "本版本未安装性别参考模型。";
                return "性别参考暂不可用，请换一张清晰的正面照片重试。";
            }
            String lean = predictedMale() ? "偏男性" : "偏女性";
            String text = String.format(Locale.CHINA,
                    "模型参考：%s（男性倾向 %s，判定阈值 %.2f）。",
                    lean, formatProbability(score), threshold);
            if (uncertain()) text += "该结果接近阈值，判断不稳定，不宜作为结论。";
            text += "该数值经验证集校准，但仍是单张照片的模型输出，不是对个人性别的判定。";
            return text;
        }

        /** 极端值不显示 0%/100%，避免把模型输出说成确定结论。 */
        private static String formatProbability(double value) {
            if (!Double.isFinite(value)) return "—";
            if (value < 0.01) return "<1%";
            if (value > 0.99) return ">99%";
            return String.format(Locale.CHINA, "%.0f%%", value * 100);
        }

        public JSONObject toJson() throws Exception {
            JSONObject json = new JSONObject().put("schemaVersion", 1).put("available", available).put("status", status);
            if (!available) return json;
            json.put("modelVersion", FORMAT).put("modelSha256", modelSha256)
                    .put("preprocessVersion", PREPROCESS_VERSION)
                    .put("resizeImplementation", "pillow-13-rgb-bilinear-antialias-v1")
                    .put("faceTaskSha256", FACE_TASK_SHA256).put("inputTensorSha256", inputTensorSha256)
                    .put("durationMillis", durationMillis)
                    .put("scoreType", "Platt-calibrated probability fitted on the validation split; not population-calibrated")
                    .put("thresholdFitSplit", "validation")
                    .put("logit", logit).put("score", score).put("threshold", threshold)
                    .put("predictedMale", predictedMale()).put("uncertain", uncertain());
            json.put("crop", new JSONArray(new int[]{crop.left, crop.top, crop.right, crop.bottom}));
            json.put("usagePolicy",
                    "参考信息，不自动切换几何打分参照组；不用于对个人性别的判定或公开标注。");
            return json;
        }

        public static Result fromJson(JSONObject json) throws Exception {
            if (json == null || json.optInt("schemaVersion") != 1) {
                throw new IllegalArgumentException("Unknown gender result schema");
            }
            if (!json.optBoolean("available")) return unavailable(json.optString("status"));
            if (!FORMAT.equals(json.optString("modelVersion"))
                    || !PREPROCESS_VERSION.equals(json.optString("preprocessVersion"))
                    || !"validation".equals(json.optString("thresholdFitSplit"))) {
                throw new IllegalArgumentException("Unknown gender result contract");
            }
            JSONArray box = json.getJSONArray("crop");
            if (box.length() != 4) throw new IllegalArgumentException("Invalid gender crop");
            Result result = new Result("success", json.getString("modelSha256"),
                    json.getString("inputTensorSha256"), json.getDouble("logit"),
                    json.getDouble("score"), json.getDouble("threshold"), json.getLong("durationMillis"),
                    new Rect(box.getInt(0), box.getInt(1), box.getInt(2), box.getInt(3)));
            if (json.optBoolean("predictedMale") != result.predictedMale()) {
                throw new IllegalArgumentException("Stored gender decision is inconsistent");
            }
            return result;
        }
    }

    /** 缺失或无效的可选资产不影响其他模块。 */
    public Result analyze(Bitmap photo, float[] landmarks) {
        if (closed) throw new IllegalStateException("Gender engine closed");
        if (landmarks == null || landmarks.length < 956) return Result.unavailable("no_face");
        if (initializationFailure != null) return Result.unavailable(initializationFailure);
        long start = SystemClock.elapsedRealtime();
        try {
            if (session == null) initialize();
            if (initializationFailure != null) return Result.unavailable(initializationFailure);
            Rect bounds = OnnxLocalEngine.faceCrop(photo, landmarks);
            Bitmap crop = Bitmap.createBitmap(photo, bounds.left, bounds.top, bounds.width(), bounds.height());
            float[] values;
            try {
                values = preprocess(crop);
            } finally {
                if (crop != photo) crop.recycle();
            }
            ByteBuffer exact = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            exact.asFloatBuffer().put(values);
            String inputHash = hash(exact.array());
            if ((context.getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0) {
                try (FileOutputStream output = new FileOutputStream(
                        new File(context.getCacheDir(), "deep_gender_input.f32"))) {
                    output.write(exact.array());
                } catch (java.io.IOException error) {
                    android.util.Log.w("FaceScoreLab", "Optional gender input trace unavailable", error);
                }
            }
            double logit;
            try (OnnxTensor input = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                    FloatBuffer.wrap(values), new long[]{1, 3, 224, 224});
                 OrtSession.Result outputs = session.run(Collections.singletonMap("input", input))) {
                FloatBuffer output = ((OnnxTensor) outputs.get("logit")
                        .orElseThrow(() -> new IllegalStateException("Missing logit"))).getFloatBuffer();
                if (output.remaining() != 1) throw new IllegalStateException("Expected one logit");
                logit = output.get(0);
            }
            if (!Double.isFinite(logit)) throw new IllegalStateException("Non-finite gender logit");
            double probability = sigmoid(calibrationA * logit + calibrationB);
            return new Result("success", modelHash, inputHash, logit, probability, threshold,
                    SystemClock.elapsedRealtime() - start, bounds);
        } catch (Exception error) {
            android.util.Log.w("FaceScoreLab", "Optional gender model unavailable", error);
            return Result.unavailable(session == null ? "validation_failed" : "inference_failed");
        }
    }

    private void initialize() throws Exception {
        String[] assetNames = context.getAssets().list("");
        if (assetNames == null || !java.util.Arrays.asList(assetNames).contains(MODEL_ASSET)
                || !java.util.Arrays.asList(assetNames).contains(METADATA_ASSET)) {
            initializationFailure = "not_installed";
            return;
        }
        JSONObject metadata;
        try (InputStream input = context.getAssets().open(METADATA_ASSET)) {
            metadata = new JSONObject(new String(readAll(input), StandardCharsets.UTF_8));
        } catch (java.io.FileNotFoundException missing) {
            initializationFailure = "not_installed";
            return;
        }
        validateMetadata(metadata);
        try (InputStream input = context.getAssets().open("face_landmarker.task")) {
            if (!FACE_TASK_SHA256.equals(hash(input))) {
                throw new IllegalStateException("Face landmark model hash mismatch");
            }
        }
        File model = new File(context.getFilesDir(), "gender-" + modelHash + ".onnx");
        synchronized (MODEL_LOCK) {
            boolean valid = false;
            if (model.isFile()) {
                try (InputStream input = new FileInputStream(model)) {
                    valid = modelHash.equals(hash(input));
                }
            }
            if (!valid) {
                try (InputStream input = context.getAssets().open(MODEL_ASSET);
                     FileOutputStream output = new FileOutputStream(model)) {
                    byte[] bytes = new byte[65536];
                    int count;
                    while ((count = input.read(bytes)) != -1) output.write(bytes, 0, count);
                } catch (java.io.FileNotFoundException missing) {
                    initializationFailure = "not_installed";
                    return;
                }
                try (InputStream input = new FileInputStream(model)) {
                    if (!modelHash.equals(hash(input))) {
                        throw new IllegalStateException("Gender model hash mismatch");
                    }
                }
            }
        }
        OrtSession loaded;
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(2);
            options.setInterOpNumThreads(1);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setCPUArenaAllocator(false);
            options.setMemoryPatternOptimization(false);
            loaded = OrtEnvironment.getEnvironment().createSession(model.getAbsolutePath(), options);
        }
        try {
            if (loaded.getInputInfo().size() != 1 || loaded.getOutputInfo().size() != 1
                    || !loaded.getInputInfo().containsKey("input")
                    || !loaded.getOutputInfo().containsKey("logit")) {
                throw new IllegalArgumentException("Gender ONNX names differ");
            }
            TensorInfo input = (TensorInfo) loaded.getInputInfo().get("input").getInfo();
            TensorInfo output = (TensorInfo) loaded.getOutputInfo().get("logit").getInfo();
            long[] in = input.getShape(), out = output.getShape();
            if (input.type != OnnxJavaType.FLOAT || output.type != OnnxJavaType.FLOAT
                    || in.length != 4 || out.length != 2
                    || in[1] != 3 || in[2] != 224 || in[3] != 224 || out[1] != 1
                    || (in[0] > 0 && in[0] != 1) || (out[0] > 0 && out[0] != 1)) {
                throw new IllegalArgumentException("Gender ONNX dimensions differ");
            }
            session = loaded;
        } catch (Exception error) {
            loaded.close();
            throw error;
        }
    }

    private void validateMetadata(JSONObject metadata) throws Exception {
        if (!FORMAT.equals(metadata.getString("format"))
                || !MODEL_ASSET.equals(metadata.getString("model_file"))
                || !"validation".equals(metadata.getString("threshold_fit_split"))
                || !"sigmoid(a*logit + b) >= threshold -> male else female".equals(metadata.getString("decision"))) {
            throw new IllegalArgumentException("Unsupported gender metadata");
        }
        modelHash = metadata.getString("model_sha256");
        if (!validHash(modelHash)) throw new IllegalArgumentException("Invalid model hash");
        threshold = metadata.getDouble("threshold");
        if (!Double.isFinite(threshold) || threshold <= 0 || threshold >= 1) {
            throw new IllegalArgumentException("Invalid gender threshold");
        }
        JSONObject calibration = metadata.optJSONObject("calibration");
        if (calibration != null) {
            calibrationA = calibration.getDouble("a");
            calibrationB = calibration.getDouble("b");
            if (!Double.isFinite(calibrationA) || !Double.isFinite(calibrationB) || calibrationA <= 0) {
                throw new IllegalArgumentException("Invalid gender calibration");
            }
            calibrated = true;
        }
        JSONArray labels = metadata.getJSONArray("labels");
        if (labels.length() != 2 || !"female".equals(labels.getString(0)) || !"male".equals(labels.getString(1))) {
            throw new IllegalArgumentException("Gender label order invalid");
        }
        JSONObject input = metadata.getJSONObject("input"), output = metadata.getJSONObject("output");
        JSONArray in = input.getJSONArray("shape"), out = output.getJSONArray("shape");
        if (!"input".equals(input.getString("name")) || !"float32".equals(input.getString("dtype"))
                || !"NCHW".equals(input.getString("layout"))
                || in.length() != 4 || !"N".equals(in.getString(0)) || in.getInt(1) != 3
                || in.getInt(2) != 224 || in.getInt(3) != 224
                || !"logit".equals(output.getString("name")) || out.length() != 2
                || !"N".equals(out.getString(0)) || out.getInt(1) != 1
                || !isSupportedActivation(output.getString("activation"))) {
            throw new IllegalArgumentException("Unsupported gender tensor contract");
        }
        JSONObject preprocessing = metadata.getJSONObject("preprocessing");
        if (!PREPROCESS_VERSION.equals(preprocessing.getString("version"))
                || !FACE_TASK_SHA256.equals(preprocessing.getString("face_task_sha256"))
                || !CROP_IMPLEMENTATION_SHA256.equals(preprocessing.getString("crop_implementation_sha256"))
                || !"largest landmark bounding-box area; first on ties".equals(preprocessing.getString("face_selection"))
                || !"224x224 bilinear".equals(preprocessing.getString("resize"))
                || !"RGB".equals(preprocessing.getString("color"))
                || !"float32".equals(preprocessing.getString("dtype"))
                || !"NCHW contiguous".equals(preprocessing.getString("layout"))
                || !"divide by 255".equals(preprocessing.getString("pixel_scale"))) {
            throw new IllegalArgumentException("Gender preprocessing contract mismatch");
        }
        JSONArray mean = preprocessing.getJSONArray("mean"), std = preprocessing.getJSONArray("std");
        float[] expectedMean = {.485f, .456f, .406f}, expectedStd = {.229f, .224f, .225f};
        if (mean.length() != 3 || std.length() != 3) {
            throw new IllegalArgumentException("Invalid normalization dimensions");
        }
        for (int i = 0; i < 3; i++) {
            if (!Double.isFinite(mean.getDouble(i)) || !Double.isFinite(std.getDouble(i))
                    || Math.abs(mean.getDouble(i) - expectedMean[i]) > 1e-7
                    || Math.abs(std.getDouble(i) - expectedStd[i]) > 1e-7) {
                throw new IllegalArgumentException("Gender normalization differs");
            }
        }
    }

    private static boolean isSupportedActivation(String value) {
        return "sigmoid".equals(value) || "logit + Platt calibration".equals(value);
    }

    private static float[] preprocess(Bitmap crop) {
        int[] original = new int[crop.getWidth() * crop.getHeight()];
        crop.getPixels(original, 0, crop.getWidth(), 0, 0, crop.getWidth(), crop.getHeight());
        int[] pixels = VisibleAttributeResize.resize(original, crop.getWidth(), crop.getHeight(), 224, 224);
        return VisibleAttributeResize.imageNetTensor(pixels);
    }

    public static double sigmoid(double logit) {
        if (logit >= 0) return 1 / (1 + Math.exp(-logit));
        double value = Math.exp(logit);
        return value / (1 + value);
    }

    private static boolean validHash(String value) {
        return value != null && value.matches("[0-9a-f]{64}");
    }

    private static String hash(byte[] bytes) throws Exception {
        return hex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    private static String hash(InputStream input) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] bytes = new byte[65536];
        int count;
        while ((count = input.read(bytes)) != -1) digest.update(bytes, 0, count);
        return hex(digest.digest());
    }

    private static String hex(byte[] bytes) {
        StringBuilder output = new StringBuilder();
        for (byte value : bytes) output.append(String.format(Locale.ROOT, "%02x", value & 255));
        return output.toString();
    }

    private static byte[] readAll(InputStream input) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] bytes = new byte[4096];
        int count;
        while ((count = input.read(bytes)) != -1) output.write(bytes, 0, count);
        return output.toByteArray();
    }

    @Override
    public void close() {
        closed = true;
        if (session != null) {
            try {
                session.close();
            } catch (Exception error) {
                android.util.Log.w("FaceScoreLab", "Gender session close failed", error);
            }
        }
        session = null;
    }
}
