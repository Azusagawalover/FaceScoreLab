package com.example.facescorelab;

import com.google.mediapipe.tasks.components.containers.Category;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

/** 汇总 MediaPipe 的 52 项可见动作系数，生成单张照片的描述性统计。
 * 结果不用于情绪或疾病分类。
 */
public final class MicroExpressionStats {

    public static final String VERSION = "mediapipe-visible-action-stats-v1";

    /** 可见动作分组。 */
    public static final String[][] GROUPS = {
            {"嘴角上扬", "mouthSmileLeft", "mouthSmileRight"},
            {"眼周收紧", "eyeSquintLeft", "eyeSquintRight"},
            {"眉部动作", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"},
            {"眼部开合", "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight"},
            {"嘴部开合", "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker"},
            {"下颌动作", "jawLeft", "jawRight", "jawForward"}
    };

    /** 单项动作。 */
    public static final class Action {
        public final String name;
        public final double coefficient;

        Action(String name, double coefficient) {
            this.name = name;
            this.coefficient = coefficient;
        }
    }

    /**
     * 汇总结果。
     *
     * <p>注意：所有数值都是模型输出的 0–1 动作系数，<b>不是</b>概率、不是表情类别、
     * 不是任何疾病的指标。
     */
    public static final class Summary {
        public int actionCount;
        public double mean, stdDev, max, min;
        /** 平均系数最高的若干项，用于展示“这张照片里哪些动作更明显”。 */
        public final List<Action> dominant = new ArrayList<>();
        /** 分组均值。 */
        public final List<Action> groupMeans = new ArrayList<>();
        /** 左右对称性：成对动作的绝对差均值，越小越对称。 */
        public double asymmetry;
        public int symmetryPairs;

        public JSONObject toJson() throws Exception {
            JSONObject json = new JSONObject();
            json.put("model", VERSION);
            json.put("valueType", "visible_action_coefficient_not_probability");
            json.put("diagnosticUse", false);
            json.put("actionCount", actionCount);
            json.put("mean", mean);
            json.put("stdDev", stdDev);
            json.put("max", max);
            json.put("min", min);
            json.put("asymmetry", asymmetry);
            json.put("symmetryPairs", symmetryPairs);
            JSONArray top = new JSONArray();
            for (Action action : dominant) {
                JSONObject item = new JSONObject();
                item.put("name", action.name);
                item.put("coefficient", action.coefficient);
                top.put(item);
            }
            json.put("dominant", top);
            JSONArray groups = new JSONArray();
            for (Action group : groupMeans) {
                JSONObject item = new JSONObject();
                item.put("name", group.name);
                item.put("mean", group.coefficient);
                groups.put(item);
            }
            json.put("groups", groups);
            json.put("method",
                    "Descriptive summary of MediaPipe face-blendshape coefficients for one still photo. "
                            + "Not an emotion label, not a probability, and not a clinical or cognitive assessment. "
                            + "No diagnostic threshold is defined or implied.");
            json.put("ethics",
                    "本功能仅用于课程技术演示与伦理讨论。它不能诊断阿尔茨海默病、抑郁或其他疾病，"
                            + "也不得用于对真实用户做健康判断或标签化。");
            return json;
        }

        /** 一行摘要，供界面与历史详情使用。 */
        public String line() {
            if (actionCount == 0) return "本次未取得可见动作系数。";
            return String.format(Locale.CHINA,
                    "可见动作统计：%d 项系数，均值 %.3f，标准差 %.3f，最大 %.3f；左右动作差异均值 %.3f。",
                    actionCount, mean, stdDev, max, asymmetry);
        }

        /** 最明显的几项，供界面列出。 */
        public String dominantText() {
            if (dominant.isEmpty()) return "";
            StringBuilder builder = new StringBuilder("较明显的可见动作：");
            for (int i = 0; i < dominant.size(); i++) {
                if (i > 0) builder.append(" · ");
                builder.append(String.format(Locale.CHINA, "%s %.3f", dominant.get(i).name, dominant.get(i).coefficient));
            }
            return builder.toString();
        }
    }

    private MicroExpressionStats() {}

    /** 从一组 blendshape 系数构建统计。 */
    public static JSONObject from(List<Category> values) throws Exception {
        return summarize(values).toJson();
    }

    public static Summary summarize(List<Category> values) {
        Summary summary = new Summary();
        if (values == null || values.isEmpty()) return summary;
        List<Action> all = new ArrayList<>();
        for (Category value : values) {
            if (value == null) continue;
            double score = value.score();
            if (!Double.isFinite(score)) continue;
            all.add(new Action(value.categoryName(), Math.max(0, Math.min(1, score))));
        }
        if (all.isEmpty()) return summary;
        summary.actionCount = all.size();
        double sum = 0, sumSquares = 0, max = -Double.MAX_VALUE, min = Double.MAX_VALUE;
        for (Action action : all) {
            sum += action.coefficient;
            sumSquares += action.coefficient * action.coefficient;
            max = Math.max(max, action.coefficient);
            min = Math.min(min, action.coefficient);
        }
        summary.mean = sum / all.size();
        summary.stdDev = Math.sqrt(Math.max(0, sumSquares / all.size() - summary.mean * summary.mean));
        summary.max = max;
        summary.min = min;

        List<Action> sorted = new ArrayList<>(all);
        Collections.sort(sorted, (a, b) -> Double.compare(b.coefficient, a.coefficient));
        for (int i = 0; i < Math.min(5, sorted.size()); i++) summary.dominant.add(sorted.get(i));

        for (String[] group : GROUPS) {
            double total = 0;
            int count = 0;
            for (int i = 1; i < group.length; i++) {
                double coefficient = find(all, group[i]);
                if (Double.isNaN(coefficient)) continue;
                total += coefficient;
                count++;
            }
            if (count > 0) summary.groupMeans.add(new Action(group[0], total / count));
        }

        // 左右对称性：只统计确实成对给出的动作。
        double difference = 0;
        int pairs = 0;
        for (Action action : all) {
            String name = action.name;
            if (!name.endsWith("Left")) continue;
            String rightName = name.substring(0, name.length() - 4) + "Right";
            double right = find(all, rightName);
            if (Double.isNaN(right)) continue;
            difference += Math.abs(action.coefficient - right);
            pairs++;
        }
        summary.asymmetry = pairs == 0 ? 0 : difference / pairs;
        summary.symmetryPairs = pairs;
        return summary;
    }

    /** blendshape 键名 → 中文显示名；未知键名原样返回。 */
    public static String displayName(String key) {
        if (key == null) return "";
        switch (key) {
            case "mouthSmileLeft": return "左嘴角上扬";
            case "mouthSmileRight": return "右嘴角上扬";
            case "eyeSquintLeft": return "左眼周收紧";
            case "eyeSquintRight": return "右眼周收紧";
            case "browDownLeft": return "左眉下压";
            case "browDownRight": return "右眉下压";
            case "browInnerUp": return "眉内侧上扬";
            case "browOuterUpLeft": return "左眉外侧上扬";
            case "browOuterUpRight": return "右眉外侧上扬";
            case "eyeBlinkLeft": return "左眼闭合";
            case "eyeBlinkRight": return "右眼闭合";
            case "eyeWideLeft": return "左眼睁大";
            case "eyeWideRight": return "右眼睁大";
            case "jawOpen": return "下颌张开";
            case "mouthClose": return "嘴部闭合";
            case "mouthFunnel": return "嘴部呈漏斗形";
            case "mouthPucker": return "嘴部收拢";
            case "jawLeft": return "下颌左移";
            case "jawRight": return "下颌右移";
            case "jawForward": return "下颌前伸";
            case "eyeLookOutLeft": return "左眼向外看";
            case "eyeLookInRight": return "右眼向内看";
            case "eyeLookInLeft": return "左眼向内看";
            case "eyeLookOutRight": return "右眼向外看";
            case "eyeLookUpLeft": return "左眼向上看";
            case "eyeLookUpRight": return "右眼向上看";
            case "eyeLookDownLeft": return "左眼向下看";
            case "eyeLookDownRight": return "右眼向下看";
            default: return key;
        }
    }

    private static double find(List<Action> values, String name) {
        for (Action action : values) if (name.equals(action.name)) return action.coefficient;
        return Double.NaN;
    }

    /** 从已保存的 JSON 读取摘要，供历史记录展示。 */
    public static String lineFromJson(JSONObject json) {
        if (json == null) return "";
        int count = json.optInt("actionCount", 0);
        if (count <= 0) return "";
        return String.format(Locale.CHINA,
                "可见动作统计：%d 项系数，均值 %.3f，标准差 %.3f，最大 %.3f；左右动作差异均值 %.3f。",
                count, json.optDouble("mean"), json.optDouble("stdDev"),
                json.optDouble("max"), json.optDouble("asymmetry"));
    }
}
