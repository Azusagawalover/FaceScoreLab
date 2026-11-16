package com.example.facescorelab;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** 基率效应、Wilson 区间、类别不平衡和 E-value 的统计计算。
 * 仅处理输入统计量，不包含人群分类模型。
 */
public final class MethodologyAudit {

    /** 常用双侧置信水平对应的 z 值。 */
    public static double zFor(double confidence) {
        if (confidence >= 0.999) return 3.290527;
        if (confidence >= 0.99) return 2.575829;
        if (confidence >= 0.95) return 1.959964;
        if (confidence >= 0.90) return 1.644854;
        if (confidence >= 0.80) return 1.281552;
        return 1.959964;
    }

    // ------------------------------------------------------------ 1. 基率效应

    /** 由敏感度、特异度与先验比例计算阳性预测值等指标。 */
    public static final class PredictiveValues {
        public double prevalence, sensitivity, specificity;
        /** 阳性预测值：检出的阳性里真阳性的比例。 */
        public double ppv;
        /** 阴性预测值。 */
        public double npv;
        /** 每 1000 人中的真阳性、假阳性、假阴性、真阴性人数。 */
        public double truePositive, falsePositive, falseNegative, trueNegative;
        /** 每检出 1 个真阳性需要伴随多少假阳性。 */
        public double falsePerTrue;

        public String summary() {
            return String.format(Locale.CHINA,
                    "先验比例 %.2f%%、敏感度 %.1f%%、特异度 %.1f%% → 阳性预测值 %.1f%%；"
                            + "每 1000 人中真阳性 %.1f 例、假阳性 %.1f 例，"
                            + "即每检出 1 个真阳性会伴随 %.1f 个假阳性。",
                    prevalence * 100, sensitivity * 100, specificity * 100, ppv * 100,
                    truePositive, falsePositive, falsePerTrue);
        }
    }

    /**
     * 基率效应计算。
     *
     * @param prevalence  目标特征在该人群中的先验比例，0–1
     * @param sensitivity 敏感度（真阳性率），0–1
     * @param specificity 特异度（真阴性率），0–1
     */
    public static PredictiveValues baseRate(double prevalence, double sensitivity, double specificity) {
        PredictiveValues result = new PredictiveValues();
        result.prevalence = clamp01(prevalence);
        result.sensitivity = clamp01(sensitivity);
        result.specificity = clamp01(specificity);
        double p = result.prevalence, se = result.sensitivity, sp = result.specificity;
        double positiveRate = se * p + (1 - sp) * (1 - p);
        result.ppv = positiveRate > 1e-12 ? se * p / positiveRate : Double.NaN;
        double negativeRate = (1 - se) * p + sp * (1 - p);
        result.npv = negativeRate > 1e-12 ? sp * (1 - p) / negativeRate : Double.NaN;
        result.truePositive = 1000 * se * p;
        result.falsePositive = 1000 * (1 - sp) * (1 - p);
        result.falseNegative = 1000 * (1 - se) * p;
        result.trueNegative = 1000 * sp * (1 - p);
        result.falsePerTrue = result.truePositive > 1e-9
                ? result.falsePositive / result.truePositive : Double.POSITIVE_INFINITY;
        return result;
    }

    // ---------------------------------------------------------- 2. 置信区间

    /** 比例的 Wilson 得分置信区间；小样本下比正态近似更可靠。 */
    public static final class Interval {
        public double lower, upper, point;
        public int sampleSize;
        public double confidence;
        public double width;

        public String summary() {
            return String.format(Locale.CHINA,
                    "n=%d、点估计 %.1f%% → %.0f%% 置信区间 [%.1f%%, %.1f%%]，宽度 %.1f 个百分点。",
                    sampleSize, point * 100, confidence * 100,
                    lower * 100, upper * 100, width * 100);
        }
    }

    /**
     * @param successes  成功数（或正确数）
     * @param total      样本总数
     * @param confidence 置信水平，如 0.95
     */
    public static Interval wilson(int successes, int total, double confidence) {
        Interval result = new Interval();
        result.sampleSize = Math.max(0, total);
        result.confidence = confidence;
        if (total <= 0) {
            result.point = result.lower = result.upper = Double.NaN;
            return result;
        }
        int x = Math.max(0, Math.min(total, successes));
        double n = total, z = zFor(confidence);
        double p = x / n;
        double denominator = 1 + z * z / n;
        double center = (p + z * z / (2 * n)) / denominator;
        double margin = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator;
        result.point = p;
        result.lower = Math.max(0, center - margin);
        result.upper = Math.min(1, center + margin);
        result.width = result.upper - result.lower;
        return result;
    }

    /** 在给定准确率下，要报告多窄的区间需要多少样本。 */
    public static int sampleSizeForHalfWidth(double accuracy, double halfWidth, double confidence) {
        if (!(halfWidth > 1e-6) || !(accuracy > 0) || !(accuracy < 1)) return -1;
        double z = zFor(confidence);
        double n = accuracy * (1 - accuracy) * z * z / (halfWidth * halfWidth);
        return (int) Math.ceil(n);
    }

    // ------------------------------------------------------- 3. 类别不平衡

    /** 类别不平衡下的“准确率幻觉”。 */
    public static final class ImbalanceAudit {
        public double positiveRate;
        /** 原始准确率（论文常报告的量）。 */
        public double accuracy;
        /** 多数类基线：什么都不学也会得到的准确率。 */
        public double majorityBaseline;
        public double sensitivity, specificity, balancedAccuracy;
        /** 相对多数类基线的真实提升。 */
        public double liftOverBaseline;

        public String summary() {
            return String.format(Locale.CHINA,
                    "阳性占比 %.1f%% 时，多数类基线已达 %.1f%%；报告的准确率 %.1f%% 相对基线只提升 %.1f 个百分点。"
                            + "平衡准确率（敏感度与特异度均值）为 %.1f%%。",
                    positiveRate * 100, majorityBaseline * 100, accuracy * 100,
                    liftOverBaseline * 100, balancedAccuracy * 100);
        }
    }

    /**
     * @param positiveRate 阳性类在样本中的占比
     * @param accuracy     论文报告的原始准确率
     * @param sensitivity  敏感度
     */
    public static ImbalanceAudit imbalance(double positiveRate, double accuracy, double sensitivity) {
        ImbalanceAudit result = new ImbalanceAudit();
        result.positiveRate = clamp01(positiveRate);
        result.accuracy = clamp01(accuracy);
        result.sensitivity = clamp01(sensitivity);
        double p = result.positiveRate, a = result.accuracy, se = result.sensitivity;
        result.majorityBaseline = Math.max(p, 1 - p);
        result.liftOverBaseline = a - result.majorityBaseline;
        // 由准确率与敏感度反解特异度：a = se*p + sp*(1-p)
        double sp = (1 - p) > 1e-9 ? (a - se * p) / (1 - p) : Double.NaN;
        result.specificity = Double.isFinite(sp) ? clamp01(sp) : Double.NaN;
        result.balancedAccuracy = Double.isFinite(result.specificity)
                ? (se + result.specificity) / 2 : Double.NaN;
        return result;
    }

    // ------------------------------------------------------- 4. E-value 混淆

    /** 未测量混淆因素的 E-value：需要多强的混淆才能解释掉一个关联。 */
    public static final class ConfoundingAudit {
        /** 风险比（或由准确率差换算的风险比）。 */
        public double riskRatio;
        /** E-value 本身：混淆因素需与暴露、结局都达到这个风险比。 */
        public double evalue;
        /** 若存在这样的混淆，真实关联可被完全解释掉。 */
        public String interpretation;

        public String summary() {
            if (!Double.isFinite(evalue)) {
                return String.format(Locale.CHINA, "观测风险比 %.3f。%s", riskRatio, interpretation);
            }
            return String.format(Locale.CHINA,
                    "观测风险比 %.3f → E-value %.3f。%s",
                    riskRatio, evalue, interpretation);
        }
    }

    /**
     * 计算 E-value（VanderWeele & Ding, 2017）。
     *
     * <p>风险比小于 1 表示模型没有超过对照基线，此时不存在“需要混淆因素去解释”
     * 的正向关联，本方法明确返回该结论，而不是取倒数制造一个虚假的 E-value。
     *
     * @param riskRatio 观测到的风险比（模型相对基线）
     */
    public static ConfoundingAudit evalue(double riskRatio) {
        ConfoundingAudit result = new ConfoundingAudit();
        if (!Double.isFinite(riskRatio) || riskRatio <= 0) {
            result.riskRatio = Double.NaN;
            result.evalue = Double.NaN;
            result.interpretation = "风险比无效，无法计算。";
            return result;
        }
        result.riskRatio = riskRatio;
        if (riskRatio <= 1 + 1e-9) {
            result.evalue = Double.NaN;
            result.interpretation = "模型未超过对照基线（风险比不大于 1），"
                    + "本就没有可被混淆因素解释的正向关联，E-value 不适用。";
            return result;
        }
        double rr = riskRatio;
        result.evalue = rr + Math.sqrt(rr * (rr - 1));
        if (result.evalue < 1.5) {
            result.interpretation = "E-value 很小：很弱的混淆因素就能解释掉这个关联，结论不稳健。";
        } else if (result.evalue < 2.5) {
            result.interpretation = "E-value 中等：中等强度的混淆因素即可解释掉该关联。";
        } else {
            result.interpretation = "E-value 较大：需要相当强的混淆因素才能解释掉该关联。";
        }
        return result;
    }

    /**
     * 由两组准确率之差换算近似风险比，用于把“准确率提升”放进 E-value 框架。
     * 这是课程内的近似换算，不是严格的流行病学设计。
     */
    public static double riskRatioFromAccuracy(double baselineAccuracy, double modelAccuracy) {
        double b = clamp01(baselineAccuracy), m = clamp01(modelAccuracy);
        if (b <= 1e-6) return Double.NaN;
        return m / b;
    }

    // ------------------------------------------- 论文数值的预设审计案例

    /** 一个可审计的论文数值案例。 */
    public static final class Case {
        public final String name;
        public final String source;
        public final double reportedAccuracy;
        public final int sampleSize;
        public final double positiveRate;
        public final double prevalence;
        public final String note;

        Case(String name, String source, double reportedAccuracy, int sampleSize,
             double positiveRate, double prevalence, String note) {
            this.name = name;
            this.source = source;
            this.reportedAccuracy = reportedAccuracy;
            this.sampleSize = sampleSize;
            this.positiveRate = positiveRate;
            this.prevalence = prevalence;
            this.note = note;
        }
    }

    /**
     * 课程中实际讨论的三个方向的预设案例。
     *
     * <p>数值为教学假设，不是对所列论文实验表格的摘录；
     * <b>不代表本应用复现了这些论文</b>，也不对任何人群做推断。
     */
    public static List<Case> presetCases() {
        List<Case> cases = new ArrayList<>();
        cases.add(new Case(
                "唐氏综合征人脸识别（FG 2024 方向）",
                "Rathgeb 等，Testing the Performance of Face Recognition for People with Down Syndrome",
                0.90, 200, 0.5, 0.001,
                "识别系统的准确率若在小样本上报告，置信区间会很宽；而目标人群先验比例很低，"
                        + "阳性预测值会远低于准确率。"));
        cases.add(new Case(
                "性取向面部推断（Kosinski 2017 方向）",
                "Wang & Kosinski, JPSP 2017；Leuner 2019 复现",
                0.81, 7000, 0.5, 0.05,
                "研究提示亮度、颜色及呈现方式也可能携带信号；形态与修饰、呈现及生活方式各自贡献仍未确定。"
                        + "不能将本教学数值当作该论文的实测还原。"));
        cases.add(new Case(
                "通用二分类基准（对照组）",
                "课程内的参照案例",
                0.85, 1000, 0.5, 0.5,
                "平衡样本上的准确率最容易给人“模型很准”的错觉；一旦先验比例失衡，"
                        + "同一模型的阳性预测值会大幅下降。"));
        return cases;
    }

    /** 对单个案例执行完整审计，返回可展示的文本。 */
    public static String auditCase(Case item) {
        Interval interval = wilson((int) Math.round(item.reportedAccuracy * item.sampleSize),
                item.sampleSize, 0.95);
        // 案例没有单独给出敏感度，用准确率作为保守近似，并明确标注。
        PredictiveValues predictive = baseRate(item.prevalence, item.reportedAccuracy, item.reportedAccuracy);
        ImbalanceAudit imbalance = imbalance(item.positiveRate, item.reportedAccuracy, item.reportedAccuracy);
        StringBuilder builder = new StringBuilder();
        builder.append("【").append(item.name).append("】\n");
        builder.append("来源：").append(item.source).append("\n");
        builder.append("教学假设准确率 ").append(String.format(Locale.CHINA, "%.1f%%", item.reportedAccuracy * 100))
                .append("，假设样本 ").append(item.sampleSize).append(" 张；不是上述论文的实测还原。\n");
        builder.append("计算假设：敏感度和特异度均取上述教学数值，这是假设而非由准确率推导。\n");
        builder.append("· ").append(interval.summary()).append("\n");
        builder.append("  说明：即使准确率真实，这个样本量也只能把区间收窄到上述宽度；")
                .append("报告值应理解为区间而非单点。\n");
        builder.append("· 基率效应：目标人群先验比例 ")
                .append(String.format(Locale.CHINA, "%.3f%%", item.prevalence * 100)).append("。\n");
        builder.append("  ").append(predictive.summary()).append("\n");
        builder.append("· 类别不平衡：").append(imbalance.summary()).append("\n");
        builder.append("· E-value：本分类案例未提供适用的关联风险比，不计算。准确率之比不能替代关联风险比。\n");
        builder.append("· 课程结论：").append(item.note).append("\n");
        builder.append("· 边界：以上是明确假设下的统计教学，不是原论文实验复现，不构成本应用对任何人群的推断，")
                .append("也不能用于诊断或评价个人。\n");
        return builder.toString();
    }

    /** 将审计结果序列化为 JSON。 */
    public static org.json.JSONObject toJson(Case item) {
        try {
            Interval interval = wilson((int) Math.round(item.reportedAccuracy * item.sampleSize),
                    item.sampleSize, 0.95);
            PredictiveValues predictive = baseRate(item.prevalence, item.reportedAccuracy, item.reportedAccuracy);
            ImbalanceAudit imbalance = imbalance(item.positiveRate, item.reportedAccuracy, item.reportedAccuracy);
            org.json.JSONObject json = new org.json.JSONObject();
            json.put("case", item.name);
            json.put("source", item.source);
            json.put("parameterOrigin", "synthetic_teaching_assumptions");
            json.put("assumedAccuracy", item.reportedAccuracy);
            json.put("sampleSize", item.sampleSize);
            json.put("ciLower", interval.lower);
            json.put("ciUpper", interval.upper);
            json.put("ciWidth", interval.width);
            json.put("ppv", predictive.ppv);
            json.put("falsePerTrue", predictive.falsePerTrue);
            json.put("majorityBaseline", imbalance.majorityBaseline);
            json.put("liftOverBaseline", imbalance.liftOverBaseline);
            json.put("evalue",org.json.JSONObject.NULL);json.put("evalueApplicable",false);
            json.put("scope", "Synthetic statistical teaching example, not reported paper results. No personal or group inference.");
            return json;
        } catch (Exception error) {
            return null;
        }
    }

    private static double clamp01(double value) {
        if (!Double.isFinite(value)) return Double.NaN;
        return Math.max(0, Math.min(1, value));
    }

    private MethodologyAudit() {}
}
