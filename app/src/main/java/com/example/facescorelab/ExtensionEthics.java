package com.example.facescorelab;

import java.util.ArrayList;
import java.util.List;

/** 扩展研究方向、数据使用要求及模型边界说明。 */
public final class ExtensionEthics {

    /** 一个扩展方向的说明。 */
    public static final class Direction {
        public final String title;
        public final String status;
        public final String technique;
        public final String boundary;
        /** 应用内是否提供任何推断输出。 */
        public final boolean inferenceProvided;

        Direction(String title, String status, String technique, String boundary, boolean inferenceProvided) {
            this.title = title;
            this.status = status;
            this.technique = technique;
            this.boundary = boundary;
            this.inferenceProvided = inferenceProvided;
        }
    }

    private ExtensionEthics() {}

    /** 扩展研究方向定义。 */
    public static List<Direction> directions() {
        List<Direction> list = new ArrayList<>();
        list.add(new Direction(
                "唐氏综合征面部特征识别",
                "方法审计与二维可见测量已实现",
                "公开研究使用面部关键点、眼距、鼻梁与面部扁平度等特征做分类。"
                        + "本应用提供方法论审计（基率、置信区间、类别不平衡、E-value），"
                        + "并展示明确标为本项目二维定义的眼距、鼻部投影比例；统计案例参数明确标为教学假设。",
                "仅用于技术演示，禁止作为医学诊断。本应用不包含唐氏综合征识别模型，"
                        + "照片卡只报告原始可见测量，不给出疾病指数、阈值或置信度。",
                false));
        list.add(new Direction(
                "阿尔茨海默相关面部/表情分析",
                "已实现单帧开合测量与可见动作统计",
                "表情识别、眼动或面部微表情统计。本应用统计 MediaPipe 的可见动作系数分布，"
                        + "不训练疾病分类器，也不采集眼动。",
                "不得宣称可诊断疾病。可见动作统计只是单张照片的模型输出分布，"
                        + "与认知状态没有经过验证的对应关系，应用不设任何临床阈值。",
                true));
        list.add(new Direction(
                "性取向识别复现",
                "图像质量对照与普通评分鲁棒性已实现",
                "Leuner复现研究说明形态与呈现等因素的贡献仍不明确。"
                        + "本应用展示原图与真实均值模糊的亮度/清晰度统计，并独立测量普通颜值CNN的图像扰动漂移；不声称复现论文分类器。",
                "必须讨论隐私、偏见和歧视风险。本应用不根据照片推断性取向或任何敏感个人属性，"
                        + "不提供属性分类入口或输出。",
                false));
        return list;
    }

    /** 数据使用与展示要求。 */
    public static String[] requirements() {
        return new String[]{
                "不得宣称 App 能诊断唐氏综合征、阿尔茨海默病或其他疾病。",
                "不得将性取向识别功能包装成“检测工具”。",
                "不得对真实用户进行公开打分、标签化或歧视性展示。",
                "人脸数据默认本地处理，不上传服务器。",
                "报告中必须说明：颜值分数只是人群主观打分的统计结果，不代表客观美丑或人的价值。"
        };
    }

    /** 各项要求对应的实现说明。 */
    public static String[] compliance() {
        return new String[]{
                "本应用最终 Manifest 不含互联网权限，照片与结果只保存在设备本地。",
                "所有“属性/动作”输出都标注为模型输出，不包装为医学检测或概率。",
                "不提供疾病识别、性取向识别或敏感个人属性的推断入口。",
                "参照组仍由用户确认；可选CelebA二元标签参考不自动决定参照组，也不作为本人身份判断。",
                "界面与报告中均声明分数来自数据集主观标注统计，不代表客观美丑或人的价值。"
        };
    }

    /** 三个方向的总体定位说明。 */
    public static String overview() {
        return "任务书给出三个可选扩展方向。本应用的做法是把它们区分为“可实现的可见测量”与"
                + "“只做文献与伦理讨论的敏感推断”：只实现前者，并在界面中明确说明后者的边界。"
                + "这样既覆盖了课程要求的技术探索，也遵守了任务书的伦理约束。";
    }
}
