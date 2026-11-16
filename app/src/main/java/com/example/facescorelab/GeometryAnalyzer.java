package com.example.facescorelab;

import android.content.Context;
import android.graphics.PointF;
import java.util.Arrays;
import java.util.Locale;

/**
 * Measures visible 2D geometry. The dlib-style 68-point training topology and
 * MediaPipe's 478-point runtime topology are linked by semantic correspondences.
 * This mapping is an approximation and is disclosed in the experiment summary.
 */
public final class GeometryAnalyzer {
    public static final String[] FEATURE_NAMES = {
            "face_aspect", "inner_eye_ratio", "eye_width_ratio", "eye_open_ratio",
            "brow_eye_ratio", "nose_width_ratio", "nose_length_ratio", "mouth_width_ratio",
            "lip_fullness_ratio", "mouth_open_ratio", "lower_face_ratio", "symmetry_error",
            "jaw_width_ratio", "jaw_angle", "nose_mouth_ratio"
    };

    // Corresponding anatomical locations, not a claim that MediaPipe implements dlib68.
    public static final int[] MEDIAPIPE_TO_68 = {
            234,93,132,58,172,136,150,149,152,378,379,365,397,288,361,323,454,
            70,63,105,66,107,336,296,334,293,300,
            168,6,197,1,98,97,2,326,327,
            33,160,158,133,153,144,362,385,387,263,373,380,
            61,40,37,0,267,270,291,321,314,17,84,91,
            78,81,13,311,308,402,14,178
    };

    private static final int[][] SYMMETRY_PAIRS = {
            {0,16},{1,15},{2,14},{3,13},{4,12},{5,11},{6,10},{7,9},
            {17,26},{18,25},{19,24},{20,23},{21,22},{31,35},{32,34},
            {36,45},{37,44},{38,43},{39,42},{40,47},{41,46},
            {48,54},{49,53},{50,52},{55,59},{56,58},{60,64},{61,63},{65,67}
    };
    private final ScoringModel model;

    public GeometryAnalyzer(Context context) {
        model = new ScoringModel(context);
    }

    public AnalysisResult analyze(PointF[] points478, int imageWidth, int imageHeight,
                                  String referenceMode, Float hairlineNormalized) {
        AnalysisResult result = new AnalysisResult();
        result.algorithmVersion = model.version;
        result.referenceMode = "male".equals(referenceMode) || "female".equals(referenceMode) ? referenceMode : "general";
        if (points478 == null || points478.length < 478 || imageWidth <= 0 || imageHeight <= 0) {
            result.qualityMessage = "未取得完整人脸关键点，请使用清晰的单人正面照片。";
            return result;
        }
        double[][] points68 = new double[68][2];
        for (int i=0; i<68; i++) {
            PointF point = points478[MEDIAPIPE_TO_68[i]];
            if (point == null || !Float.isFinite(point.x) || !Float.isFinite(point.y)) {
                result.qualityMessage = "检测到无效关键点，请重新拍摄。";
                return result;
            }
            points68[i][0] = point.x;
            points68[i][1] = point.y;
        }
        final Normalized normalized;
        final double[] values;
        try {
            normalized = normalize(points68);
            values = featureVector68(points68);
        } catch (IllegalArgumentException error) {
            result.qualityMessage = "关键点不足以稳定计算比例，请调整正面姿态。";
            return result;
        }
        result.symmetry = Math.max(0, Math.min(100, 100*(1-values[11])));
        result.eyeAspect = values[3];
        result.mouthAspect = values[9];
        double eyeCenterDistance = distance(mean(normalized.points,36,42),mean(normalized.points,42,48));
        double noseOffset = normalized.points[30][0] / Math.max(eyeCenterDistance,1e-8);
        result.yaw = Math.toDegrees(Math.atan(2*noseOffset));
        addMetrics(result, values, normalized, points478, imageHeight, hairlineNormalized);
        if (normalized.width/imageWidth < 0.12 || normalized.width < 48) {
            result.qualityMessage = "人脸在画面中太小，请靠近相机或选择更清晰的照片。";
            return result;
        }
        if (Math.abs(result.yaw) > 45) {
            result.qualityMessage = "左右偏转较大，当前仅显示测量结果；请转向正面后评分。";
            return result;
        }
        if (values[0] < 0.35 || values[0] > 2.0 || values[2] < 0.04 || values[2] > 0.45) {
            result.qualityMessage = "面部比例超出可靠测量范围，请检查遮挡或拍摄角度。";
            return result;
        }
        ScoringModel.Prediction prediction = model.predict(values, result.referenceMode);
        result.score = prediction.score;
        result.baseline = prediction.referenceMean;
        result.referenceSampleCount = prediction.referenceCount;
        result.scorable = true;
        result.qualityMessage = prediction.clippedFeatures >= 4
                ? "部分比例超出训练样本的常见范围，结果仅供课程模型参考。"
                : "已完成本地几何分析；保持正面、均匀光照有助于稳定测量。";
        String group = "male".equals(result.referenceMode) ? "男性参考" : "female".equals(result.referenceMode) ? "女性参考" : "通用参考";
        result.reasons.add(String.format(Locale.CHINA, "%s来自训练集%d张标注图片，平均评分%.2f/10。",group,prediction.referenceCount,prediction.referenceMean));
        // Show the three largest additive terms of the actual fitted regression.
        Integer[] order = new Integer[values.length];
        for(int i=0; i<order.length; i++) order[i]=i;
        Arrays.sort(order,(a,b)->Double.compare(Math.abs(prediction.contributions[b]),Math.abs(prediction.contributions[a])));
        for(int rank=0; rank<3; rank++) {
            int index = order[rank];
            double referenceZ = (values[index]-prediction.referenceFeatureMean[index])/prediction.referenceFeatureStd[index];
            String relative = Math.abs(referenceZ)<0.35 ? "接近" : referenceZ>0 ? "高于" : "低于";
            result.reasons.add(String.format(Locale.CHINA,
                    "%s%s该参考组均值；在当前模型中的加性项约%+.2f分。",
                    model.labels[index],relative,prediction.contributions[index]));
        }
        result.reasons.add("参考模式只选择已有样本统计，不从照片推断性别；各组使用同一通用回归模型。");
        result.reasons.add("该分数学的是数据集中评分者的主观意见，不代表客观美丑或人的价值。");
        result.reasons.add("训练使用原始68点标注，手机采用478点语义映射，存在跨检测器偏差；报告中的测试误差不是手机端准确率。");
        // 使用当前预测的参照组统计生成评分依据。
        result.scoringBasis = ScoringBasis.toJson(values, prediction, result.referenceMode,
                values[3], values[9], null, null, 0, 0, null);
        return result;
    }

    private static void addMetrics(AnalysisResult r, double[] f, Normalized n, PointF[] points478,
                                   int imageHeight, Float hairlineNormalized) {
        metric(r,"左右几何相似度",r.symmetry,"%","以眼睛轴校正倾斜后，比较左右对应点；仅描述图像几何。");
        metric(r,"面部高宽比",f[0],"倍","眉部均值至下巴的投影高度/面部宽度，不包含真实发际线。");
        metric(r,"五眼估计",1/f[2],"眼宽","可见面部宽度/双眼平均宽度，是图像测量的近似值。");
        metric(r,"眼间距/脸宽",f[1],"倍","双眼内眼角距离除以面部宽度。");
        metric(r,"眉眼距/脸宽",f[4],"倍","同侧眉毛和眼睛中心距离的平均值除以脸宽。");
        metric(r,"鼻部投影长度",f[6],"倍","鼻根至鼻底的二维长度/眉部至下巴高度，不测量三维鼻梁高度。");
        metric(r,"鼻宽/嘴宽",f[14],"倍","鼻翼间距除以嘴角间距。");
        metric(r,"外唇高度/嘴宽",f[8],"倍","上、下外唇中点距离除以嘴宽，受表情和拍摄角度影响。");
        metric(r,"眼睛开合 EAR",f[3],"","双眼标准眼睛纵横比的平均值，仅作可见动作统计。");
        metric(r,"嘴部开合 MAR",f[9],"","内唇开合距离除以嘴宽，仅作可见动作统计。");
        double hairY;
        if(hairlineNormalized!=null && Float.isFinite(hairlineNormalized)
                && hairlineNormalized>=0 && hairlineNormalized<=1) {
            hairY = n.projectY(n.centerX,hairlineNormalized*imageHeight);
            r.hairlineCalibrated = hairY<n.browY;
        } else {
            PointF forehead = points478[10];
            hairY = forehead==null ? n.browY-n.height*0.5 : n.projectY(forehead.x,forehead.y);
        }
        if(hairY>=n.browY) {
            hairY=n.browY-n.height*0.5;
            r.hairlineCalibrated=false;
        }
        double total=n.points[8][1]-hairY;
        double top=n.browY-hairY, middle=n.points[33][1]-n.browY, bottom=n.points[8][1]-n.points[33][1];
        String suffix=r.hairlineCalibrated ? "（已校准）" : "（估算）";
        String description=r.hairlineCalibrated
                ? "发际线位置由用户手动校准；按图像中的投影长度计算，未用于评分模型。"
                : "网格额顶部不是实际发际线；请手动校准后查看三庭比例，当前估算未用于评分。";
        if(total>1e-8 && top>=0 && middle>=0 && bottom>=0) {
            metric(r,"上庭占比"+suffix,100*top/total,"%",description);
            metric(r,"中庭占比"+suffix,100*middle/total,"%",description);
            metric(r,"下庭占比"+suffix,100*bottom/total,"%",description);
        }
    }

    private static void metric(AnalysisResult r,String name,double value,String unit,String description) {
        r.metrics.add(new AnalysisResult.Metric(name,value,unit,description));
    }

    /** Shared feature definitions used by the trainer and parity-checked against its output. */
    public static double[] featureVector68(double[][] points) {
        Normalized n=normalize(points);
        double[][] q=n.points;
        double eyeWidth=(distance(q[36],q[39])+distance(q[42],q[45]))/2;
        double mouthWidth=distance(q[48],q[54]);
        if(eyeWidth<=1e-7 || mouthWidth<=1e-7) throw new IllegalArgumentException("无效五官间距");
        double ear=((distance(q[37],q[41])+distance(q[38],q[40]))/(2*distance(q[36],q[39]))
                +(distance(q[43],q[47])+distance(q[44],q[46]))/(2*distance(q[42],q[45])))/2;
        double browDistance=(distance(mean(q,17,22),mean(q,36,42))+distance(mean(q,22,27),mean(q,42,48)))/2;
        double symmetry=0;
        for(int[] pair:SYMMETRY_PAIRS) symmetry+=Math.hypot(q[pair[0]][0]+q[pair[1]][0],q[pair[0]][1]-q[pair[1]][1]);
        symmetry/=SYMMETRY_PAIRS.length;
        double ax=q[4][0]-q[8][0], ay=q[4][1]-q[8][1];
        double bx=q[12][0]-q[8][0], by=q[12][1]-q[8][1];
        double cosine=(ax*bx+ay*by)/(Math.hypot(ax,ay)*Math.hypot(bx,by));
        double angle=Math.acos(Math.max(-1,Math.min(1,cosine)))/Math.PI;
        double[] out={n.height,distance(q[39],q[42]),eyeWidth,ear,browDistance,
                distance(q[31],q[35]),distance(q[27],q[33])/n.height,mouthWidth,
                distance(q[51],q[57])/mouthWidth,distance(q[62],q[66])/mouthWidth,
                (q[8][1]-q[33][1])/n.height,symmetry,distance(q[4],q[12]),angle,
                distance(q[31],q[35])/mouthWidth};
        for(double value:out) if(!Double.isFinite(value)) throw new IllegalArgumentException("无效几何值");
        return out;
    }

    private static Normalized normalize(double[][] points) {
        if(points==null || points.length!=68) throw new IllegalArgumentException("需要68个对应点");
        for(double[] p:points) if(p==null || p.length!=2 || !Double.isFinite(p[0]) || !Double.isFinite(p[1])) throw new IllegalArgumentException("无效坐标");
        double[] le=mean(points,36,42), re=mean(points,42,48);
        double dx=re[0]-le[0], dy=re[1]-le[1], eyeDistance=Math.hypot(dx,dy);
        if(eyeDistance<=1e-7) throw new IllegalArgumentException("双眼重合");
        Normalized n=new Normalized();
        n.ux=dx/eyeDistance; n.uy=dy/eyeDistance;
        n.vx=-n.uy; n.vy=n.ux;
        n.centerX=(le[0]+re[0])/2; n.centerY=(le[1]+re[1])/2;
        if((points[8][0]-n.centerX)*n.vx+(points[8][1]-n.centerY)*n.vy<0) { n.vx=-n.vx; n.vy=-n.vy; }
        n.points=new double[68][2];
        for(int i=0;i<68;i++) {
            n.points[i][0]=(points[i][0]-n.centerX)*n.ux+(points[i][1]-n.centerY)*n.uy;
            n.points[i][1]=(points[i][0]-n.centerX)*n.vx+(points[i][1]-n.centerY)*n.vy;
        }
        n.width=Math.abs(n.points[16][0]-n.points[0][0]);
        double rawBrowY=mean(n.points,17,27)[1];
        double rawHeight=n.points[8][1]-rawBrowY;
        if(n.width<=1 || rawHeight<=1) throw new IllegalArgumentException("脸部范围无效");
        for(double[] point:n.points) {point[0]/=n.width;point[1]/=n.width;}
        n.height=rawHeight/n.width;
        n.browY=rawBrowY/n.width;
        return n;
    }

    private static double[] mean(double[][] points,int start,int end) {
        double[] out={0,0};
        for(int i=start;i<end;i++){out[0]+=points[i][0];out[1]+=points[i][1];}
        out[0]/=(end-start);out[1]/=(end-start);
        return out;
    }

    private static double distance(double[] a,double[] b){return Math.hypot(a[0]-b[0],a[1]-b[1]);}

    private static final class Normalized {
        double[][] points;
        double width,height,browY,centerX,centerY,ux,uy,vx,vy;
        double projectY(double x,double y){return ((x-centerX)*vx+(y-centerY)*vy)/width;}
    }
}
