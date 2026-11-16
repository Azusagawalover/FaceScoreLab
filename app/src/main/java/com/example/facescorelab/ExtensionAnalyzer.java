package com.example.facescorelab;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PointF;
import android.graphics.Rect;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

/** Paper-linked visible observations. Definitions here are implementation choices, not clinical norms. */
public final class ExtensionAnalyzer {
    public static final String VERSION="paper-visible-measurements-v1";
    public static final class Measurement {
        public final String key,name,unit,definition;
        public final double value;
        public final boolean available;
        Measurement(String key,String name,double value,String unit,String definition){
            this.key=key;this.name=name;this.value=value;this.unit=unit;this.definition=definition;
            available=Double.isFinite(value);
        }
        JSONObject toJson()throws Exception{
            return new JSONObject().put("key",key).put("name",name).put("value",available?value:JSONObject.NULL)
                    .put("unit",unit).put("available",available).put("definition",definition);
        }
    }
    public static final class ExtensionReport {
        public final String directionKey,title,source,limitation,status;
        public final boolean available;
        public final List<Measurement> measurements=new ArrayList<>();
        ExtensionReport(String key,String title,String source,String limitation,String status){
            directionKey=key;this.title=title;this.source=source;this.limitation=limitation;this.status=status;
            available="available".equals(status);
        }
        void add(String key,String name,double value,String unit,String definition){
            measurements.add(new Measurement(key,name,value,unit,definition));
        }
        public double getValue(String key){
            for(Measurement row:measurements)if(row.key.equals(key))return row.value;
            return Double.NaN;
        }
        public JSONObject toJson()throws Exception{
            JSONArray rows=new JSONArray();for(Measurement row:measurements)rows.put(row.toJson());
            return new JSONObject().put("schemaVersion",1).put("methodVersion",VERSION).put("direction",directionKey)
                    .put("title",title).put("available",available).put("status",status).put("measurements",rows)
                    .put("source",source).put("limitation",limitation).put("diagnosticUse",false)
                    .put("personalAttributeInference",false);
        }
        public String formatDisplayText(){
            StringBuilder text=new StringBuilder(title).append('\n');
            if(!available)text.append("本次测量不可用：").append(status).append('\n');
            for(Measurement row:measurements){
                text.append("· ").append(row.name).append("：");
                if(row.available)text.append(String.format(Locale.CHINA,"%.4f",row.value)).append(row.unit);
                else text.append("不可用（关键点或分母不足）");
                text.append("\n  ").append(row.definition).append('\n');
            }
            return text.append("\n文献关系：").append(source).append("\n测量限制：").append(limitation).toString();
        }
    }
    private ExtensionAnalyzer(){}

    public static ExtensionReport measureGeometry(PointF[] points,Bitmap photo){
        ExtensionReport report=new ExtensionReport("geometry_observation","二维几何观察",
                "任务书参考 Rathgeb et al., arXiv:2405.11240，PDF p.2 §III–IV；Rondina et al., arXiv:2502.06341，PDF p.4 §3.2。研究关注系统表现与数据/拍摄条件；下列比例为本项目明确写出的二维实现，不是论文诊断阈值。",
                "眼角中点只是眼中心近似，不是真实瞳孔或虹膜定位；二维投影不能测三维鼻梁高度/面部扁平度，不换算疾病指数。",
                valid(points)?"available":"invalid_landmarks");
        if(!report.available)return report;
        PointF left=mid(points[33],points[133]),right=mid(points[362],points[263]);
        report.add("intercanthal_eye_center_ratio","内眼角间距 / 双眼角中心距",ratio(dist(points[133],points[362]),dist(left,right))," 倍",
                "133–362距离 ÷ 两侧眼角中点距离；中心点分别为(33,133)及(362,263)中点。未使用瞳距常模。 ");
        report.add("nasal_projection_aspect","鼻部二维投影长宽比",ratio(dist(points[168],points[1]),dist(points[98],points[327]))," 倍",
                "168–1投影距离 ÷ 98–327鼻翼距离，受姿态和关键点误差影响。 ");
        report.add("face_width_upper_height","上面部二维宽高比（本项目定义）",ratio(dist(points[234],points[454]),dist(points[9],points[0]))," 倍",
                "234–454面宽 ÷ 9–0投影高度；不等同于所有论文的fWHR定义，不据此判断个人属性。 ");
        return report;
    }

    public static ExtensionReport measureVisibleActions(PointF[] points,double geometricSimilarity){
        ExtensionReport report=new ExtensionReport("visible_action_observation","眼睑与嘴部可见测量",
                "FER2013原始数据集报告 arXiv:1307.0414，PDF p.3 §3描述数据与学习任务，并不提供由EAR/MAR推导情绪的公式。本卡直接测量网格几何；不是FER2013分类模型。",
                "单张照片只反映可见开合与几何相似度，不是眼动、视频微表情、情绪类别或认知状态；已有MediaPipe动作卡独立展示模型系数。",
                valid(points)?"available":"invalid_landmarks");
        if(!report.available)return report;
        double left=ear(points,new int[]{33,160,158,133,153,144});
        double right=ear(points,new int[]{362,385,387,263,373,380});
        String earDefinition="六点EAR=(两条对应上下眼睑距离之和)/(2×眼角宽度)，两组分别33/160/158/133/153/144及362/385/387/263/373/380。";
        report.add("ear_left","第一眼组 EAR",left,"",earDefinition);
        report.add("ear_right","第二眼组 EAR",right,"",earDefinition);
        report.add("ear_mean","双眼平均 EAR",(left+right)/2,"","两组EAR的算术平均；没有临床/情绪判定阈值。 ");
        report.add("ear_difference","左右眼组开合差",Math.abs(left-right),"","两组EAR绝对差；不是神经、肌肉或微表情诊断。 ");
        double width=dist(points[61],points[291]);
        report.add("mar","内唇开合 / 嘴宽",ratio(dist(points[13],points[14]),width)," 倍","13–14内唇间距 ÷ 61–291嘴角间距。 ");
        report.add("corner_lift","嘴角相对唇中心纵向位移",ratio((points[13].y+points[14].y-points[61].y-points[291].y)/2,width),"",
                "图像y向下；(唇中心y−嘴角平均y)/嘴宽，随头部倾斜变化，不转换为情绪概率。 ");
        report.add("geometric_similarity","左右二维关键点相似度",Double.isFinite(geometricSimilarity)&&geometricSimilarity>=0&&geometricSimilarity<=100?geometricSimilarity:Double.NaN,"%",
                "本项目经验映射：100×(1−左右对称误差)，截断到0–100；不是论文指标或百分比准确率。 ");
        return report;
    }

    public static ExtensionReport measureCaptureQuality(PointF[] points,Bitmap photo,boolean blurControl){
        boolean input=valid(points)&&photo!=null&&!photo.isRecycled();
        Rect bounds=input?bounds(points,photo.getWidth(),photo.getHeight()):null;
        ExtensionReport report=new ExtensionReport("image_condition_observation","拍摄条件与真实像素对照",
                "Leuner, arXiv:1902.10739，PDF p.3摘要讨论形态、呈现与数据局限；不提供个人性取向几何判定式。此处只观察图像条件，普通颜值CNN的9次扰动在上一张卡独立运行。",
                "图像统计不是任何属性模型的置信度；本对照只处理像素，不重新检测网格、不验证敏感属性分类器。分析图最长边限制512像素。",
                !input?"image_or_landmarks_unavailable":bounds==null?"invalid_face_region":"available");
        if(!report.available)return report;
        Bitmap crop=Bitmap.createBitmap(photo,bounds.left,bounds.top,bounds.width(),bounds.height());
        Bitmap analysis=crop,blurred=null;
        try{
            int longest=Math.max(crop.getWidth(),crop.getHeight());
            if(longest>512){float scale=512f/longest;analysis=Bitmap.createScaledBitmap(crop,Math.max(1,Math.round(crop.getWidth()*scale)),Math.max(1,Math.round(crop.getHeight()*scale)),true);}
            double[] normal=quality(analysis);
            report.add("analysis_width","分析图宽度",analysis.getWidth()," px","人脸网格包围区域，等比例缩至最长边不超过512。 ");
            report.add("analysis_height","分析图高度",analysis.getHeight()," px","全部对照使用相同尺寸和裁剪。 ");
            report.add("brightness_mean","平均编码亮度",normal[0],"","RGB编码值加权(0.299R+0.587G+0.114B)/255；不是三维形态或肤色类别。 ");
            report.add("brightness_std","编码亮度标准差",normal[1],"","该区域亮度的总体标准差；同时受光照、纹理与遮挡影响。 ");
            report.add("laplacian_energy","拉普拉斯平均平方能量",normal[2],"","归一化亮度四邻域拉普拉斯的均方；是清晰度代理量，不是概率。 ");
            if(blurControl){
                blurred=meanBlur(analysis);double[] control=quality(blurred);
                report.add("brightness_mean_blur","均值模糊后亮度均值",control[0],"","实际生成3×3等权均值模糊位图后重新统计，边界仅使用有效邻居。 ");
                report.add("brightness_std_blur","均值模糊后亮度标准差",control[1],"","来自真实处理后的像素，没有预先写死惩罚分。 ");
                report.add("laplacian_energy_blur","均值模糊后拉普拉斯能量",control[2],"","原图和对照采用同一能量定义。 ");
                report.add("laplacian_delta","能量变化（对照−原图）",control[2]-normal[2],"","展示实际差值，不据此推断个人或群体属性。 ");
            }
            return report;
        }finally{
            if(blurred!=null&&!blurred.isRecycled())blurred.recycle();
            if(analysis!=crop&&!analysis.isRecycled())analysis.recycle();
            if(crop!=photo&&!crop.isRecycled())crop.recycle();
        }
    }
    private static boolean valid(PointF[] points){
        if(points==null||points.length<478)return false;
        for(PointF point:points)if(point==null||!Float.isFinite(point.x)||!Float.isFinite(point.y))return false;
        return true;
    }
    private static Rect bounds(PointF[] points,int width,int height){
        float l=Float.MAX_VALUE,t=l,r=-l,b=-l;
        for(PointF p:points){l=Math.min(l,p.x);t=Math.min(t,p.y);r=Math.max(r,p.x);b=Math.max(b,p.y);}
        int x1=Math.max(0,(int)Math.floor(l)),y1=Math.max(0,(int)Math.floor(t));
        int x2=Math.min(width,(int)Math.ceil(r)),y2=Math.min(height,(int)Math.ceil(b));
        return x2-x1>=3&&y2-y1>=3?new Rect(x1,y1,x2,y2):null;
    }
    private static double ear(PointF[] p,int[] i){return ratio(dist(p[i[1]],p[i[5]])+dist(p[i[2]],p[i[4]]),2*dist(p[i[0]],p[i[3]]));}
    private static double ratio(double a,double b){return Double.isFinite(a)&&Double.isFinite(b)&&b>1e-4?a/b:Double.NaN;}
    private static double dist(PointF a,PointF b){return Math.hypot(a.x-b.x,a.y-b.y);}
    private static PointF mid(PointF a,PointF b){return new PointF((a.x+b.x)/2,(a.y+b.y)/2);}
    private static double luma(int color){return (.299*Color.red(color)+.587*Color.green(color)+.114*Color.blue(color))/255.0;}
    private static double[] quality(Bitmap bitmap){
        int w=bitmap.getWidth(),h=bitmap.getHeight();int[] pixels=new int[w*h];bitmap.getPixels(pixels,0,w,0,0,w,h);
        double sum=0,squared=0,lapSquared=0;int count=0;
        for(int color:pixels){double v=luma(color);sum+=v;squared+=v*v;}
        double mean=sum/pixels.length;
        for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
            int i=y*w+x;double lap=4*luma(pixels[i])-luma(pixels[i-1])-luma(pixels[i+1])-luma(pixels[i-w])-luma(pixels[i+w]);
            lapSquared+=lap*lap;count++;
        }
        return new double[]{mean,Math.sqrt(Math.max(0,squared/pixels.length-mean*mean)),count==0?Double.NaN:lapSquared/count};
    }
    private static Bitmap meanBlur(Bitmap bitmap){
        int w=bitmap.getWidth(),h=bitmap.getHeight();int[] pixels=new int[w*h],out=new int[w*h];bitmap.getPixels(pixels,0,w,0,0,w,h);
        for(int y=0;y<h;y++)for(int x=0;x<w;x++){
            int r=0,g=0,b=0,n=0;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                int xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=w||yy>=h)continue;
                int c=pixels[yy*w+xx];r+=Color.red(c);g+=Color.green(c);b+=Color.blue(c);n++;
            }
            out[y*w+x]=Color.rgb(r/n,g/n,b/n);
        }
        return Bitmap.createBitmap(out,w,h,Bitmap.Config.ARGB_8888);
    }
}
