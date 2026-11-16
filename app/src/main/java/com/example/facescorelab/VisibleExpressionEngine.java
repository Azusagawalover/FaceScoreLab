package com.example.facescorelab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import java.io.File;
import java.io.FileOutputStream;
import java.util.List;
import org.json.JSONObject;

/** Observable mouth movements from the bundled blendshape model, never emotions. */
public final class VisibleExpressionEngine {
    public static final String VERSION="mediapipe-visible-mouth-actions-v1";
    public static final double SMILE_THRESHOLD=.5;
    public static final double MOUTH_OPEN_THRESHOLD=.25;
    private final Context context;

    public static final class Result {
        public boolean available;
        public int faceCount,selectedFace=-1;
        public double smileLeft,smileRight,smile,jawOpen;
        public String unavailableReason="";
        /** 52 项可见动作系数的汇总统计；见 MicroExpressionStats。 */
        public JSONObject microExpression;

        public static Result unavailable(String reason){
            Result result=new Result();result.unavailableReason=reason;return result;
        }
        public JSONObject toJson()throws Exception{
            JSONObject json=new JSONObject();json.put("model",VERSION);json.put("available",available);
            json.put("faceCount",faceCount);json.put("selectedFaceIndex",selectedFace);
            json.put("valueType","visible_action_coefficient_not_probability");
            json.put("smileThreshold",SMILE_THRESHOLD);json.put("mouthOpenThreshold",MOUTH_OPEN_THRESHOLD);
            if(available){
                json.put("mouthSmileLeft",smileLeft);json.put("mouthSmileRight",smileRight);
                json.put("smileCoefficient",smile);json.put("smileDetected",smile>=SMILE_THRESHOLD);
                json.put("jawOpenCoefficient",jawOpen);json.put("mouthOpenDetected",jawOpen>=MOUTH_OPEN_THRESHOLD);
            }else json.put("reason",unavailableReason);
            if(microExpression!=null)json.put("microExpression",microExpression);
            json.put("method","Mean of mouthSmileLeft and mouthSmileRight; jawOpen is separate. Fixed demonstration thresholds, not calibrated probabilities or emotion labels.");
            return json;
        }
    }

    public VisibleExpressionEngine(Context context){this.context=context.getApplicationContext();}

    public Result analyze(Bitmap photo)throws Exception{
        FaceLandmarker.FaceLandmarkerOptions options=FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.IMAGE).setNumFaces(3)
                .setMinFaceDetectionConfidence(.5f).setMinFacePresenceConfidence(.5f)
                .setMinTrackingConfidence(.5f).setOutputFaceBlendshapes(true).build();
        // MPImage owns its bitmap. Preserve the caller's displayed/stored photo.
        Bitmap owned=photo.copy(Bitmap.Config.ARGB_8888,false);
        if(owned==null)throw new IllegalStateException("无法分配动作分析图像");
        MPImage input=new BitmapImageBuilder(owned).build();
        try(FaceLandmarker detector=FaceLandmarker.createFromOptions(context,options)){
            if((context.getApplicationInfo().flags&android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE)!=0){
                try(FileOutputStream output=new FileOutputStream(new File(context.getCacheDir(),"deep_expression_input.png"))){
                    owned.compress(Bitmap.CompressFormat.PNG,100,output);
                }
            }
            FaceLandmarkerResult detected=detector.detect(input);
            if(detected.faceLandmarks().isEmpty())return Result.unavailable("face_not_detected");
            if(!detected.faceBlendshapes().isPresent())return Result.unavailable("blendshapes_not_returned");
            int selected=-1;double largest=-1;
            for(int i=0;i<detected.faceLandmarks().size();i++){
                RectF box=new RectF(Float.MAX_VALUE,Float.MAX_VALUE,-Float.MAX_VALUE,-Float.MAX_VALUE);
                for(NormalizedLandmark point:detected.faceLandmarks().get(i)){
                    box.left=Math.min(box.left,point.x());box.top=Math.min(box.top,point.y());
                    box.right=Math.max(box.right,point.x());box.bottom=Math.max(box.bottom,point.y());
                }
                double area=(box.right-box.left)*(box.bottom-box.top);
                if(area>largest){largest=area;selected=i;}
            }
            List<List<Category>> values=detected.faceBlendshapes().get();
            if(selected<0||selected>=values.size())return Result.unavailable("selected_face_has_no_blendshapes");
            double left=coefficient(values.get(selected),"mouthSmileLeft");
            double right=coefficient(values.get(selected),"mouthSmileRight");
            double jaw=coefficient(values.get(selected),"jawOpen");
            if(!Double.isFinite(left)||!Double.isFinite(right)||!Double.isFinite(jaw))return Result.unavailable("required_coefficients_missing");
            Result result=new Result();result.available=true;result.faceCount=detected.faceLandmarks().size();result.selectedFace=selected;
            result.smileLeft=left;result.smileRight=right;result.smile=(left+right)/2;result.jawOpen=jaw;
            try{result.microExpression=MicroExpressionStats.from(values.get(selected));}
            catch(Exception error){android.util.Log.w("FaceScoreLab","Micro-expression statistics unavailable",error);}
            return result;
        }finally{input.close();}
    }
    private static double coefficient(List<Category> values,String name){
        for(Category value:values)if(name.equals(value.categoryName())){
            double score=value.score();return Double.isFinite(score)&&score>=0&&score<=1?score:Double.NaN;
        }
        return Double.NaN;
    }
}
