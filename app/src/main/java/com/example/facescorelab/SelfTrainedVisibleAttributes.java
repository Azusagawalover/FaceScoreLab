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

/** Optional three-label classifier, independent from FaceAttribNet and mouth action coefficients. */
public final class SelfTrainedVisibleAttributes implements AutoCloseable {
    public static final String MODEL_ASSET="visible_attributes.onnx";
    public static final String METADATA_ASSET="visible_attributes_model.json";
    public static final String FORMAT="visible-attributes-mnv3-v1";
    public static final String PREPROCESS_VERSION="mediapipe-largest-face-facecrop-v1-rgb224";
    public static final String FACE_TASK_SHA256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff";
    private static final String CROP_IMPLEMENTATION_SHA256="8205571ffe7e2e36511b27ff7e40fc4f104509f2fb2432660f38b3afd120a256";
    public static final String[] LABELS={"Smiling","Eyeglasses","Mouth_Slightly_Open"};
    public static final String[] DISPLAY_LABELS={"微笑","眼镜","嘴巴张开"};
    private static final Object MODEL_LOCK=new Object();
    private final Context context;
    private OrtSession session;
    private String modelHash;
    private double[] thresholds;
    private String initializationFailure;
    private boolean closed;

    public SelfTrainedVisibleAttributes(Context context){this.context=context.getApplicationContext();}

    public static final class Result {
        public final boolean available;
        public final String status,modelSha256,inputTensorSha256;
        public final double[] logits,scores,thresholds;
        public final long durationMillis;
        public final Rect crop;
        private Result(String status,String hash,String inputHash,double[] logits,double[] thresholds,long elapsed,Rect crop){
            this.status=status;available="success".equals(status);modelSha256=hash;inputTensorSha256=inputHash;
            this.logits=logits==null?null:logits.clone();this.thresholds=thresholds==null?null:thresholds.clone();
            durationMillis=elapsed;this.crop=crop==null?null:new Rect(crop);
            scores=logits==null?null:new double[3];
            if(available){
                if(!validHash(hash)||!validHash(inputHash)||logits.length!=3||thresholds.length!=3||crop==null||crop.width()<32||crop.height()<32||elapsed<0)
                    throw new IllegalArgumentException("Invalid visible attribute result");
                for(int i=0;i<3;i++){
                    if(!Double.isFinite(logits[i])||!Double.isFinite(thresholds[i])||thresholds[i]<=0||thresholds[i]>=1)
                        throw new IllegalArgumentException("Invalid visible attribute values");
                    scores[i]=sigmoid(logits[i]);
                }
            }
        }
        public static Result unavailable(String status){
            if("success".equals(status))throw new IllegalArgumentException("Unavailable result cannot be successful");
            return new Result(status,null,null,null,null,0,null);
        }
        public boolean detected(int index){return available&&scores[index]>=thresholds[index];}
        public JSONObject toJson()throws Exception{
            JSONObject json=new JSONObject().put("schemaVersion",1).put("available",available).put("status",status);
            if(!available)return json;
            json.put("modelVersion",FORMAT).put("modelSha256",modelSha256).put("preprocessVersion",PREPROCESS_VERSION)
                    .put("resizeImplementation","pillow-13-rgb-bilinear-antialias-v1")
                    .put("faceTaskSha256",FACE_TASK_SHA256).put("inputTensorSha256",inputTensorSha256)
                    .put("durationMillis",durationMillis).put("scoreType","uncalibrated sigmoid reference value")
                    .put("thresholdFitSplit","validation");
            json.put("crop",new JSONArray(new int[]{crop.left,crop.top,crop.right,crop.bottom}));
            JSONArray values=new JSONArray();
            for(int i=0;i<3;i++)values.put(new JSONObject().put("key",LABELS[i]).put("name",DISPLAY_LABELS[i])
                    .put("logit",logits[i]).put("score",scores[i]).put("threshold",thresholds[i]).put("present",detected(i)));
            return json.put("values",values);
        }
        public static Result fromJson(JSONObject json)throws Exception{
            if(json==null||json.getInt("schemaVersion")!=1)throw new IllegalArgumentException("Unknown visible attribute result schema");
            if(!json.getBoolean("available"))return unavailable(json.getString("status"));
            if(!"success".equals(json.getString("status"))||!FORMAT.equals(json.getString("modelVersion"))
                    ||!PREPROCESS_VERSION.equals(json.getString("preprocessVersion"))||!FACE_TASK_SHA256.equals(json.getString("faceTaskSha256"))
                    ||!"pillow-13-rgb-bilinear-antialias-v1".equals(json.getString("resizeImplementation"))
                    ||!"validation".equals(json.getString("thresholdFitSplit")))throw new IllegalArgumentException("Unknown visible attribute result contract");
            JSONArray values=json.getJSONArray("values"),box=json.getJSONArray("crop");
            if(values.length()!=3||box.length()!=4)throw new IllegalArgumentException("Invalid visible attribute result dimensions");
            double[] logits=new double[3],thresholds=new double[3];
            for(int i=0;i<3;i++){
                JSONObject value=values.getJSONObject(i);
                if(!LABELS[i].equals(value.getString("key"))||!DISPLAY_LABELS[i].equals(value.getString("name")))
                    throw new IllegalArgumentException("Visible attribute order changed");
                logits[i]=value.getDouble("logit");thresholds[i]=value.getDouble("threshold");
            }
            Result result=new Result("success",json.getString("modelSha256"),json.getString("inputTensorSha256"),logits,thresholds,
                    json.getLong("durationMillis"),new Rect(box.getInt(0),box.getInt(1),box.getInt(2),box.getInt(3)));
            for(int i=0;i<3;i++){
                JSONObject value=values.getJSONObject(i);double score=value.getDouble("score");
                if(!Double.isFinite(score)||Math.abs(score-result.scores[i])>1e-9||value.getBoolean("present")!=result.detected(i))
                    throw new IllegalArgumentException("Stored visible attribute decision is inconsistent");
            }
            return result;
        }
    }

    /** Missing or invalid optional assets never abort the existing deep-analysis modules. */
    public Result analyze(Bitmap photo,float[] landmarks){
        if(closed)throw new IllegalStateException("Visible attribute engine closed");
        if(landmarks==null||landmarks.length<956)return Result.unavailable("no_face");
        if(initializationFailure!=null)return Result.unavailable(initializationFailure);
        long start=SystemClock.elapsedRealtime();
        try{
            if(session==null)initialize();
            if(initializationFailure!=null)return Result.unavailable(initializationFailure);
            Rect bounds=OnnxLocalEngine.faceCrop(photo,landmarks);
            Bitmap crop=Bitmap.createBitmap(photo,bounds.left,bounds.top,bounds.width(),bounds.height());
            float[] values;
            try{values=preprocess(crop);}finally{if(crop!=photo)crop.recycle();}
            ByteBuffer exact=ByteBuffer.allocate(values.length*4).order(ByteOrder.LITTLE_ENDIAN);exact.asFloatBuffer().put(values);
            String inputHash=hash(exact.array());
            if((context.getApplicationInfo().flags&android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE)!=0){
                try(FileOutputStream output=new FileOutputStream(new File(context.getCacheDir(),"deep_visible_attributes_input.f32"))){output.write(exact.array());}
                catch(java.io.IOException error){android.util.Log.w("FaceScoreLab","Optional visible input trace unavailable",error);}
            }
            double[] logits=new double[3];
            try(OnnxTensor input=OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),FloatBuffer.wrap(values),new long[]{1,3,224,224});
                OrtSession.Result outputs=session.run(Collections.singletonMap("input",input))){
                FloatBuffer output=((OnnxTensor)outputs.get("logits").orElseThrow(()->new IllegalStateException("Missing logits"))).getFloatBuffer();
                if(output.remaining()!=3)throw new IllegalStateException("Expected three independent logits");
                for(int i=0;i<3;i++)logits[i]=output.get(i);
            }
            return new Result("success",modelHash,inputHash,logits,thresholds,SystemClock.elapsedRealtime()-start,bounds);
        }catch(Exception error){
            android.util.Log.w("FaceScoreLab","Optional self-trained visible attribute model unavailable",error);
            return Result.unavailable(session==null?"validation_failed":"inference_failed");
        }
    }

    private void initialize()throws Exception{
        String[] assetNames=context.getAssets().list("");
        if(assetNames==null||!java.util.Arrays.asList(assetNames).contains(MODEL_ASSET)||!java.util.Arrays.asList(assetNames).contains(METADATA_ASSET)){
            initializationFailure="not_installed";return;
        }
        JSONObject metadata;
        try(InputStream input=context.getAssets().open(METADATA_ASSET)){
            metadata=new JSONObject(new String(readAll(input),StandardCharsets.UTF_8));
        }catch(java.io.FileNotFoundException missing){initializationFailure="not_installed";return;}
        validateMetadata(metadata);
        try(InputStream input=context.getAssets().open("face_landmarker.task")){
            if(!FACE_TASK_SHA256.equals(hash(input)))throw new IllegalStateException("Face landmark model hash mismatch");
        }
        File model=new File(context.getFilesDir(),"visible-attributes-"+modelHash+".onnx");
        synchronized(MODEL_LOCK){
            boolean valid=false;
            if(model.isFile())try(InputStream input=new FileInputStream(model)){valid=modelHash.equals(hash(input));}
            if(!valid){
                try(InputStream input=context.getAssets().open(MODEL_ASSET);FileOutputStream output=new FileOutputStream(model)){
                    byte[] bytes=new byte[65536];int count;while((count=input.read(bytes))!=-1)output.write(bytes,0,count);
                }catch(java.io.FileNotFoundException missing){initializationFailure="not_installed";return;}
                try(InputStream input=new FileInputStream(model)){
                    if(!modelHash.equals(hash(input)))throw new IllegalStateException("Visible attribute model hash mismatch");
                }
            }
        }
        OrtSession loaded;
        try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()){
            options.setIntraOpNumThreads(2);options.setInterOpNumThreads(1);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setCPUArenaAllocator(false);options.setMemoryPatternOptimization(false);
            loaded=OrtEnvironment.getEnvironment().createSession(model.getAbsolutePath(),options);
        }
        try{
            if(loaded.getInputInfo().size()!=1||loaded.getOutputInfo().size()!=1
                    ||!loaded.getInputInfo().containsKey("input")||!loaded.getOutputInfo().containsKey("logits"))
                throw new IllegalArgumentException("Visible attribute ONNX names differ");
            TensorInfo input=(TensorInfo)loaded.getInputInfo().get("input").getInfo();
            TensorInfo output=(TensorInfo)loaded.getOutputInfo().get("logits").getInfo();
            long[] in=input.getShape(),out=output.getShape();
            if(input.type!=OnnxJavaType.FLOAT||output.type!=OnnxJavaType.FLOAT||in.length!=4||out.length!=2
                    ||in[1]!=3||in[2]!=224||in[3]!=224||out[1]!=3||(in[0]>0&&in[0]!=1)||(out[0]>0&&out[0]!=1))
                throw new IllegalArgumentException("Visible attribute ONNX dimensions differ");
            session=loaded;
        }catch(Exception error){loaded.close();throw error;}
    }

    private void validateMetadata(JSONObject metadata)throws Exception{
        if(!FORMAT.equals(metadata.getString("format"))||!MODEL_ASSET.equals(metadata.getString("model_file"))
                ||metadata.optString("status").contains("contract_only")||!metadata.has("smoke_not_for_release")
                ||metadata.getBoolean("smoke_not_for_release")||!"validation".equals(metadata.getString("threshold_fit_split"))
                ||!"sigmoid(logit[i]) >= thresholds[i]".equals(metadata.getString("decision")))
            throw new IllegalArgumentException("Untrained or unsupported visible attribute metadata");
        modelHash=metadata.getString("model_sha256");
        if(!validHash(modelHash))throw new IllegalArgumentException("Invalid model hash");
        JSONArray labels=metadata.getJSONArray("labels"),display=metadata.getJSONArray("display_labels"),cuts=metadata.getJSONArray("thresholds");
        if(labels.length()!=3||display.length()!=3||cuts.length()!=3)throw new IllegalArgumentException("Expected exactly three labels and thresholds");
        thresholds=new double[3];
        for(int i=0;i<3;i++){
            thresholds[i]=cuts.getDouble(i);
            if(!LABELS[i].equals(labels.getString(i))||!DISPLAY_LABELS[i].equals(display.getString(i))
                    ||!Double.isFinite(thresholds[i])||thresholds[i]<=0||thresholds[i]>=1)
                throw new IllegalArgumentException("Visible attribute label order or threshold invalid");
        }
        JSONObject input=metadata.getJSONObject("input"),output=metadata.getJSONObject("output");
        JSONArray in=input.getJSONArray("shape"),out=output.getJSONArray("shape");
        if(!"input".equals(input.getString("name"))||!"float32".equals(input.getString("dtype"))||!"NCHW".equals(input.getString("layout"))
                ||in.length()!=4||!"N".equals(in.getString(0))||in.getInt(1)!=3||in.getInt(2)!=224||in.getInt(3)!=224
                ||!"logits".equals(output.getString("name"))||out.length()!=2||!"N".equals(out.getString(0))||out.getInt(1)!=3
                ||!"independent sigmoid, NOT softmax".equals(output.getString("activation")))
            throw new IllegalArgumentException("Unsupported visible attribute tensor contract");
        JSONObject preprocessing=metadata.getJSONObject("preprocessing");
        if(!PREPROCESS_VERSION.equals(preprocessing.getString("version"))||!FACE_TASK_SHA256.equals(preprocessing.getString("face_task_sha256"))
                ||!CROP_IMPLEMENTATION_SHA256.equals(preprocessing.getString("crop_implementation_sha256"))
                ||preprocessing.getInt("num_faces")!=3||preprocessing.getDouble("min_face_detection_confidence")!=.5
                ||preprocessing.getDouble("min_face_presence_confidence")!=.5||preprocessing.getDouble("min_tracking_confidence")!=.5
                ||!"largest landmark bounding-box area; first on ties".equals(preprocessing.getString("face_selection"))
                ||!"224x224 bilinear".equals(preprocessing.getString("resize"))||!"RGB".equals(preprocessing.getString("color"))
                ||!"float32".equals(preprocessing.getString("dtype"))||!"NCHW contiguous".equals(preprocessing.getString("layout"))
                ||!"divide by 255".equals(preprocessing.getString("pixel_scale")))
            throw new IllegalArgumentException("Visible attribute preprocessing contract mismatch");
        JSONArray mean=preprocessing.getJSONArray("mean"),std=preprocessing.getJSONArray("std");
        float[] expectedMean={.485f,.456f,.406f},expectedStd={.229f,.224f,.225f};
        if(mean.length()!=3||std.length()!=3)throw new IllegalArgumentException("Invalid normalization dimensions");
        for(int i=0;i<3;i++)if(!Double.isFinite(mean.getDouble(i))||!Double.isFinite(std.getDouble(i))
                ||Math.abs(mean.getDouble(i)-expectedMean[i])>1e-7||Math.abs(std.getDouble(i)-expectedStd[i])>1e-7)
            throw new IllegalArgumentException("Visible attribute normalization differs");
    }

    private static float[] preprocess(Bitmap crop){
        int[] original=new int[crop.getWidth()*crop.getHeight()];crop.getPixels(original,0,crop.getWidth(),0,0,crop.getWidth(),crop.getHeight());
        int[] pixels=VisibleAttributeResize.resize(original,crop.getWidth(),crop.getHeight(),224,224);
        return VisibleAttributeResize.imageNetTensor(pixels);
    }
    public static double sigmoid(double logit){
        if(logit>=0)return 1/(1+Math.exp(-logit));double value=Math.exp(logit);return value/(1+value);
    }
    private static boolean validHash(String value){return value!=null&&value.matches("[0-9a-f]{64}");}
    private static String hash(byte[] bytes)throws Exception{return hex(MessageDigest.getInstance("SHA-256").digest(bytes));}
    private static String hash(InputStream input)throws Exception{
        MessageDigest digest=MessageDigest.getInstance("SHA-256");byte[] bytes=new byte[65536];int count;
        while((count=input.read(bytes))!=-1)digest.update(bytes,0,count);return hex(digest.digest());
    }
    private static String hex(byte[] bytes){StringBuilder output=new StringBuilder();for(byte value:bytes)output.append(String.format(Locale.ROOT,"%02x",value&255));return output.toString();}
    private static byte[] readAll(InputStream input)throws Exception{
        ByteArrayOutputStream output=new ByteArrayOutputStream();byte[] bytes=new byte[4096];int count;
        while((count=input.read(bytes))!=-1)output.write(bytes,0,count);return output.toByteArray();
    }
    @Override public void close(){closed=true;if(session!=null)try{session.close();}catch(Exception error){android.util.Log.w("FaceScoreLab","Visible attribute session close failed",error);}session=null;}
}
