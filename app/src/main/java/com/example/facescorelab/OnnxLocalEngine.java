package com.example.facescorelab;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Rect;
import android.os.SystemClock;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Collections;
import org.json.JSONArray;
import org.json.JSONObject;

/** Local image models with independent scoring, visible attributes and region ablation. */
public final class OnnxLocalEngine implements AutoCloseable {
    public static final String CNN_VERSION="scut-mobilenetv3-small-selftrained-fold1-v1";
    public static final String ATTRIBUTE_VERSION="faceattribnet-qualcomm-2025-v1";
    public static final String[] ATTRIBUTE_NAMES={"左眼睁开","右眼睁开","普通眼镜","口罩","太阳镜"};
    public static final String[] ATTRIBUTE_KEYS={"left_eye_open","right_eye_open","eyeglasses","mask","sunglasses"};
    public static final String PARSER_VERSION="bisenet-resnet18-celebamaskhq-v1";
    public static final String FPEM_VERSION="fpem-official-public-0.9243-port-v1";
    public static final String[] LABELS={"背景","皮肤","左眉","右眉","左眼","右眼","眼镜","左耳","右耳","耳饰","鼻子","口腔","上唇","下唇","颈部","项链","衣服","头发","帽子"};
    public static final int[] COLORS={0,0xfff6b34e,0xffb45ef7,0xff8f66e8,0xff4cceeb,0xff37b5dd,0xff6ed8a0,0xffed8c7b,0xffd7758a,0xfff295b5,0xff70d97e,0xff9c5eb5,0xfff3669c,0xffd84f75,0xffd7b698,0xffe9d076,0xff8caee3,0xff6261df,0xffd1a164};
    private final Context context;
    private final String referenceProfile;
    private ScoreProfiles.Profile scoringProfile;
    private static final Object MODEL_FILE_LOCK=new Object();
    private JSONObject manifest;
    private OrtSession beautySession,attributeSession,parsingSession;
    private SelfTrainedVisibleAttributes selfTrainedAttributes;
    private boolean closed;
    public interface Progress {void update(String message);}
    public static final class Result {
        public PrimaryScore primaryScore;
        public double rawScore,score;
        public double fpemRawScore=Double.NaN,fpemScore=Double.NaN;
        public long fpemMillis;
        public long cnnMillis,parsingMillis;
        public long attributeMillis,explanationMillis;
        public long expressionMillis;
        public VisibleExpressionEngine.Result mouthActions;
        public SelfTrainedVisibleAttributes.Result selfTrainedVisibleAttributes;
        /** 分割皮肤区域上的亮度离散度；见 ScoringBasis.skinUniformity。 */
        public ScoringBasis.SkinUniformity skinUniformity;
        public double[] attributes;
        public final JSONArray occlusion=new JSONArray();
        public int size=512;
        public Rect crop;
        public byte[] mask;
        public final int[] counts=new int[LABELS.length];
        public JSONObject toJson() throws Exception {
            JSONObject json=new JSONObject();
            json.put("schemaVersion",PrimaryScore.SCHEMA_VERSION);
            json.put("cnnModel",primaryScore==null?CNN_VERSION:primaryScore.modelVersion);json.put("segmentationModel",PARSER_VERSION);
            if(primaryScore!=null)json.put("primaryScore",primaryScore.toJson());
            json.put("cnnRawScore",rawScore);json.put("cnnNativeScale","1-5");json.put("cnnScore",score);
            json.put("displayMapping","1 + 9 * (clip(raw,1,5)-1) / 4");
            json.put("cnnMillis",cnnMillis);json.put("parsingMillis",parsingMillis);
            json.put("attributeMillis",attributeMillis);json.put("explanationMillis",explanationMillis);
            json.put("attributeModel",ATTRIBUTE_VERSION);
            json.put("expressionMillis",expressionMillis);
            if(mouthActions!=null)json.put("mouthActions",mouthActions.toJson());
            if(selfTrainedVisibleAttributes!=null)json.put("selfTrainedVisibleAttributes",selfTrainedVisibleAttributes.toJson());
            if(attributes!=null){
                JSONArray values=new JSONArray();
                for(int i=0;i<attributes.length;i++){
                    JSONObject value=new JSONObject();value.put("key",ATTRIBUTE_KEYS[i]);value.put("name",ATTRIBUTE_NAMES[i]);
                    value.put("probability",attributes[i]);value.put("present",attributes[i]>=.5);values.put(value);
                }
                json.put("visibleAttributes",values);
            }
            if(skinUniformity!=null)json.put("skinUniformity",skinUniformity.toJson());
            json.put("regionAblation",occlusion);
            json.put("explanationMethod","Segment-region occlusion with RGB 128 replacement; original score minus occluded score. Model sensitivity, not a causal beauty standard.");
            if(Double.isFinite(fpemScore)){
                json.put("fpemModel",FPEM_VERSION);json.put("fpemRawScore",fpemRawScore);
                json.put("fpemScore",fpemScore);json.put("fpemMillis",fpemMillis);
            }
            json.put("crop",new JSONArray(new int[]{crop.left,crop.top,crop.right,crop.bottom}));
            JSONArray regions=new JSONArray();
            for(int i=0;i<counts.length;i++){
                JSONObject region=new JSONObject();region.put("id",i);region.put("name",LABELS[i]);
                region.put("pixels",counts[i]);region.put("percent",100.0*counts[i]/(size*size));regions.put(region);
            }
            json.put("regions",regions);
            json.put("accuracyNote","Self-trained SCUT fold-1 MobileNetV3 regressor. Held-out benchmark results are in scut_evaluation.json; they do not establish real-phone accuracy. FaceAttribNet and parser use third-party weights. Optional self-trained visible labels have separate provenance and validation thresholds.");
            return json;
        }
    }
    public OnnxLocalEngine(Context context){this(context,ScoreProfiles.GENERAL);}
    public OnnxLocalEngine(Context context,String referenceMode){
        this.context=context.getApplicationContext();this.referenceProfile=ScoreProfiles.normalize(referenceMode);
    }
    public String getReferenceProfile(){return referenceProfile;}
    private ScoreProfiles.Profile scoringProfile()throws Exception{
        if(scoringProfile==null)scoringProfile=ScoreProfiles.resolve(manifest(),referenceProfile);
        return scoringProfile;
    }

    /** Lightweight shared entry point: one CNN run, with no parser, attributes or explanation. */
    public PrimaryScore scoreOnly(Bitmap photo,float[] points)throws Exception{return scoreOnly(photo,points,true);}
    public PrimaryScore scoreOnly(Bitmap photo,float[] points,boolean writeDebugInput)throws Exception{
        Rect bounds=faceCrop(photo,points);
        Bitmap crop=Bitmap.createBitmap(photo,bounds.left,bounds.top,bounds.width(),bounds.height());
        try{return scoreCrop(crop,bounds,photo.getWidth(),photo.getHeight(),writeDebugInput);}
        finally{if(crop!=photo)crop.recycle();}
    }

    /**
     * 直接对一个已裁剪的人脸位图打分，返回 1～5 原始分。
     * 供鲁棒性演示复用同一个已加载会话，避免反复创建会话。
     */
    public double scoreFaceBitmap(Bitmap face)throws Exception{
        if(closed)throw new IllegalStateException("推理引擎已关闭");
        if(face==null||face.isRecycled())throw new IllegalArgumentException("人脸位图不可用");
        OrtSession session=beautySession();
        try(OnnxTensor input=OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(preprocess(face,224,false)),new long[]{1,3,224,224});
            OrtSession.Result output=session.run(Collections.singletonMap(session.getInputNames().iterator().next(),input))){
            return ((OnnxTensor)output.get(0)).getFloatBuffer().get(0);
        }
    }
    private PrimaryScore scoreCrop(Bitmap crop,Rect bounds,int width,int height,boolean writeDebugInput)throws Exception{
        long start=SystemClock.elapsedRealtime();
        OrtSession session=beautySession();double raw;
        try(OnnxTensor input=OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),FloatBuffer.wrap(preprocess(crop,224,writeDebugInput)),new long[]{1,3,224,224});
            OrtSession.Result output=session.run(Collections.singletonMap(session.getInputNames().iterator().next(),input))){
            raw=((OnnxTensor)output.get(0)).getFloatBuffer().get(0);
        }
        ScoreProfiles.Profile profile=scoringProfile();
        return new PrimaryScore(raw,profile.modelVersion,profile.sha256,profile.preprocessVersion,
                SystemClock.elapsedRealtime()-start,bounds,width,height,referenceProfile);
    }
    private OrtSession newSession(String asset)throws Exception{
        if(closed)throw new IllegalStateException("推理引擎已关闭");
        try(OrtSession.SessionOptions options=options()){
            return OrtEnvironment.getEnvironment().createSession(modelFile(asset),options);
        }
    }
    private OrtSession beautySession()throws Exception{
        if(closed)throw new IllegalStateException("推理引擎已关闭");
        if(beautySession==null)beautySession=newSession(scoringProfile().file);return beautySession;
    }
    private OrtSession attributeSession()throws Exception{
        if(closed)throw new IllegalStateException("推理引擎已关闭");
        if(attributeSession==null)attributeSession=newSession("face_attributes.onnx");return attributeSession;
    }
    private OrtSession parsingSession()throws Exception{
        if(closed)throw new IllegalStateException("推理引擎已关闭");
        if(parsingSession==null)parsingSession=newSession("face_parsing_mask.onnx");return parsingSession;
    }
    /** Call on the same owning worker, after its inference tasks have finished. */
    @Override public void close(){
        closed=true;
        for(OrtSession session:new OrtSession[]{beautySession,attributeSession,parsingSession}){
            if(session!=null)try{session.close();}catch(Exception error){android.util.Log.w("FaceScoreLab","Session close failed",error);}
        }
        beautySession=null;attributeSession=null;parsingSession=null;
        if(selfTrainedAttributes!=null)selfTrainedAttributes.close();selfTrainedAttributes=null;
    }

    public void analyzeFpem(Bitmap photo,Result result)throws Exception{
        if(closed)throw new IllegalStateException("推理引擎已关闭");
        // Keep the large optional model from overlapping parser/attribute working memory.
        for(OrtSession session:new OrtSession[]{attributeSession,parsingSession})if(session!=null)session.close();
        attributeSession=null;parsingSession=null;
        if(selfTrainedAttributes!=null)selfTrainedAttributes.close();selfTrainedAttributes=null;
        Bitmap crop=Bitmap.createBitmap(photo,result.crop.left,result.crop.top,result.crop.width(),result.crop.height());
        OrtEnvironment environment=OrtEnvironment.getEnvironment();long start=SystemClock.elapsedRealtime();
        try(OrtSession.SessionOptions options=options();
            OrtSession session=environment.createSession(modelFile("fpem_mobile.onnx"),options);
            OnnxTensor input224=OnnxTensor.createTensor(environment,FloatBuffer.wrap(preprocessFpem(crop,224)),new long[]{1,3,224,224});
            OnnxTensor input112=OnnxTensor.createTensor(environment,FloatBuffer.wrap(preprocessFpem(crop,112)),new long[]{1,3,112,112});
            OnnxTensor input160=OnnxTensor.createTensor(environment,FloatBuffer.wrap(preprocessFpem(crop,160)),new long[]{1,3,160,160})){
            java.util.Map<String,OnnxTensor> inputs=new java.util.HashMap<>();
            inputs.put("image224",input224);inputs.put("image112",input112);inputs.put("image160",input160);
            try(OrtSession.Result output=session.run(inputs)){
                double raw=((OnnxTensor)output.get(0)).getFloatBuffer().get(0);
                if(!Double.isFinite(raw))throw new IllegalStateException("FPEM输出无效");
                result.fpemRawScore=raw;result.fpemScore=1+9*(Math.max(1,Math.min(5,raw))-1)/4;
            }
            result.fpemMillis=SystemClock.elapsedRealtime()-start;
        }finally{if(crop!=photo)crop.recycle();}
    }
    private float[] preprocessFpem(Bitmap photo,int size){
        float factor=size/(float)Math.max(photo.getWidth(),photo.getHeight());
        int width=Math.max(1,(int)(photo.getWidth()*factor)),height=Math.max(1,(int)(photo.getHeight()*factor));
        Bitmap scaled=Bitmap.createScaledBitmap(photo,width,height,true);
        int[] pixels=new int[width*height];scaled.getPixels(pixels,0,width,0,0,width,height);
        int area=size*size;float[] data=new float[area*3];
        for(int y=0;y<height;y++)for(int x=0;x<width;x++){
            int pixel=pixels[y*width+x],i=y*size+x;
            data[i]=((pixel>>16)&255)/255f;data[area+i]=((pixel>>8)&255)/255f;data[2*area+i]=(pixel&255)/255f;
        }
        if((context.getApplicationInfo().flags&android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE)!=0){
            Bitmap padded=Bitmap.createBitmap(size,size,Bitmap.Config.ARGB_8888);padded.eraseColor(Color.BLACK);
            new android.graphics.Canvas(padded).drawBitmap(scaled,0,0,null);
            try(FileOutputStream output=new FileOutputStream(new File(context.getCacheDir(),"deep_fpem_"+size+".png"))){padded.compress(Bitmap.CompressFormat.PNG,100,output);}catch(Exception ignored){}
            padded.recycle();
        }
        if(scaled!=photo)scaled.recycle();return data;
    }

    public Result analyze(Bitmap photo,float[] points,Progress progress)throws Exception{
        Result result=new Result();result.crop=faceCrop(photo,points);
        Bitmap crop=Bitmap.createBitmap(photo,result.crop.left,result.crop.top,result.crop.width(),result.crop.height());
        OrtEnvironment environment=OrtEnvironment.getEnvironment();
        try{
            progress.update("正在运行"+ScoreProfiles.label(referenceProfile)+"…");
            result.primaryScore=scoreCrop(crop,result.crop,photo.getWidth(),photo.getHeight(),true);
            result.rawScore=result.primaryScore.rawValue;result.score=result.primaryScore.displayValue;
            result.cnnMillis=result.primaryScore.durationMillis;
            progress.update("正在分割眼眉、鼻子、嘴唇与头发…");
            long start=SystemClock.elapsedRealtime();
            OrtSession parser=parsingSession();
            try(OnnxTensor input=OnnxTensor.createTensor(environment,FloatBuffer.wrap(preprocess(crop,512)),new long[]{1,3,512,512});
                OrtSession.Result output=parser.run(Collections.singletonMap(parser.getInputNames().iterator().next(),input))){
                ByteBuffer values=((OnnxTensor)output.get(0)).getByteBuffer();
                if(values.remaining()!=512*512)throw new IllegalStateException("分割输出尺寸无效");
                result.mask=new byte[512*512];values.get(result.mask);
                for(byte b:result.mask){int label=b&255;if(label>=LABELS.length)throw new IllegalStateException("分割类别无效");result.counts[label]++;}
            }
            result.parsingMillis=SystemClock.elapsedRealtime()-start;
            progress.update("正在统计肤色区域均匀度…");
            result.skinUniformity=ScoringBasis.skinUniformityForCrop(crop,result.mask,result.size,result.size,result.crop,
                    ScoringBasis.faceBox(pointsToPoints(points),photo.getWidth(),photo.getHeight()));
            progress.update("正在分析眼睛、眼镜与口罩等可见属性…");
            start=SystemClock.elapsedRealtime();
            OrtSession attributes=attributeSession();
            try(OnnxTensor input=OnnxTensor.createTensor(environment,FloatBuffer.wrap(preprocessAttributes(crop)),new long[]{1,3,128,128});
                OrtSession.Result output=attributes.run(Collections.singletonMap(attributes.getInputNames().iterator().next(),input))){
                FloatBuffer values=((OnnxTensor)output.get(0)).getFloatBuffer();
                if(values.remaining()!=5)throw new IllegalStateException("属性输出尺寸无效");
                result.attributes=new double[5];
                for(int i=0;i<5;i++){
                    double value=values.get(i);
                    if(!Double.isFinite(value)||value<0||value>1)throw new IllegalStateException("属性输出无效");
                    result.attributes[i]=value;
                }
            }
            result.attributeMillis=SystemClock.elapsedRealtime()-start;
            progress.update("正在检查自训练可见属性模型…");
            if(selfTrainedAttributes==null)selfTrainedAttributes=new SelfTrainedVisibleAttributes(context);
            result.selfTrainedVisibleAttributes=selfTrainedAttributes.analyze(photo,points);
            progress.update("正在检测微笑与嘴部张开等可见动作…");
            start=SystemClock.elapsedRealtime();
            try{result.mouthActions=new VisibleExpressionEngine(context).analyze(photo);}
            catch(Exception error){
                android.util.Log.w("FaceScoreLab","Visible mouth action analysis unavailable",error);
                result.mouthActions=VisibleExpressionEngine.Result.unavailable("analysis_failed");
            }
            result.expressionMillis=SystemClock.elapsedRealtime()-start;
            progress.update("正在计算五官区域对模型评分的影响…");
            start=SystemClock.elapsedRealtime();explain(crop,result,environment);
            result.explanationMillis=SystemClock.elapsedRealtime()-start;
            return result;
        }finally{if(crop!=photo)crop.recycle();}
    }
    private void explain(Bitmap crop,Result result,OrtEnvironment environment)throws Exception{
        String[] names={"眼眉","鼻子","嘴唇","头发","皮肤"};
        int[][] labels={{2,3,4,5},{10},{11,12,13},{17},{1}};
        float[] original=preprocess(crop,224);int area=224*224;
        float[] fill={(128/255f-.485f)/.229f,(128/255f-.456f)/.224f,(128/255f-.406f)/.225f};
        OrtSession session=beautySession();
        {
            for(int region=0;region<names.length;region++){
                float[] altered=original.clone();int changed=0;
                for(int y=0;y<224;y++)for(int x=0;x<224;x++){
                    int sourceY=Math.min(511,(int)((y+.5)*512/224));
                    int sourceX=Math.min(511,(int)((x+.5)*512/224));
                    int label=result.mask[sourceY*512+sourceX]&255;boolean include=false;
                    for(int id:labels[region])if(label==id){include=true;break;}
                    if(!include)continue;
                    int pixel=y*224+x;for(int c=0;c<3;c++)altered[c*area+pixel]=fill[c];changed++;
                }
                if(changed==0)continue;
                try(OnnxTensor tensor=OnnxTensor.createTensor(environment,FloatBuffer.wrap(altered),new long[]{1,3,224,224});
                    OrtSession.Result output=session.run(Collections.singletonMap(session.getInputNames().iterator().next(),tensor))){
                    double raw=((OnnxTensor)output.get(0)).getFloatBuffer().get(0);
                    if(!Double.isFinite(raw))throw new IllegalStateException("解释计算无效");
                    double score=1+9*(Math.max(1,Math.min(5,raw))-1)/4;
                    JSONObject item=new JSONObject();item.put("name",names[region]);item.put("maskedPixels",changed);
                    item.put("occludedRawScore",raw);item.put("occludedScore",score);
                    item.put("deltaRaw",result.rawScore-raw);item.put("deltaScore",result.score-score);
                    result.occlusion.put(item);
                }
            }
        }
    }
    private float[] preprocessAttributes(Bitmap crop){
        int size=128;float factor=size/(float)Math.max(crop.getWidth(),crop.getHeight());
        int width=Math.max(1,(int)Math.floor(crop.getWidth()*factor));
        int height=Math.max(1,(int)Math.floor(crop.getHeight()*factor));
        if(crop.getWidth()>=crop.getHeight())width=size;else height=size;
        Bitmap scaled=Bitmap.createScaledBitmap(crop,width,height,true);
        Bitmap padded=Bitmap.createBitmap(size,size,Bitmap.Config.ARGB_8888);padded.eraseColor(Color.BLACK);
        new android.graphics.Canvas(padded).drawBitmap(scaled,(size-width)/2,(size-height)/2,null);
        int area=size*size;int[] pixels=new int[area];padded.getPixels(pixels,0,size,0,0,size,size);
        float[] data=new float[area*3];
        for(int i=0;i<area;i++){
            int p=pixels[i];data[i]=((p>>16)&255)/255f;data[area+i]=((p>>8)&255)/255f;data[2*area+i]=(p&255)/255f;
        }
        if((context.getApplicationInfo().flags&android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE)!=0){
            try(FileOutputStream output=new FileOutputStream(new File(context.getCacheDir(),"deep_attribute_input.png"))){padded.compress(Bitmap.CompressFormat.PNG,100,output);}catch(Exception ignored){}
        }
        padded.recycle();if(scaled!=crop)scaled.recycle();return data;
    }
    private OrtSession.SessionOptions options()throws Exception{
        OrtSession.SessionOptions options=new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(2);options.setInterOpNumThreads(1);
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setCPUArenaAllocator(false);options.setMemoryPatternOptimization(false);
        return options;
    }
    private String modelFile(String asset)throws Exception{
        synchronized(MODEL_FILE_LOCK){return prepareModelFile(asset);}
    }
    private JSONObject manifest()throws Exception{
        if(manifest!=null)return manifest;
        try(InputStream input=context.getAssets().open("deep_models.json");java.io.ByteArrayOutputStream bytes=new java.io.ByteArrayOutputStream()){
            byte[] data=new byte[8192];int n;while((n=input.read(data))!=-1)bytes.write(data,0,n);
            manifest=new JSONObject(bytes.toString("UTF-8"));return manifest;
        }
    }
    private String modelHash(String asset)throws Exception{return manifest().getJSONObject("assets").getJSONObject(asset).getString("sha256");}
    private String prepareModelFile(String asset)throws Exception{
        // Content-addressed names also invalidate equal-size retrained model caches.
        String expectedHash=modelHash(asset);
        if(!expectedHash.matches("[0-9a-f]{64}"))throw new IllegalStateException("模型校验标识无效");
        File folder=new File(context.getNoBackupFilesDir(),"deep_models_v2");
        if(!folder.isDirectory()&&!folder.mkdirs())throw new IllegalStateException("模型缓存目录不可用");
        File file=new File(folder,expectedHash.substring(0,16)+"-"+asset);
        try(InputStream input=context.getAssets().open(asset);AssetFileDescriptor descriptor=context.getAssets().openFd(asset)){
            long expected=descriptor.getLength();
            if(file.isFile()&&file.length()==expected&&expectedHash.equals(sha256(file)))return file.getAbsolutePath();
            File legacy=new File(folder,asset);
            if(legacy.isFile()&&legacy.length()==expected&&expectedHash.equals(sha256(legacy))&&legacy.renameTo(file))return file.getAbsolutePath();
            File temporary=new File(folder,expectedHash.substring(0,16)+"-"+asset+".tmp");
            try(FileOutputStream output=new FileOutputStream(temporary)){
                byte[] buffer=new byte[65536];int n;
                while((n=input.read(buffer))!=-1)output.write(buffer,0,n);
                output.getFD().sync();
            }
            if(!expectedHash.equals(sha256(temporary))){temporary.delete();throw new IllegalStateException("模型文件校验失败");}
            if(!temporary.renameTo(file))throw new IllegalStateException("模型缓存写入失败");
        }
        return file.getAbsolutePath();
    }
    private static String sha256(File file)throws Exception{
        java.security.MessageDigest digest=java.security.MessageDigest.getInstance("SHA-256");
        try(InputStream input=new java.io.FileInputStream(file)){
            byte[] buffer=new byte[65536];int n;while((n=input.read(buffer))!=-1)digest.update(buffer,0,n);
        }
        StringBuilder result=new StringBuilder();for(byte value:digest.digest())result.append(String.format(java.util.Locale.ROOT,"%02x",value&255));
        return result.toString();
    }
    private float[] preprocess(Bitmap bitmap,int size){return preprocess(bitmap,size,true);}
    private float[] preprocess(Bitmap bitmap,int size,boolean writeDebugInput){
        Bitmap scaled=Bitmap.createScaledBitmap(bitmap,size,size,true);
        int area=size*size;int[] pixels=new int[area];scaled.getPixels(pixels,0,size,0,0,size,size);
        if(writeDebugInput&&(context.getApplicationInfo().flags&android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE)!=0){
            try(FileOutputStream output=new FileOutputStream(new File(context.getCacheDir(),size==224?"deep_cnn_input.png":"deep_parser_input.png"))){scaled.compress(Bitmap.CompressFormat.PNG,100,output);}catch(Exception ignored){}
        }
        float[] data=new float[area*3];
        for(int i=0;i<area;i++){
            int p=pixels[i];data[i]=(((p>>16)&255)/255f-.485f)/.229f;
            data[area+i]=(((p>>8)&255)/255f-.456f)/.224f;
            data[2*area+i]=((p&255)/255f-.406f)/.225f;
        }
        if(scaled!=bitmap)scaled.recycle();return data;
    }
    /** 关键点数组 → PointF[]，供人脸框换算使用。 */
    private static android.graphics.PointF[] pointsToPoints(float[] points){
        if(points==null||points.length<2)return null;
        android.graphics.PointF[] output=new android.graphics.PointF[points.length/2];
        for(int i=0;i<output.length;i++)output[i]=new android.graphics.PointF(points[2*i],points[2*i+1]);
        return output;
    }
    public static Rect faceCrop(Bitmap image,float[] points){
        if(points==null||points.length<478*2)throw new IllegalArgumentException("需要完整人脸关键点");
        float left=Float.MAX_VALUE,top=Float.MAX_VALUE,right=-Float.MAX_VALUE,bottom=-Float.MAX_VALUE;
        for(int i=0;i<points.length;i+=2){
            float x=points[i],y=points[i+1];
            if(!Float.isFinite(x)||!Float.isFinite(y))throw new IllegalArgumentException("关键点无效");
            left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);
        }
        float size=Math.max(right-left,bottom-top)*1.45f;
        float cx=(left+right)/2,cy=(top+bottom)/2-size*.06f;
        int x1=Math.max(0,(int)Math.floor(cx-size/2)),y1=Math.max(0,(int)Math.floor(cy-size/2));
        int x2=Math.min(image.getWidth(),(int)Math.ceil(cx+size/2)),y2=Math.min(image.getHeight(),(int)Math.ceil(cy+size/2));
        if(x2-x1<32||y2-y1<32)throw new IllegalArgumentException("人脸裁剪区域过小");
        return new Rect(x1,y1,x2,y2);
    }
}
