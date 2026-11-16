package com.example.facescorelab;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.hardware.camera2.*;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.*;
import android.util.Size;
import android.view.*;
import android.widget.*;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/** Native Camera2 preview and local frame analysis. All image work stays on a worker. */
public class CameraActivity extends AppCompatActivity {
    private static final int PURPLE=VisualStyle.ACCENT, INK=VisualStyle.INK;
    private FrameCanvas preview;
    private TextView status, summary;
    private Button capture, flip;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private ImageReader reader;
    private HandlerThread thread;
    private Handler worker;
    private final Handler main=new Handler(Looper.getMainLooper());
    private volatile boolean running, capturing;
    private volatile int cameraGeneration;
    private volatile Bitmap currentFrame;
    private LandmarkEngine engine;
    private GeometryAnalyzer analyzer;
    private boolean live, front=true, eyeClosed, mouthOpen, permissionRequested, permissionPending;
    private String referenceMode;
    private int sensorOrientation=90, blinks, mouths;
    private long lastFrameTime, lastBlinkTime;
    private double smoothed=Double.NaN;
    private static final long CNN_INTERVAL_MS=500,CNN_MAX_AGE_MS=1800;
    private final ExecutorService scoreWorker=Executors.newSingleThreadExecutor();
    private final AtomicBoolean scoreInFlight=new AtomicBoolean(false);
    private final AtomicInteger scoreGeneration=new AtomicInteger();
    private OnnxLocalEngine scoreEngine; // Owned exclusively by scoreWorker.
    private PrimaryScore latestCnn; // Published back on the camera worker.
    private long lastCnnRequest,lastCnnResult;
    private String cnnFailure="";

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        live=getIntent().getBooleanExtra("live_mode",false);
        referenceMode=ScoreProfiles.normalize(getIntent().getStringExtra("reference_mode"));
        getWindow().setStatusBarColor(VisualStyle.SURFACE);
        getWindow().setNavigationBarColor(VisualStyle.SURFACE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        LinearLayout root=new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(VisualStyle.SURFACE); root.setPadding(dp(16),dp(12),dp(16),dp(12));
        LinearLayout bar=new LinearLayout(this); bar.setGravity(Gravity.CENTER_VERTICAL);
        TextView back=new TextView(this); back.setText("‹ 返回"); back.setTextSize(VisualStyle.PAGE); back.setTextColor(PURPLE);
        back.setPadding(0,dp(8),dp(16),dp(8)); back.setOnClickListener(v->finish());
        bar.addView(back);
        TextView title=new TextView(this); title.setText(live?"实时人脸与动作分析":"拍照分析");
        title.setTextSize(VisualStyle.CARD_TITLE); title.setTextColor(INK); title.setTypeface(null,Typeface.BOLD);
        bar.addView(title); root.addView(bar);
        preview=new FrameCanvas();
        LinearLayout.LayoutParams imageParams=new LinearLayout.LayoutParams(-1,0,1);
        imageParams.setMargins(0,dp(12),0,dp(10)); root.addView(preview,imageParams);
        status=new TextView(this); status.setText("正在启动相机…"); status.setTextColor(INK); status.setTextSize(VisualStyle.BODY);
        status.setMinHeight(dp(48)); root.addView(status);
        summary=new TextView(this); summary.setText(live?"眨眼 0 次  ·  张口 0 次\n动作统计仅描述画面中的可见变化":"请将人脸放在画面中央，保持光线均匀");
        summary.setTextColor(VisualStyle.MUTED); summary.setTextSize(VisualStyle.BODY); summary.setPadding(0,dp(6),0,dp(12)); root.addView(summary);
        LinearLayout actions=new LinearLayout(this);
        flip=VisualStyle.button(this,"切换镜头",false); flip.setOnClickListener(v->switchCamera());
        actions.addView(flip,new LinearLayout.LayoutParams(0,dp(52),1));
        capture=VisualStyle.button(this,live?"定格并查看结果":"拍摄并分析",true);
        capture.setEnabled(false); capture.setOnClickListener(v->captureFrame());
        LinearLayout.LayoutParams cp=new LinearLayout.LayoutParams(0,dp(52),1.5f); cp.setMargins(dp(8),0,0,0);
        actions.addView(capture,cp); root.addView(actions); setContentView(root);
        thread=new HandlerThread("face-camera-worker"); thread.start(); worker=new Handler(thread.getLooper());
    }

    @Override protected void onResume(){
        super.onResume(); running=true; capturing=false;
        if(checkSelfPermission(Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED){
            if(!permissionPending)restartCamera(false);
        }else if(!permissionRequested){
            permissionRequested=true;permissionPending=true;
            requestPermissions(new String[]{Manifest.permission.CAMERA},701);
        }else if(!permissionPending){
            status.setText("未获得相机权限。可返回后使用相册或示例图片。");
        }
    }

    @Override public void onRequestPermissionsResult(int request,@NonNull String[] permissions,@NonNull int[] grants){
        super.onRequestPermissionsResult(request,permissions,grants);
        if(request==701){
            permissionPending=false;
            if(grants.length>0 && grants[0]==PackageManager.PERMISSION_GRANTED){if(running)restartCamera(false);}
            else {status.setText("未获得相机权限。可返回后使用相册或示例图片。"); capture.setEnabled(false);}
        }
    }

    private void switchCamera(){
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){
            status.setText("未获得相机权限。可返回后使用相册或示例图片。");return;
        }
        restartCamera(true);
    }

    private void restartCamera(boolean toggle){
        scoreGeneration.incrementAndGet();
        currentFrame=null; capture.setEnabled(false); flip.setEnabled(false);
        status.setText(toggle?"正在切换镜头…":"正在启动相机…");
        worker.post(()->{
            stopDevices();
            if(!running)return;
            if(toggle)front=!front;
            lastFrameTime=0; eyeClosed=false; mouthOpen=false;clearCnn();
            openCamera();
        });
    }

    @SuppressWarnings("MissingPermission") private void openCamera(){
        if(!running || worker==null)return;
        try {
            CameraManager manager=(CameraManager)getSystemService(CAMERA_SERVICE);
            String chosen=null; CameraCharacteristics characteristics=null;
            String[] ids=manager.getCameraIdList();
            for(String id:ids){
                CameraCharacteristics c=manager.getCameraCharacteristics(id);
                Integer facing=c.get(CameraCharacteristics.LENS_FACING);
                if(facing!=null && facing==(front?CameraCharacteristics.LENS_FACING_FRONT:CameraCharacteristics.LENS_FACING_BACK)){
                    chosen=id; characteristics=c; break;
                }
            }
            if(chosen==null && ids.length>0){chosen=ids[0]; characteristics=manager.getCameraCharacteristics(chosen);}
            if(chosen==null){fail("此设备没有可用摄像头，请使用相册分析。");return;}
            front=Objects.equals(characteristics.get(CameraCharacteristics.LENS_FACING),CameraCharacteristics.LENS_FACING_FRONT);
            Integer orientation=characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            sensorOrientation=orientation==null?90:orientation;
            StreamConfigurationMap map=characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if(map==null){fail("相机没有可用图像格式。");return;}
            Size selected=null; long distance=Long.MAX_VALUE;
            for(Size s:map.getOutputSizes(ImageFormat.YUV_420_888)){
                long candidate=Math.abs((long)s.getWidth()*s.getHeight()-640*480);
                if(candidate<distance){distance=candidate;selected=s;}
            }
            if(selected==null){fail("相机不支持所需图像格式。");return;}
            final int token=cameraGeneration;
            final ImageReader openedReader=ImageReader.newInstance(selected.getWidth(),selected.getHeight(),ImageFormat.YUV_420_888,2);
            reader=openedReader;
            reader.setOnImageAvailableListener(source->receiveFrame(source,token),worker);
            manager.openCamera(chosen,new CameraDevice.StateCallback(){
                @Override public void onOpened(@NonNull CameraDevice device){
                    if(!running || token!=cameraGeneration){device.close();return;}
                    camera=device; startSession(device,openedReader,token);
                }
                @Override public void onDisconnected(@NonNull CameraDevice device){device.close();if(token==cameraGeneration)fail("相机已断开，请重新进入。");}
                @Override public void onError(@NonNull CameraDevice device,int code){device.close();if(token==cameraGeneration)fail("相机暂不可用（"+code+"），请检查是否被其他应用占用。");}
            },worker);
        }catch(Exception error){fail("无法打开相机："+safeMessage(error));}
    }

    private void startSession(CameraDevice openedCamera,ImageReader openedReader,int token){
        try {
            if(!running || token!=cameraGeneration)return;
            final Surface target=openedReader.getSurface();
            openedCamera.createCaptureSession(Collections.singletonList(target),new CameraCaptureSession.StateCallback(){
                @Override public void onConfigured(@NonNull CameraCaptureSession configured){
                    if(!running || token!=cameraGeneration || camera!=openedCamera){configured.close();return;} session=configured;
                    try {
                        CaptureRequest.Builder request=openedCamera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        request.addTarget(target); request.set(CaptureRequest.CONTROL_MODE,CaptureRequest.CONTROL_MODE_AUTO);
                        session.setRepeatingRequest(request.build(),null,worker);
                    }catch(Exception e){fail("相机预览启动失败："+safeMessage(e));}
                }
                @Override public void onConfigureFailed(@NonNull CameraCaptureSession failed){if(token==cameraGeneration)fail("相机预览配置失败，请尝试另一个镜头。");}
            },worker);
        }catch(Exception e){fail("相机配置失败："+safeMessage(e));}
    }

    private void receiveFrame(ImageReader source,int token){
        Image image=null;
        try {
            image=source.acquireLatestImage(); if(image==null)return;
            long now=SystemClock.elapsedRealtime();
            if(!running || token!=cameraGeneration || now-lastFrameTime<150)return;
            lastFrameTime=now;
            Bitmap frame=decodeFrame(image);
            image.close(); image=null;
            if(frame==null)return;
            PointF[] points=new PointF[0];
            String info="画面已就绪，点击下方按钮拍摄";
            if(live){
                if(engine==null)engine=new LandmarkEngine(this);
                if(analyzer==null)analyzer=new GeometryAnalyzer(this);
                points=engine.detect(frame);
                if(points.length>0){
                    AnalysisResult result=analyzer.analyze(points,frame.getWidth(),frame.getHeight(),referenceMode,null);
                    if(result.scorable && engine.lastFaceCount==1){
                        updateActions(result,now);
                        requestCnn(frame,points,now,token);
                        if(latestCnn!=null&&now-lastCnnResult<CNN_MAX_AGE_MS){
                            info=String.format(Locale.CHINA,"%s %.1f / 10（平滑）\n几何对照 %.1f / 10 · CNN %d ms",ScoreProfiles.label(referenceMode),smoothed,result.score,latestCnn.durationMillis);
                        }else info=String.format(Locale.CHINA,"%s\n几何对照 %.1f / 10",cnnFailure.isEmpty()?"CNN 图像评分计算中…":cnnFailure,result.score);
                    }else{
                        eyeClosed=false;mouthOpen=false;clearCnn();
                        info=engine.lastFaceCount>1?"检测到多人，请让一张正面人脸进入画面":result.qualityMessage;
                    }
                }else{info="未检测到人脸，请面向镜头并保持光线充足";eyeClosed=false;mouthOpen=false;clearCnn();}
            }
            final PointF[] drawPoints=points; final String message=info;
            currentFrame=frame;
            main.post(()->{
                if(!running || token!=cameraGeneration || isFinishing())return;
                preview.setFrame(frame,drawPoints); status.setText(message); capture.setEnabled(!capturing);flip.setEnabled(!capturing);
                if(live)summary.setText("眨眼 "+blinks+" 次  ·  张口 "+mouths+" 次\n动作统计仅描述画面中的可见变化");
            });
        }catch(Exception e){if(running)fail("画面处理失败："+safeMessage(e));}
        finally{if(image!=null)image.close();}
    }

    /** At most one CNN frame in flight; excess frames are dropped, never queued. */
    private void requestCnn(Bitmap frame,PointF[] points,long now,int cameraToken){
        if(!running||capturing||scoreWorker.isShutdown()||now-lastCnnRequest<CNN_INTERVAL_MS||!scoreInFlight.compareAndSet(false,true))return;
        lastCnnRequest=now;
        final int scoreToken=scoreGeneration.get();
        final Bitmap owned=frame.copy(Bitmap.Config.ARGB_8888,false);
        if(owned==null){scoreInFlight.set(false);return;}
        final float[] coordinates=new float[points.length*2];
        for(int i=0;i<points.length;i++){coordinates[2*i]=points[i].x;coordinates[2*i+1]=points[i].y;}
        try{scoreWorker.execute(()->{
            try{
                if(!running||cameraToken!=cameraGeneration||scoreToken!=scoreGeneration.get())return;
                if(scoreEngine==null)scoreEngine=new OnnxLocalEngine(this,referenceMode);
                PrimaryScore measured=scoreEngine.scoreOnly(owned,coordinates,false);
                worker.post(()->{
                    if(!running||cameraToken!=cameraGeneration||scoreToken!=scoreGeneration.get()||SystemClock.elapsedRealtime()-now>CNN_MAX_AGE_MS)return;
                    latestCnn=measured;lastCnnResult=now;cnnFailure="";
                    smoothed=Double.isNaN(smoothed)?measured.displayValue:.75*smoothed+.25*measured.displayValue;
                    android.util.Log.d("FaceScoreLab",String.format(Locale.ROOT,"Live CNN %s raw=%.6f display=%.6f ms=%d",measured.modelVersion,measured.rawValue,measured.displayValue,measured.durationMillis));
                });
            }catch(Exception error){
                android.util.Log.w("FaceScoreLab","Live CNN unavailable",error);
                worker.post(()->{if(running&&cameraToken==cameraGeneration&&scoreToken==scoreGeneration.get()){latestCnn=null;smoothed=Double.NaN;cnnFailure="CNN 暂不可用，正在重试";}});
            }finally{owned.recycle();scoreInFlight.set(false);}
        });}catch(RejectedExecutionException stopped){owned.recycle();scoreInFlight.set(false);}
    }
    private void clearCnn(){
        scoreGeneration.incrementAndGet();latestCnn=null;lastCnnRequest=0;lastCnnResult=0;smoothed=Double.NaN;cnnFailure="";
    }

    private void updateActions(AnalysisResult r,long time){
        if(r.eyeAspect<0.17)eyeClosed=true;
        else if(r.eyeAspect>0.23 && eyeClosed){if(time-lastBlinkTime>350){blinks++;lastBlinkTime=time;}eyeClosed=false;}
        if(r.mouthAspect>0.18 && !mouthOpen){mouths++;mouthOpen=true;}
        else if(r.mouthAspect<0.10)mouthOpen=false;
    }

    private Bitmap decodeFrame(Image image)throws IOException{
        int w=image.getWidth(),h=image.getHeight(); byte[] nv21=new byte[w*h*3/2];
        Image.Plane[] planes=image.getPlanes();
        for(int plane=0;plane<3;plane++){
            ByteBuffer buffer=planes[plane].getBuffer();
            int rowStride=planes[plane].getRowStride(),pixelStride=planes[plane].getPixelStride();
            int cols=plane==0?w:w/2,rows=plane==0?h:h/2;
            for(int row=0;row<rows;row++)for(int col=0;col<cols;col++){
                int index=row*rowStride+col*pixelStride;
                int destination=plane==0?row*w+col:w*h+row*w+col*2+(plane==1?1:0);
                nv21[destination]=buffer.get(index);
            }
        }
        ByteArrayOutputStream encoded=new ByteArrayOutputStream();
        if(!new YuvImage(nv21,ImageFormat.NV21,w,h,null).compressToJpeg(new Rect(0,0,w,h),85,encoded))return null;
        byte[] bytes=encoded.toByteArray(); Bitmap raw=BitmapFactory.decodeByteArray(bytes,0,bytes.length);
        int rotation=getWindowManager().getDefaultDisplay().getRotation();
        int degrees=rotation==Surface.ROTATION_90?90:rotation==Surface.ROTATION_180?180:rotation==Surface.ROTATION_270?270:0;
        int angle=front?(sensorOrientation+degrees)%360:(sensorOrientation-degrees+360)%360;
        Matrix matrix=new Matrix(); matrix.postRotate(angle); if(front)matrix.postScale(-1,1);
        Bitmap oriented=Bitmap.createBitmap(raw,0,0,w,h,matrix,true);
        if(oriented!=raw)raw.recycle();
        return oriented;
    }

    private void captureFrame(){
        Bitmap bitmap=currentFrame; if(!running || capturing || bitmap==null){status.setText("画面尚未就绪，请稍候。");return;}
        capturing=true;capture.setEnabled(false);flip.setEnabled(false);
        worker.post(()->{
            try {
                File file=File.createTempFile("camera-",".jpg",getCacheDir());
                try(FileOutputStream stream=new FileOutputStream(file)){if(!bitmap.compress(Bitmap.CompressFormat.JPEG,95,stream))throw new IOException("图片保存失败");}
                main.post(()->{if(!running || isFinishing())return;setResult(RESULT_OK,new Intent().putExtra("image_path",file.getAbsolutePath()).putExtra("reference_mode",referenceMode));finish();});
            }catch(Exception error){main.post(()->{if(!running)return;capturing=false;status.setText("拍摄失败，请重试。");capture.setEnabled(true);flip.setEnabled(true);});}
        });
    }

    private void fail(String text){main.post(()->{if(running&&!isFinishing()){status.setText(text);currentFrame=null;capture.setEnabled(false);flip.setEnabled(true);}});}
    private String safeMessage(Exception error){String s=error.getMessage();return s==null?error.getClass().getSimpleName():s;}
    private int dp(float n){return Math.round(n*getResources().getDisplayMetrics().density);}

    private void stopDevices(){
        cameraGeneration++;
        try{if(session!=null){session.close();session=null;}}catch(Exception ignored){}
        try{if(camera!=null){camera.close();camera=null;}}catch(Exception ignored){}
        try{if(reader!=null){reader.close();reader=null;}}catch(Exception ignored){}
    }

    @Override protected void onPause(){
        scoreGeneration.incrementAndGet();
        running=false; currentFrame=null; capture.setEnabled(false);preview.setFrame(null,new PointF[0]);
        worker.post(()->{stopDevices();clearCnn();if(engine!=null){engine.close();engine=null;}});
        scoreWorker.execute(()->{if(scoreEngine!=null){scoreEngine.close();scoreEngine=null;}});
        super.onPause();
    }

    @Override protected void onDestroy(){
        running=false;scoreGeneration.incrementAndGet();
        scoreWorker.execute(()->{if(scoreEngine!=null){scoreEngine.close();scoreEngine=null;}});scoreWorker.shutdown();
        worker.post(()->{stopDevices();if(engine!=null){engine.close();engine=null;}thread.quitSafely();});
        super.onDestroy();
    }

    private final class FrameCanvas extends View{
        private Bitmap bitmap; private PointF[] points=new PointF[0]; private final Paint paint=new Paint(3);
        FrameCanvas(){super(CameraActivity.this);setBackgroundColor(VisualStyle.BORDER);}
        void setFrame(Bitmap frame,PointF[] landmarks){bitmap=frame;points=landmarks;invalidate();}
        @Override protected void onDraw(Canvas canvas){
            super.onDraw(canvas); if(bitmap==null)return;
            float scale=Math.min(getWidth()/(float)bitmap.getWidth(),getHeight()/(float)bitmap.getHeight());
            float x=(getWidth()-bitmap.getWidth()*scale)/2,y=(getHeight()-bitmap.getHeight()*scale)/2;
            canvas.save();canvas.translate(x,y);canvas.scale(scale,scale);paint.setColor(Color.WHITE);
            canvas.drawBitmap(bitmap,0,0,paint);paint.setColor(Color.argb(200,96,245,181));
            for(int i=0;i<points.length;i+=3)canvas.drawCircle(points[i].x,points[i].y,1.0f/scale,paint);
            if(points.length>0){
                RectF rect=new RectF(Float.MAX_VALUE,Float.MAX_VALUE,-Float.MAX_VALUE,-Float.MAX_VALUE);
                for(PointF p:points){rect.left=Math.min(rect.left,p.x);rect.top=Math.min(rect.top,p.y);rect.right=Math.max(rect.right,p.x);rect.bottom=Math.max(rect.bottom,p.y);}
                paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(1.2f/scale);canvas.drawRoundRect(rect,10,10,paint);paint.setStyle(Paint.Style.FILL);
            }
            canvas.restore();
        }
    }
}
