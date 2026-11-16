package com.example.facescorelab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.os.SystemClock;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import java.util.List;

/** Loads the bundled generic landmark model. It never sends image data to a server. */
public final class LandmarkEngine implements AutoCloseable {
    private final FaceLandmarker detector;
    public volatile int lastFaceCount;
    public volatile long lastDetectionMillis;

    public LandmarkEngine(Context context) {
        BaseOptions base = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build();
        detector = FaceLandmarker.createFromOptions(context.getApplicationContext(),
                FaceLandmarker.FaceLandmarkerOptions.builder().setBaseOptions(base)
                        .setRunningMode(RunningMode.IMAGE).setNumFaces(3)
                        .setMinFaceDetectionConfidence(0.5f)
                        .setMinFacePresenceConfidence(0.5f)
                        .setMinTrackingConfidence(0.5f).build());
    }

    public synchronized PointF[] detect(Bitmap image) {
        long start = SystemClock.elapsedRealtime();
        // MPImage owns its bitmap and recycles it on close. Keep the caller's
        // photo alive for preview, saving, and repeated analysis.
        Bitmap argb = image.copy(Bitmap.Config.ARGB_8888, false);
        if (argb == null) throw new IllegalStateException("无法分配检测图像");
        MPImage input = new BitmapImageBuilder(argb).build();
        try {
            FaceLandmarkerResult result = detector.detect(input);
            lastFaceCount = result.faceLandmarks().size();
            if (lastFaceCount == 0) return new PointF[0];
            List<NormalizedLandmark> selected = null;
            double largest = -1;
            for (List<NormalizedLandmark> face : result.faceLandmarks()) {
                float minX=1, minY=1, maxX=0, maxY=0;
                for (NormalizedLandmark p : face) {
                    minX=Math.min(minX,p.x()); minY=Math.min(minY,p.y());
                    maxX=Math.max(maxX,p.x()); maxY=Math.max(maxY,p.y());
                }
                double area=(maxX-minX)*(maxY-minY);
                if (area>largest) {largest=area; selected=face;}
            }
            PointF[] points = new PointF[selected.size()];
            for (int i=0;i<points.length;i++)
                points[i]=new PointF(selected.get(i).x()*image.getWidth(),selected.get(i).y()*image.getHeight());
            return points;
        } finally {
            lastDetectionMillis=SystemClock.elapsedRealtime()-start;
            input.close();
        }
    }

    @Override public synchronized void close() { detector.close(); }
}
