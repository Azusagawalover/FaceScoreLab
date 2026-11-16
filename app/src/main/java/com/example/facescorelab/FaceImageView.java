package com.example.facescorelab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/** 在同一缩放变换下绘制照片和检测结果，避免关键点与照片错位。 */
public class FaceImageView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final RectF target = new RectF();
    private Bitmap bitmap;
    private PointF[] points = new PointF[0];
    private boolean drawPoints = true;
    private Float hairline;

    public FaceImageView(Context context) { super(context); init(); }
    public FaceImageView(Context context, AttributeSet attrs) { super(context, attrs); init(); }
    private void init() { setLayerType(View.LAYER_TYPE_SOFTWARE, null); }
    public void setImage(Bitmap image, PointF[] landmarks) {
        bitmap = image;
        points = landmarks == null ? new PointF[0] : landmarks;
        invalidate();
    }
    public void setShowPoints(boolean show) { drawPoints = show; invalidate(); }
    public void setHairline(Float value) { hairline = value; invalidate(); }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(VisualStyle.SURFACE);
        canvas.drawRoundRect(0,0,getWidth(),getHeight(),dp(16),dp(16),paint);
        if (bitmap == null || bitmap.isRecycled()) return;
        float scale = Math.min((float)getWidth()/bitmap.getWidth(),(float)getHeight()/bitmap.getHeight());
        float width=bitmap.getWidth()*scale, height=bitmap.getHeight()*scale;
        target.set((getWidth()-width)/2,(getHeight()-height)/2,(getWidth()+width)/2,(getHeight()+height)/2);
        Path clip=new Path(); clip.addRoundRect(new RectF(0,0,getWidth(),getHeight()),dp(16),dp(16),Path.Direction.CW);
        canvas.save(); canvas.clipPath(clip);
        paint.setColor(Color.WHITE); paint.setAlpha(255);
        canvas.drawBitmap(bitmap,null,target,paint);
        if(drawPoints && points.length>0){
            float minX=Float.MAX_VALUE,minY=Float.MAX_VALUE,maxX=-Float.MAX_VALUE,maxY=-Float.MAX_VALUE;
            paint.setColor(Color.argb(205,15,118,110));
            for(PointF p:points){
                if(p==null||Float.isNaN(p.x)||Float.isNaN(p.y)) continue;
                minX=Math.min(minX,p.x); minY=Math.min(minY,p.y); maxX=Math.max(maxX,p.x); maxY=Math.max(maxY,p.y);
                canvas.drawCircle(target.left+p.x*scale,target.top+p.y*scale,dp(.55f),paint);
            }
            if(minX<maxX){
                paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(dp(.85f));
                canvas.drawRoundRect(target.left+minX*scale-dp(5),target.top+minY*scale-dp(5),
                        target.left+maxX*scale+dp(5),target.top+maxY*scale+dp(5),dp(8),dp(8),paint);
                paint.setStyle(Paint.Style.FILL);
            }
        }
        if(hairline!=null){
            paint.setColor(Color.rgb(255,206,99)); paint.setStrokeWidth(dp(2));
            float y=target.top+hairline*height;
            canvas.drawLine(target.left,y,target.right,y,paint);
            paint.setTextSize(dp(11));
            canvas.drawText("发际线校准",target.left+dp(8),Math.max(target.top+dp(14),y-dp(6)),paint);
        }
        canvas.restore();
    }
    private float dp(float value){return value*getResources().getDisplayMetrics().density;}
}
