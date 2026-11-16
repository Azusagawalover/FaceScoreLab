package com.example.facescorelab;

import android.content.Context;
import android.graphics.*;
import android.view.View;

/** Displays learned pixel masks in exactly the source photo's coordinates. */
public final class SegmentationView extends View {
    private final Bitmap photo;
    private final OnnxLocalEngine.Result result;
    private final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG|Paint.FILTER_BITMAP_FLAG);
    private final RectF photoRect=new RectF(),maskRect=new RectF();
    private Bitmap overlay;
    private int opacity=140;
    private boolean show=true;
    public SegmentationView(Context context,Bitmap photo,OnnxLocalEngine.Result result){
        super(context);this.photo=photo;this.result=result;setBackgroundColor(Color.rgb(234,238,247));setFilter(-1);
    }
    public void setOpacity(int percent){opacity=Math.round(percent*2.55f);invalidate();}
    public void setShow(boolean enabled){show=enabled;invalidate();}
    public void setFilter(int filter){
        if(result==null||result.mask==null)return;
        int[] pixels=new int[result.mask.length];
        for(int i=0;i<pixels.length;i++){
            int label=result.mask[i]&255;
            boolean selected=filter==-1||(filter==20&&label>=2&&label<=6)||(filter==21&&label==10)
                    ||(filter==22&&label>=11&&label<=13)||(filter==23&&label==17);
            pixels[i]=selected?OnnxLocalEngine.COLORS[label]:Color.TRANSPARENT;
        }
        overlay=Bitmap.createBitmap(pixels,result.size,result.size,Bitmap.Config.ARGB_8888);invalidate();
    }
    @Override protected void onDraw(Canvas canvas){
        super.onDraw(canvas);if(photo==null||photo.isRecycled())return;
        float scale=Math.min(getWidth()/(float)photo.getWidth(),getHeight()/(float)photo.getHeight());
        float x=(getWidth()-photo.getWidth()*scale)/2,y=(getHeight()-photo.getHeight()*scale)/2;
        photoRect.set(x,y,x+photo.getWidth()*scale,y+photo.getHeight()*scale);
        paint.setAlpha(255);canvas.drawBitmap(photo,null,photoRect,paint);
        if(show&&overlay!=null&&result!=null){
            Rect c=result.crop;maskRect.set(x+c.left*scale,y+c.top*scale,x+c.right*scale,y+c.bottom*scale);
            paint.setAlpha(opacity);paint.setFilterBitmap(false);canvas.drawBitmap(overlay,null,maskRect,paint);
            paint.setFilterBitmap(true);paint.setAlpha(255);
        }
    }
}
