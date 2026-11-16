package com.example.facescorelab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PointF;
import java.util.Iterator;
import org.json.JSONArray;
import org.json.JSONObject;

/** Synthetic coordinate/image contracts for visible measurements, without personal labels. */
public final class VisibleMeasurementContractChecks {
    private VisibleMeasurementContractChecks() {}

    public static boolean verifyKnownGeometryAndInvariance(Context context) throws Exception {
        Bitmap photo=solidPhoto();
        try{
            PointF[] points=points();
            ExtensionAnalyzer.ExtensionReport original=ExtensionAnalyzer.measureGeometry(points,photo);
            require(original.available,"Known valid geometry should be available");
            close(original.getValue("intercanthal_eye_center_ratio"),2.0/3.0,"Known eye-center ratio");
            close(original.getValue("nasal_projection_aspect"),1.0,"Known projected nasal aspect");
            ExtensionAnalyzer.ExtensionReport translated=ExtensionAnalyzer.measureGeometry(transform(points,1f,111f,71f),photo);
            ExtensionAnalyzer.ExtensionReport scaled=ExtensionAnalyzer.measureGeometry(transform(points,2.5f,111f,71f),photo);
            require(translated.available&&scaled.available,"Rigid translation/uniform scaling must retain valid geometry");
            for(String key:new String[]{"intercanthal_eye_center_ratio","nasal_projection_aspect"}){
                close(translated.getValue(key),original.getValue(key),"Translation invariance: "+key);
                close(scaled.getValue(key),original.getValue(key),"Uniform scale invariance: "+key);
            }
            require(Double.isNaN(original.getValue("not_a_measurement")),"A missing metric must remain missing");
            return true;
        }finally{photo.recycle();}
    }

    public static boolean verifyMissingAndNonFiniteAreUnavailable(Context context) throws Exception {
        Bitmap photo=solidPhoto();
        try{
            PointF[] nonFinite=new PointF[478];
            for(int i=0;i<nonFinite.length;i++)nonFinite[i]=new PointF(Float.NaN,Float.NaN);
            PointF[][] inputs={null,new PointF[10],new PointF[478],nonFinite};
            for(PointF[] input:inputs){
                ExtensionAnalyzer.ExtensionReport measured=ExtensionAnalyzer.measureGeometry(input,photo);
                require(!measured.available,"Missing/nonfinite geometry must not become an available measurement");
                require(Double.isNaN(measured.getValue("intercanthal_eye_center_ratio")),"Missing eye ratio became a fabricated zero");
                require(Double.isNaN(measured.getValue("nasal_projection_aspect")),"Missing nasal ratio became a fabricated zero");
                assertMeasurementJson(measured.toJson());
            }
            ExtensionAnalyzer.ExtensionReport absentActions=ExtensionAnalyzer.measureVisibleActions(null,Double.NaN);
            require(!absentActions.available,"Missing action points must remain unavailable");
            require(Double.isNaN(absentActions.getValue("ear_mean")),"Missing action points became zero EAR");
            assertMeasurementJson(absentActions.toJson());
            return true;
        }finally{photo.recycle();}
    }

    public static boolean verifyVisibleAperturesStayMeasurements(Context context) throws Exception {
        PointF[] points=points();
        ExtensionAnalyzer.ExtensionReport original=ExtensionAnalyzer.measureVisibleActions(points,87.0);
        require(original.available,"Known visible aperture fixture should be available");
        close(original.getValue("ear_left"),0.4,"Six-point EAR, first point group");
        close(original.getValue("ear_right"),0.4,"Six-point EAR, second point group");
        close(original.getValue("ear_mean"),0.4,"Mean EAR");
        close(original.getValue("mar"),0.2,"Known visible mouth opening ratio");
        close(original.getValue("geometric_similarity"),87.0,"Caller-provided geometric similarity");

        ExtensionAnalyzer.ExtensionReport moved=ExtensionAnalyzer.measureVisibleActions(transform(points,2.5f,33f,19f),87.0);
        require(moved.available,"Moved/scaled visible apertures should remain available");
        for(String key:new String[]{"ear_left","ear_right","ear_mean","mar"}){
            close(moved.getValue(key),original.getValue(key),"Aperture geometry invariance: "+key);
        }
        assertMeasurementJson(original.toJson());
        assertMeasurementJson(moved.toJson());

        // A measured closed aperture can be zero; missing input is the distinct NaN case above.
        PointF[] closed=points();
        for(int i:new int[]{160,158,153,144,159,145})closed[i].y=20f;
        for(int i:new int[]{385,387,373,380,386,374})closed[i].y=20f;
        closed[13].y=50f;closed[14].y=50f;
        ExtensionAnalyzer.ExtensionReport zero=ExtensionAnalyzer.measureVisibleActions(closed,Double.NaN);
        require(zero.available,"Visible closed apertures are valid raw measurements");
        close(zero.getValue("ear_mean"),0.0,"Measured zero EAR must be retained");
        close(zero.getValue("mar"),0.0,"Measured zero MAR must be retained");
        require(Double.isNaN(zero.getValue("geometric_similarity")),"Missing similarity must not become a fabricated value");
        assertMeasurementJson(zero.toJson());
        return true;
    }

    public static boolean verifyRealBlurDecreasesTextureEnergy(Context context) throws Exception {
        Bitmap photo=checkerboard();
        try{
            ExtensionAnalyzer.ExtensionReport plain=ExtensionAnalyzer.measureCaptureQuality(points(),photo,false);
            ExtensionAnalyzer.ExtensionReport blurred=ExtensionAnalyzer.measureCaptureQuality(points(),photo,true);
            require(plain.available&&blurred.available,"Synthetic texture should provide capture-quality measurements");
            double original=blurred.getValue("laplacian_energy");
            double control=blurred.getValue("laplacian_energy_blur");
            require(Double.isFinite(original)&&original>0,"Checkerboard must have nonzero texture energy");
            require(Double.isFinite(control)&&control>=0&&control<original,"Real mean blur must reduce checkerboard texture energy");
            close(plain.getValue("laplacian_energy"),original,"Control must not alter the original image measurement");
            require(Double.isNaN(plain.getValue("laplacian_energy_blur")),"A disabled control must not invent a blur result");
            require(Double.isFinite(blurred.getValue("laplacian_delta")),"Actual control difference should be recorded");
            require(Double.isFinite(blurred.getValue("brightness_mean")),"Brightness mean should be finite");
            require(Double.isFinite(blurred.getValue("brightness_std"))&&blurred.getValue("brightness_std")>0,
                    "Checkerboard brightness dispersion should be nonzero");
            require(photo.getPixel(20,20)==Color.BLACK&&photo.getPixel(21,20)==Color.WHITE,
                    "Computing a blur control must not mutate the source bitmap");
            assertMeasurementJson(plain.toJson());
            assertMeasurementJson(blurred.toJson());
            return true;
        }finally{photo.recycle();}
    }

    public static boolean verifyJsonIsMeasurementOnly(Context context) throws Exception {
        Bitmap photo=checkerboard();
        try{
            ExtensionAnalyzer.ExtensionReport[] reports={
                    ExtensionAnalyzer.measureGeometry(points(),photo),
                    ExtensionAnalyzer.measureVisibleActions(points(),87.0),
                    ExtensionAnalyzer.measureCaptureQuality(points(),photo,true)
            };
            for(ExtensionAnalyzer.ExtensionReport report:reports){
                JSONObject json=report.toJson();
                assertMeasurementJson(json);
                JSONArray measurements=json.getJSONArray("measurements");
                require(measurements.length()>0,"Available report must include actual measurements");
                // A fresh parse also verifies that optional/missing states remain valid JSON.
                assertMeasurementJson(new JSONObject(json.toString()));
            }
            return true;
        }finally{photo.recycle();}
    }

    private static void assertMeasurementJson(JSONObject json) throws Exception {
        require(json!=null,"Measurement JSON must exist");
        require(json.has("diagnosticUse")&&!json.getBoolean("diagnosticUse"),"Measurement report must explicitly disable diagnostic use");
        require(json.has("personalAttributeInference")&&!json.getBoolean("personalAttributeInference"),
                "Measurement report must explicitly disable personal-attribute inference");
        inspectJson(json);
    }

    private static void inspectJson(Object value) throws Exception {
        if(value instanceof JSONObject){
            JSONObject json=(JSONObject)value;
            Iterator<String> keys=json.keys();
            while(keys.hasNext()){
                String key=keys.next();
                String normalized=key.toLowerCase(java.util.Locale.ROOT);
                for(String forbidden:new String[]{"score","confidence","probability","emotion","emotion_label",
                        "predicted_emotion","diagnosis","disease_label","risk_score","sexual_orientation","inferred_orientation"}){
                    require(!normalized.equals(forbidden),"Raw measurements must not contain personal classification field: "+key);
                }
                inspectJson(json.get(key));
            }
        }else if(value instanceof JSONArray){
            JSONArray array=(JSONArray)value;
            for(int i=0;i<array.length();i++)inspectJson(array.get(i));
        }else if(value instanceof Number){
            require(Double.isFinite(((Number)value).doubleValue()),"JSON must not contain a nonfinite numeric value");
        }
    }

    private static PointF[] points(){
        PointF[] points=new PointF[478];
        for(int i=0;i<points.length;i++)points[i]=new PointF(30f,35f);
        put(points,33,10,20);put(points,133,20,20);put(points,362,40,20);put(points,263,50,20);
        put(points,160,12,18);put(points,158,18,18);put(points,153,18,22);put(points,144,12,22);
        put(points,385,42,18);put(points,387,48,18);put(points,373,48,22);put(points,380,42,22);
        put(points,159,15,18);put(points,145,15,22);put(points,386,45,18);put(points,374,45,22);
        put(points,168,30,25);put(points,1,30,45);put(points,98,20,45);put(points,327,40,45);
        put(points,61,20,50);put(points,291,40,50);put(points,13,30,48);put(points,14,30,52);
        return points;
    }

    private static PointF[] transform(PointF[] source,float scale,float dx,float dy){
        PointF[] output=new PointF[source.length];
        for(int i=0;i<source.length;i++)output[i]=new PointF(source[i].x*scale+dx,source[i].y*scale+dy);
        return output;
    }

    private static void put(PointF[] points,int index,float x,float y){points[index]=new PointF(x,y);}
    private static Bitmap solidPhoto(){
        Bitmap photo=Bitmap.createBitmap(512,512,Bitmap.Config.ARGB_8888);
        photo.eraseColor(Color.rgb(128,128,128));
        return photo;
    }
    private static Bitmap checkerboard(){
        int width=64,height=64;
        int[] pixels=new int[width*height];
        for(int y=0;y<height;y++)for(int x=0;x<width;x++)pixels[y*width+x]=((x+y)&1)==0?Color.BLACK:Color.WHITE;
        Bitmap photo=Bitmap.createBitmap(width,height,Bitmap.Config.ARGB_8888);
        photo.setPixels(pixels,0,width,0,0,width,height);
        return photo;
    }
    private static void close(double actual,double expected,String message){
        require(Double.isFinite(actual)&&Math.abs(actual-expected)<=1e-6*Math.max(1,Math.abs(expected)),
                message+": expected "+expected+", actual "+actual);
    }
    private static void require(boolean condition,String message){if(!condition)throw new AssertionError(message);}
}
