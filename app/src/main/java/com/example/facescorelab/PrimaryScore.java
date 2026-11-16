package com.example.facescorelab;

import android.graphics.Rect;
import org.json.JSONArray;
import org.json.JSONObject;

/** Immutable CNN measurement. The legacy geometry score deliberately remains separate. */
public final class PrimaryScore {
    public static final int SCHEMA_VERSION=2;
    public static final String MAPPING_VERSION="scut-1to5-to-1to10-v1";
    public static final String PREPROCESS_VERSION="landmark478-crop145-up006-rgb-imagenet-bilinear224-v1";
    public final double rawValue,displayValue;
    public final String modelVersion,modelSha256,preprocessVersion;
    public final String referenceProfile;
    public final boolean manualProfileSelected;
    public final long durationMillis;
    public final Rect crop;
    public final int inputWidth,inputHeight;

    public PrimaryScore(double rawValue,String modelVersion,String modelSha256,String preprocessVersion,
                        long durationMillis,Rect crop,int inputWidth,int inputHeight){
        this(rawValue,modelVersion,modelSha256,preprocessVersion,durationMillis,crop,inputWidth,inputHeight,ScoreProfiles.GENERAL,true);
    }
    public PrimaryScore(double rawValue,String modelVersion,String modelSha256,String preprocessVersion,
                        long durationMillis,Rect crop,int inputWidth,int inputHeight,String referenceProfile){
        this(rawValue,modelVersion,modelSha256,preprocessVersion,durationMillis,crop,inputWidth,inputHeight,referenceProfile,true);
    }
    private PrimaryScore(double rawValue,String modelVersion,String modelSha256,String preprocessVersion,
                         long durationMillis,Rect crop,int inputWidth,int inputHeight,String referenceProfile,boolean manualProfileSelected){
        if(!Double.isFinite(rawValue))throw new IllegalArgumentException("CNN评分无效");
        if(modelSha256==null||!modelSha256.matches("[0-9a-f]{64}"))throw new IllegalArgumentException("模型校验信息无效");
        if(!ScoreProfiles.isKnown(referenceProfile))throw new IllegalArgumentException("评分参照组无效");
        this.rawValue=rawValue;this.displayValue=display(rawValue);
        this.modelVersion=modelVersion;this.modelSha256=modelSha256;this.preprocessVersion=preprocessVersion;
        this.durationMillis=durationMillis;this.crop=new Rect(crop);this.inputWidth=inputWidth;this.inputHeight=inputHeight;
        this.referenceProfile=referenceProfile;this.manualProfileSelected=manualProfileSelected;
    }
    public static double display(double raw){return 1+9*(Math.max(1,Math.min(5,raw))-1)/4;}
    public JSONObject toJson()throws Exception{
        JSONObject json=new JSONObject();json.put("schemaVersion",SCHEMA_VERSION);json.put("kind","cnn");json.put("status","success");
        json.put("modelVersion",modelVersion);json.put("modelSha256",modelSha256);json.put("preprocessVersion",preprocessVersion);
        if(manualProfileSelected){
            json.put("referenceProfile",referenceProfile);json.put("referenceProfileSelection",ScoreProfiles.SELECTION);
            json.put("referenceProfileNote",ScoreProfiles.SELECTION_NOTE);
        }
        json.put("rawValue",rawValue);json.put("nativeScale",new JSONArray(new int[]{1,5}));
        json.put("displayValue",displayValue);json.put("displayScale",new JSONArray(new int[]{1,10}));
        json.put("mappingVersion",MAPPING_VERSION);json.put("displayMapping","1 + 9 * (clip(raw,1,5)-1) / 4");
        json.put("durationMillis",durationMillis);json.put("provider","CPU");
        json.put("crop",new JSONArray(new int[]{crop.left,crop.top,crop.right,crop.bottom}));
        json.put("inputWidth",inputWidth);json.put("inputHeight",inputHeight);return json;
    }
    /** Read-only history validation. Never remap or overwrite a previously stored score. */
    public static boolean isSupported(JSONObject json){
        if(json==null||json.optInt("schemaVersion")!=SCHEMA_VERSION||!"cnn".equals(json.optString("kind"))
                ||!"success".equals(json.optString("status"))||!MAPPING_VERSION.equals(json.optString("mappingVersion")))return false;
        if(json.has("referenceProfile")&&(!(json.opt("referenceProfile") instanceof String)
                ||!ScoreProfiles.isKnown(json.optString("referenceProfile"))
                ||!ScoreProfiles.SELECTION.equals(json.optString("referenceProfileSelection",ScoreProfiles.SELECTION))))return false;
        double raw=json.optDouble("rawValue",Double.NaN),shown=json.optDouble("displayValue",Double.NaN);
        return Double.isFinite(raw)&&Double.isFinite(shown)&&Math.abs(display(raw)-shown)<1e-6;
    }
    public static PrimaryScore fromJson(JSONObject json)throws Exception{
        if(!isSupported(json))throw new IllegalArgumentException("不支持的评分记录");
        JSONArray bounds=json.getJSONArray("crop");
        return new PrimaryScore(json.getDouble("rawValue"),json.getString("modelVersion"),json.getString("modelSha256"),
                json.getString("preprocessVersion"),json.optLong("durationMillis"),
                new Rect(bounds.getInt(0),bounds.getInt(1),bounds.getInt(2),bounds.getInt(3)),json.getInt("inputWidth"),json.getInt("inputHeight"),
                referenceProfileFromJson(json),json.has("referenceProfile"));
    }
    /** Old schema-2 CNN records used the general model even when geometry referenceMode differed. */
    public static String referenceProfileFromJson(JSONObject json){
        return json!=null&&json.has("referenceProfile")?json.optString("referenceProfile"):ScoreProfiles.GENERAL;
    }
    public static boolean matchesProfile(JSONObject json,String profile){
        return isSupported(json)&&referenceProfileFromJson(json).equals(ScoreProfiles.normalize(profile));
    }
}
