package com.example.facescorelab;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PointF;
import android.graphics.Rect;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.json.JSONArray;
import org.json.JSONObject;

/** Manual score-profile contracts; no profile is inferred from a photo or legacy geometry mode. */
public final class ScoreProfileContractChecks {
    private static final String[] PROFILES={ScoreProfiles.GENERAL,ScoreProfiles.MALE,ScoreProfiles.FEMALE};
    private ScoreProfileContractChecks() {}

    public static boolean verifyRegistryAndFiles(Context context) throws Exception {
        JSONObject manifest=manifest(context);Set<String> files=new HashSet<>(),hashes=new HashSet<>();
        for(String id:PROFILES){
            ScoreProfiles.Profile profile=ScoreProfiles.resolve(manifest,id);
            require(id.equals(profile.id),"Profile identity changed");
            require(files.add(profile.file)&&hashes.add(profile.sha256),"Three manual profiles need distinct files and weights");
            try(InputStream input=context.getAssets().open(profile.file)){
                require(profile.sha256.equals(hash(input)),"Profile asset hash mismatch: "+id);
            }
            if(!ScoreProfiles.GENERAL.equals(id)){
                require(profile.notice.contains("尚未证实更准"),"Experimental profile must not claim greater accuracy");
                require(profile.notice.contains("验证MAE"),"Profile must disclose its group-validation comparison");
            }
        }
        ScoreProfiles.Profile general=ScoreProfiles.resolve(manifest,ScoreProfiles.GENERAL);
        require("beauty_mobilenetv3_small.onnx".equals(general.file),"The original general model must remain the general default");
        require(manifest.getJSONObject("cnn").getString("model_version").equals(general.modelVersion),"General model provenance changed");
        require(ScoreProfiles.GENERAL.equals(ScoreProfiles.normalize(null)),"An absent profile must default to general");
        return true;
    }

    public static boolean verifyManifestRejectsMixedAssets(Context context) throws Exception {
        JSONObject manifest=manifest(context);
        JSONObject changed=new JSONObject(manifest.toString());
        changed.getJSONObject("score_profiles").getJSONObject("male").put("sha256",
                ScoreProfiles.resolve(manifest,ScoreProfiles.GENERAL).sha256);
        requireResolveRejected(changed,ScoreProfiles.MALE,"A male profile must not silently use general-model metadata");
        changed=new JSONObject(manifest.toString());
        changed.getJSONObject("score_profiles").getJSONObject("female").put("file","beauty_mobilenetv3_small.onnx");
        requireResolveRejected(changed,ScoreProfiles.FEMALE,"A female profile must not point at the general weights");
        JSONObject legacy=new JSONObject(manifest.toString());legacy.remove("score_profiles");
        require(ScoreProfiles.GENERAL.equals(ScoreProfiles.resolve(legacy,ScoreProfiles.GENERAL).id),"Legacy general manifest must remain supported");
        requireResolveRejected(legacy,ScoreProfiles.MALE,"A missing specialized model must not masquerade as a working profile");
        return true;
    }

    public static boolean verifyPrimaryScoreRoundTripsAndLegacy(Context context) throws Exception {
        for(String id:PROFILES){
            PrimaryScore score=sample(id);JSONObject json=score.toJson();PrimaryScore restored=PrimaryScore.fromJson(json);
            require(id.equals(restored.referenceProfile)&&restored.manualProfileSelected,"Manual profile was lost on round trip");
            require(json.toString().equals(restored.toJson().toString()),"New profile score round trip changed fields");
            require(PrimaryScore.matchesProfile(json,id),"Saved profile no longer matches itself");
            for(String other:PROFILES)if(!other.equals(id))require(!PrimaryScore.matchesProfile(json,other),"Different profile reused a saved CNN result");
            require(ScoreProfiles.SELECTION.equals(json.getString("referenceProfileSelection")),"Profile choice must be manual");
        }
        JSONObject legacy=sample(ScoreProfiles.GENERAL).toJson();
        legacy.remove("referenceProfile");legacy.remove("referenceProfileSelection");legacy.remove("referenceProfileNote");
        // Older referenceMode described the geometry comparison, not the CNN's model.
        legacy.put("referenceMode",ScoreProfiles.FEMALE);
        PrimaryScore old=PrimaryScore.fromJson(legacy);
        require(ScoreProfiles.GENERAL.equals(old.referenceProfile)&&!old.manualProfileSelected,"Legacy geometry mode was mistaken for a dedicated CNN profile");
        require(!old.toJson().has("referenceProfile"),"Reading a legacy score must not invent a saved manual choice");
        require(old.rawValue==3.25&&old.displayValue==6.0625,"Legacy score was recomputed or remapped");
        JSONObject invalid=sample(ScoreProfiles.GENERAL).toJson();invalid.put("referenceProfile","unknown-future-profile");
        require(!PrimaryScore.isSupported(invalid),"Unknown profile must not be reinterpreted as general");
        return true;
    }

    public static boolean verifyModelSessionMetadata(Context context) throws Exception {
        Bitmap photo;try(InputStream input=context.getAssets().open("demo/girl.jpg")){photo=BitmapFactory.decodeStream(input);}
        require(photo!=null,"Demo fixture must decode");
        try(LandmarkEngine detector=new LandmarkEngine(context)){
            PointF[] landmarks=detector.detect(photo);require(landmarks!=null&&landmarks.length>=478,"Demo needs full landmarks");
            float[] points=new float[landmarks.length*2];
            for(int i=0;i<landmarks.length;i++){points[i*2]=landmarks[i].x;points[i*2+1]=landmarks[i].y;}
            JSONObject manifest=manifest(context);Rect crop=null;
            for(String id:PROFILES){
                ScoreProfiles.Profile expected=ScoreProfiles.resolve(manifest,id);
                try(OnnxLocalEngine engine=new OnnxLocalEngine(context,id)){
                    PrimaryScore first=engine.scoreOnly(photo,points,false),again=engine.scoreOnly(photo,points,false);
                    require(id.equals(engine.getReferenceProfile())&&id.equals(first.referenceProfile),"Session selected the wrong profile");
                    require(expected.sha256.equals(first.modelSha256)&&expected.modelVersion.equals(first.modelVersion),"Inference metadata does not identify selected weights");
                    require(expected.preprocessVersion.equals(first.preprocessVersion),"Selected model preprocessing drifted");
                    require(Math.abs(first.rawValue-again.rawValue)<1e-6,"Same-profile session reuse is not deterministic");
                    if(crop==null)crop=new Rect(first.crop);else require(crop.equals(first.crop),"Profiles must use the identical input crop");
                    require(PrimaryScore.matchesProfile(first.toJson(),id),"Actual inference lost its manual profile");
                }
            }
            return true;
        }finally{photo.recycle();}
    }

    public static boolean verifyHistoryRetainsSavedProfile(Context context) throws Exception {
        final File files=new File(context.getCacheDir(),"contract-score-profiles-"+System.nanoTime()+"/files");
        require(files.mkdirs(),"Cannot create isolated profile history");
        Context isolated=new ContextWrapper(context){@Override public File getFilesDir(){return files;}};
        HistoryStore history=new HistoryStore(isolated);File folder=new File(files,"history");
        File image=new File(folder,"legacy.jpg"),index=new File(folder,"records.json");
        Bitmap bitmap=Bitmap.createBitmap(40,40,Bitmap.Config.ARGB_8888);
        try{
            try(FileOutputStream output=new FileOutputStream(image)){bitmap.compress(Bitmap.CompressFormat.JPEG,90,output);}
            JSONObject oldPrimary=sample(ScoreProfiles.GENERAL).toJson();
            oldPrimary.remove("referenceProfile");oldPrimary.remove("referenceProfileSelection");oldPrimary.remove("referenceProfileNote");
            JSONObject old=new JSONObject().put("id","legacy-profile-record").put("imagePath",image.getAbsolutePath())
                    .put("timestamp",123456789L).put("score",7.0).put("referenceMode",ScoreProfiles.FEMALE)
                    .put("schemaVersion",2).put("legacyScoreKind","geometry").put("primaryScore",oldPrimary).put("unknown","preserve");
            write(index,new JSONArray().put(old).toString());String before=read(index);
            List<HistoryStore.Record> legacy=history.readAll();
            require(legacy.size()==1&&legacy.get(0).hasPrimaryScore(),"Old primary score must remain readable");
            require(ScoreProfiles.GENERAL.equals(PrimaryScore.referenceProfileFromJson(legacy.get(0).primaryScore)),"Old geometry preference changed CNN interpretation");
            require(before.equals(read(index)),"Opening profile history rewrote saved data");
            AnalysisResult result=new AnalysisResult();result.score=6.5;result.scorable=true;result.referenceMode=ScoreProfiles.FEMALE;
            result.primaryScore=sample(ScoreProfiles.FEMALE);result.primaryScoreStatus="success";
            result.deepAnalysis=new JSONObject().put("primaryScore",result.primaryScore.toJson());
            history.save(bitmap,result);
            List<HistoryStore.Record> saved=history.readAll();
            require(saved.size()==2&&ScoreProfiles.FEMALE.equals(PrimaryScore.referenceProfileFromJson(saved.get(0).primaryScore)),"New manual profile was not saved");
            require(saved.get(1).json.toString().equals(old.toString()),"Saving a new profile rewrote an old score or unknown field");
            require(ScoreProfiles.FEMALE.equals(saved.get(0).json.getJSONObject("deepAnalysis").getJSONObject("primaryScore").getString("referenceProfile")),"Deep history uses another profile");
            return true;
        }finally{bitmap.recycle();}
    }

    private static PrimaryScore sample(String profile){
        return new PrimaryScore(3.25,"profile-contract",String.format(java.util.Locale.ROOT,"%064d",0),PrimaryScore.PREPROCESS_VERSION,
                3,new Rect(0,0,40,40),40,40,profile);
    }
    private static JSONObject manifest(Context context) throws Exception {
        try(InputStream input=context.getAssets().open("deep_models.json");ByteArrayOutputStream output=new ByteArrayOutputStream()){
            byte[] bytes=new byte[8192];int count;while((count=input.read(bytes))!=-1)output.write(bytes,0,count);
            return new JSONObject(new String(output.toByteArray(),StandardCharsets.UTF_8));
        }
    }
    private static void requireResolveRejected(JSONObject manifest,String profile,String message) throws Exception {
        boolean rejected=false;try{ScoreProfiles.resolve(manifest,profile);}catch(Exception expected){rejected=true;}require(rejected,message);
    }
    private static String hash(InputStream input) throws Exception {
        MessageDigest digest=MessageDigest.getInstance("SHA-256");byte[] bytes=new byte[65536];int count;
        while((count=input.read(bytes))!=-1)digest.update(bytes,0,count);
        StringBuilder result=new StringBuilder();for(byte value:digest.digest())result.append(String.format(java.util.Locale.ROOT,"%02x",value&255));
        return result.toString();
    }
    private static void write(File file,String value) throws Exception {
        try(FileOutputStream output=new FileOutputStream(file)){output.write(value.getBytes(StandardCharsets.UTF_8));}
    }
    private static String read(File file) throws Exception {
        try(FileInputStream input=new FileInputStream(file);ByteArrayOutputStream output=new ByteArrayOutputStream()){
            byte[] bytes=new byte[4096];int count;while((count=input.read(bytes))!=-1)output.write(bytes,0,count);
            return new String(output.toByteArray(),StandardCharsets.UTF_8);
        }
    }
    private static void require(boolean value,String message){if(!value)throw new AssertionError(message);}
}
