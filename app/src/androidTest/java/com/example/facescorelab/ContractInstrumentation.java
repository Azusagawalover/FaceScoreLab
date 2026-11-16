package com.example.facescorelab;

import android.app.Activity;
import android.app.Instrumentation;
import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PointF;
import android.graphics.Rect;
import android.os.Bundle;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

/** Isolated on-device contracts; the user's history is never opened or modified. */
public final class ContractInstrumentation extends Instrumentation {
    private final JSONArray checks=new JSONArray();
    private int skippedCount;
    private Bundle arguments;
    @Override public void onCreate(Bundle arguments){super.onCreate(arguments);this.arguments=arguments;start();}
    @Override public void onStart(){
        Bundle output=new Bundle();JSONObject report=new JSONObject();int code=Activity.RESULT_OK;
        try{
            scoreContract();historyContract();resizeContract();repairContracts();
            check("visible labels use independent sigmoid rather than softmax",Math.abs(SelfTrainedVisibleAttributes.sigmoid(Math.log(9))-.9)<1e-12
                    &&SelfTrainedVisibleAttributes.sigmoid(1000)==1&&SelfTrainedVisibleAttributes.sigmoid(-1000)==0);
            if(arguments!=null&&"true".equals(arguments.getString("models")))modelContract();
            else skip("optional visible model device checks","models=true was not requested");
            report.put("passed",true);report.put("status",skippedCount==0?"PASS":"PASS_WITH_SKIPS");output.putString("result",skippedCount==0?"PASS":"PASS_WITH_SKIPS");
        }catch(Throwable error){
            code=Activity.RESULT_CANCELED;
            try{report.put("passed",false);report.put("error",android.util.Log.getStackTraceString(error));}catch(Exception ignored){}
            output.putString("result","FAIL: "+error);
        }finally{
            try{
                report.put("checks",checks);report.put("checkCount",checks.length());report.put("skippedCount",skippedCount);
                int passed=0,failed=0;for(int i=0;i<checks.length();i++){
                    JSONObject item=checks.getJSONObject(i);if("PASS".equals(item.optString("status")))passed++;
                    if("FAIL".equals(item.optString("status")))failed++;
                }
                report.put("passedCount",passed);report.put("failedCount",failed);
                File file=new File(getTargetContext().getCacheDir(),"android_contract_checks.json");
                write(file,report.toString(2));output.putString("report",file.getAbsolutePath());
                output.putInt("checkCount",checks.length());
            }catch(Exception error){output.putString("reportError",error.toString());code=Activity.RESULT_CANCELED;}
            finish(code,output);
        }
    }
    private void repairContracts()throws Exception{
        SkinUniformityContractChecks.verifyTranslatedFaceBox();check("skin mask coordinates agree across translated crops",true);
        SkinUniformityContractChecks.verifyNonSkinExclusion();check("skin statistics exclude non-skin pixels",true);
        SkinUniformityContractChecks.verifyInsufficientSkin();check("insufficient skin stays unavailable",true);
        SkinUniformityContractChecks.verifyInvalidCoordinateSpace();check("skin coordinate contract rejects invalid spaces",true);
        Context context=getTargetContext();
        check("first-run history is empty",HistoryBackupContractChecks.verifyMissingIndexIsEmpty(context));
        check("history recovers backup with no base",HistoryBackupContractChecks.verifyBackupOnlyRecovery(context));
        check("history backup replaces interrupted base",HistoryBackupContractChecks.verifyBackupOverridesInterruptedBase(context));
        check("corrupt history is not silently reset",HistoryBackupContractChecks.verifyCorruptionIsNotReset(context));
        check("history IO failure is not an empty history",HistoryBackupContractChecks.verifyReadIoFailureIsNotEmpty(context));
        check("save preserves recovered opaque rows",HistoryBackupContractChecks.verifySavePreservesRecoveredRows(context));
        check("recovered future history stays write-protected",HistoryBackupContractChecks.verifyRecoveredFutureSchemaStaysProtected(context));
        check("visible geometry has known values and transformation invariance",VisibleMeasurementContractChecks.verifyKnownGeometryAndInvariance(context));
        check("missing visible measurements remain unavailable",VisibleMeasurementContractChecks.verifyMissingAndNonFiniteAreUnavailable(context));
        check("eye and lip apertures stay visible measurements",VisibleMeasurementContractChecks.verifyVisibleAperturesStayMeasurements(context));
        check("real blur changes measured texture without rewriting input",VisibleMeasurementContractChecks.verifyRealBlurDecreasesTextureEnergy(context));
        check("visible measurement JSON carries definitions rather than classifications",VisibleMeasurementContractChecks.verifyJsonIsMeasurementOnly(context));
        check("partial robustness failures serialize and preserve input",RobustnessContractChecks.verifyPartialFailureJson());
        check("robustness missing original is not fabricated",RobustnessContractChecks.verifyMissingBaselineIsNotFabricated());
        check("manual scoring profiles have distinct verified assets",ScoreProfileContractChecks.verifyRegistryAndFiles(context));
        check("profile registry rejects mixed assets",ScoreProfileContractChecks.verifyManifestRejectsMixedAssets(context));
        check("manual scoring profile round trip preserves legacy semantics",ScoreProfileContractChecks.verifyPrimaryScoreRoundTripsAndLegacy(context));
        check("history retains manually selected scoring profile",ScoreProfileContractChecks.verifyHistoryRetainsSavedProfile(context));
    }
    private void scoreContract()throws Exception{
        check("mapping endpoints and midpoint",PrimaryScore.display(1)==1&&PrimaryScore.display(3)==5.5&&PrimaryScore.display(5)==10);
        check("display clipping preserves valid range",PrimaryScore.display(-1)==1&&PrimaryScore.display(7)==10);
        PrimaryScore value=sampleScore(3.25);JSONObject json=value.toJson();
        PrimaryScore restored=PrimaryScore.fromJson(json);
        check("score metadata round trip",restored.rawValue==3.25&&restored.displayValue==6.0625
                &&restored.modelSha256.equals(value.modelSha256)&&restored.preprocessVersion.equals(value.preprocessVersion));
        JSONObject future=new JSONObject(json.toString());future.put("schemaVersion",99);
        check("future primary schema not misinterpreted",!PrimaryScore.isSupported(future));
        JSONObject inconsistent=new JSONObject(json.toString());inconsistent.put("displayValue",9.9);
        check("inconsistent stored display is rejected",!PrimaryScore.isSupported(inconsistent));
        boolean rejected=false;try{sampleScore(Double.NaN);}catch(IllegalArgumentException expected){rejected=true;}
        check("nonfinite inference rejected",rejected);
    }
    private void historyContract()throws Exception{
        File root=new File(getTargetContext().getCacheDir(),"contract-history-"+System.nanoTime());
        File files=new File(root,"files");if(!files.mkdirs())throw new IllegalStateException("Cannot create isolated test folder");
        Context isolated=new ContextWrapper(getTargetContext()){@Override public File getFilesDir(){return files;}};
        HistoryStore store=new HistoryStore(isolated);File folder=new File(files,"history");
        File image=new File(folder,"legacy.jpg"),index=new File(folder,"records.json");
        Bitmap bitmap=Bitmap.createBitmap(40,40,Bitmap.Config.ARGB_8888);
        try{
            try(FileOutputStream out=new FileOutputStream(image)){bitmap.compress(Bitmap.CompressFormat.JPEG,90,out);}
            JSONObject legacy=new JSONObject();legacy.put("id","legacy-fixture");legacy.put("imagePath",image.getAbsolutePath());
            legacy.put("timestamp",123456789L);legacy.put("score",6.75);legacy.put("referenceMode","general");
            legacy.put("algorithmVersion","geometry-legacy-v1");legacy.put("unknownField","preserve-me");
            legacy.put("deepAnalysis",new JSONObject().put("cnnRawScore",2.75).put("cnnScore",4.9375).put("unknownNested",true));
            write(index,new JSONArray().put(legacy).toString());String before=read(index);
            List<HistoryStore.Record> old=store.readAll();
            check("legacy read keeps geometry interpretation",old.size()==1&&!old.get(0).hasPrimaryScore()&&old.get(0).displayScore()==6.75);
            check("legacy read does not migrate disk",before.equals(read(index)));
            AnalysisResult result=new AnalysisResult();result.score=7.25;result.algorithmVersion="geometry-current";
            result.scorable=true;result.primaryScore=sampleScore(3.25);result.primaryScoreStatus="success";
            store.save(bitmap,result);List<HistoryStore.Record> records=store.readAll();
            check("new history separates CNN and legacy geometry",records.size()==2&&records.get(0).score==7.25
                    &&records.get(0).hasPrimaryScore()&&records.get(0).displayScore()==6.0625);
            check("old history fields retained on save",records.get(1).json.toString().equals(legacy.toString()));
            check("new history records model provenance",records.get(0).primaryScore.getString("modelSha256").equals(result.primaryScore.modelSha256)
                    &&records.get(0).primaryScore.getString("mappingVersion").equals(PrimaryScore.MAPPING_VERSION));
            JSONObject broken=new JSONObject().put("unknownFutureData","keep this exact object");
            JSONObject missingImage=new JSONObject(legacy.toString());missingImage.put("id","missing-image");missingImage.put("imagePath",new File(folder,"missing.jpg").getAbsolutePath());
            JSONArray withBadRows=new JSONArray(read(index)).put(broken).put("unreadable-row").put(missingImage);
            write(index,withBadRows.toString());String beforeBadRead=read(index);
            check("one bad row does not hide valid history",store.readAll().size()==2&&store.getUnreadableCount()==3);
            check("skipped rows remain unchanged on read",beforeBadRead.equals(read(index)));
            store.save(bitmap,result);JSONArray preserved=new JSONArray(read(index));
            check("saving preserves malformed and missing-image rows",preserved.length()==withBadRows.length()+1
                    &&preserved.getJSONObject(3).toString().equals(broken.toString())
                    &&preserved.getString(4).equals("unreadable-row")
                    &&preserved.getJSONObject(5).toString().equals(missingImage.toString()));
            JSONArray hiddenFuture=new JSONArray(read(index)).put(new JSONObject().put("schemaVersion",99).put("opaque","unreadable future record"));
            write(index,hiddenFuture.toString());String hiddenFutureBytes=read(index);boolean hiddenRejected=false;
            try{store.save(bitmap,result);}catch(java.io.IOException expected){hiddenRejected=true;}
            check("future schema protection includes undisplayable rows",hiddenRejected&&hiddenFutureBytes.equals(read(index)));
            write(index,preserved.toString());
            JSONArray futureRecords=new JSONArray(read(index));futureRecords.getJSONObject(0).put("schemaVersion",99);write(index,futureRecords.toString());
            String futureBytes=read(index);boolean rejected=false;
            try{store.save(bitmap,result);}catch(java.io.IOException expected){rejected=true;}
            check("newer history version is read-only",rejected&&futureBytes.equals(read(index)));
            check("future record remains readable as legacy geometry",store.readAll().get(0).displayScore()==7.25);
            JSONArray retention=new JSONArray().put(legacy);for(int i=1;i<50;i++)retention.put(broken);
            write(index,retention.toString());String retentionBefore=read(index);boolean retained=false;
            try{store.save(bitmap,result);}catch(java.io.IOException expected){retained=true;}
            check("retention never silently drops opaque records",retained&&retentionBefore.equals(read(index)));
        }finally{bitmap.recycle();}
    }
    private void resizeContract()throws Exception{
        int[][] dimensions={{640,480},{73,109},{225,257},{1,19}};
        String[] rgbHashes={"702ed38f74997e35675e0849e813a402be28b6bb66b9db63b1a19baac3fe61b5",
                "1bf0a308a865160eb96927502d495347252dddde862ea2690c6dd759d54ad5f7",
                "2b1b5629812fd385fa940303eaca2519c735a93464b79d460db7975949676a28",
                "f6daf6a9770ce8b9d6ab4aa6a20a149dca6ff762d23fbd216677060646164a8b"};
        String[] tensorHashes={"04d39646dfb9994aeb61faa5a46fa72446255e1c5ef0ad179fbd8d514747cc0a",
                "605dc1abd6c40bf3e253a4a590b5873bdef2cb553ae5ca916d9334ec6095fe36",
                "1a9fd63f6b9ff994bc52dc0f7e381feb9247b30598cdfe76bfc3ff7f6d093967",
                "0b10be55e168720c1abfb7592dbe48bb1f33013019a32e04b62a490e4d837ddf"};
        for(int sample=0;sample<dimensions.length;sample++){
            int width=dimensions[sample][0],height=dimensions[sample][1];int[] pixels=new int[width*height];
            for(int y=0;y<height;y++)for(int x=0;x<width;x++)pixels[y*width+x]=0xff000000
                    |(((x*17+y*29+(x*y)%251)%256)<<16)|(((x*7+y*31+13)%256)<<8)|((x*19+y*3+47)%256);
            int[] resized=VisibleAttributeResize.resize(pixels,width,height,224,224);byte[] rgb=new byte[resized.length*3];
            for(int i=0;i<resized.length;i++){rgb[i*3]=(byte)(resized[i]>>16);rgb[i*3+1]=(byte)(resized[i]>>8);rgb[i*3+2]=(byte)resized[i];}
            float[] tensor=VisibleAttributeResize.imageNetTensor(resized);
            java.nio.ByteBuffer bytes=java.nio.ByteBuffer.allocate(tensor.length*4).order(java.nio.ByteOrder.LITTLE_ENDIAN);bytes.asFloatBuffer().put(tensor);
            check("Pillow 13 golden RGB and tensor "+width+"x"+height,rgbHashes[sample].equals(hash(rgb))&&tensorHashes[sample].equals(hash(bytes.array())));
        }
    }
    private void modelContract()throws Exception{
        check("all three scoring sessions identify the selected trained weights",ScoreProfileContractChecks.verifyModelSessionMetadata(getTargetContext()));
        Bitmap photo;try(InputStream input=getTargetContext().getAssets().open("demo/girl.jpg")){photo=BitmapFactory.decodeStream(input);}
        if(photo==null)throw new IllegalStateException("Demo image unavailable");
        try(LandmarkEngine landmarks=new LandmarkEngine(getTargetContext());OnnxLocalEngine engine=new OnnxLocalEngine(getTargetContext())){
            PointF[] points=landmarks.detect(photo);float[] coordinates=new float[points.length*2];
            for(int i=0;i<points.length;i++){coordinates[2*i]=points[i].x;coordinates[2*i+1]=points[i].y;}
            check("model fixture has full face landmarks",points.length>=478);
            PrimaryScore first=engine.scoreOnly(photo,coordinates,false),second=engine.scoreOnly(photo,coordinates,false);
            check("reused CNN session is deterministic",Math.abs(first.rawValue-second.rawValue)<1e-6);
            OnnxLocalEngine.Result deep=engine.analyze(photo,coordinates,message->{});
            check("deep skin statistics use aligned mask contract",deep.skinUniformity!=null&&deep.skinUniformity.available
                    &&"skin-luminance-mask-aligned-v2".equals(deep.skinUniformity.toJson().optString("methodVersion")));
            check("lightweight and deep CNN paths agree",Math.abs(first.rawValue-deep.rawScore)<1e-6
                    &&deep.primaryScore.modelSha256.equals(first.modelSha256)&&deep.primaryScore.crop.equals(first.crop));
            check("advanced modules retained",deep.mask!=null&&deep.mask.length==512*512&&deep.attributes!=null
                    &&deep.attributes.length==5&&deep.mouthActions!=null&&deep.occlusion.length()>0);
            optionalVisibleContract(photo,coordinates,deep.selfTrainedVisibleAttributes);
            engine.close();boolean rejected=false;
            try{engine.scoreOnly(photo,coordinates,false);}catch(IllegalStateException expected){rejected=true;}
            check("closed engine cannot restart silently",rejected);
        }finally{photo.recycle();}
    }
    private void optionalVisibleContract(Bitmap photo,float[] points,SelfTrainedVisibleAttributes.Result visible)throws Exception{
        check("optional visible model always reports availability",visible!=null);
        if(!visible.available){
            check("absent optional model degrades explicitly", "not_installed".equals(visible.status));
            skip("installed visible model output and session reuse","Optional visible model and/or metadata are not installed");
            skip("visible attribute exact-input CPU parity","Optional model unavailable; no inference was compared");
            skip("trained visible result history round trip","Optional model unavailable; no trained result exists");
            return;
        }
        check("visible model returns three independent sigmoid scores",visible.logits.length==3&&visible.scores.length==3&&visible.thresholds.length==3);
        for(int i=0;i<3;i++)check("visible label sigmoid and validation threshold: "+SelfTrainedVisibleAttributes.LABELS[i],
                Math.abs(visible.scores[i]-SelfTrainedVisibleAttributes.sigmoid(visible.logits[i]))<1e-12
                &&visible.thresholds[i]>0&&visible.thresholds[i]<1&&visible.detected(i)==(visible.scores[i]>=visible.thresholds[i]));
        JSONObject json=visible.toJson();SelfTrainedVisibleAttributes.Result restored=SelfTrainedVisibleAttributes.Result.fromJson(json);
        check("visible result provenance round trip",json.toString().equals(restored.toJson().toString()));
        JSONObject wrongOrder=new JSONObject(json.toString());wrongOrder.getJSONArray("values").getJSONObject(0).put("key","Eyeglasses");
        boolean orderRejected=false;try{SelfTrainedVisibleAttributes.Result.fromJson(wrongOrder);}catch(Exception expected){orderRejected=true;}
        check("visible result rejects label order drift",orderRejected);
        JSONObject wrongDecision=new JSONObject(json.toString());JSONObject row=wrongDecision.getJSONArray("values").getJSONObject(0);row.put("present",!row.getBoolean("present"));
        boolean decisionRejected=false;try{SelfTrainedVisibleAttributes.Result.fromJson(wrongDecision);}catch(Exception expected){decisionRejected=true;}
        check("visible result rejects inconsistent stored decision",decisionRejected);
        try(SelfTrainedVisibleAttributes engine=new SelfTrainedVisibleAttributes(getTargetContext())){
            SelfTrainedVisibleAttributes.Result first=engine.analyze(photo,points),second=engine.analyze(photo,points);
            boolean equal=first.available&&second.available&&first.inputTensorSha256.equals(visible.inputTensorSha256)&&second.modelSha256.equals(visible.modelSha256);
            if(equal)for(int i=0;i<3;i++)equal&=Math.abs(first.logits[i]-second.logits[i])<1e-6&&Math.abs(first.logits[i]-visible.logits[i])<1e-6;
            check("independent visible session reuse is deterministic",equal);
            SelfTrainedVisibleAttributes.Result noFace=engine.analyze(photo,new float[0]);
            check("no face does not become three negative labels",!noFace.available&&"no_face".equals(noFace.status)&&!noFace.toJson().has("values"));
            engine.close();boolean rejected=false;try{engine.analyze(photo,points);}catch(IllegalStateException expected){rejected=true;}
            check("closed visible engine cannot restart silently",rejected);
        }
        write(new File(getTargetContext().getCacheDir(),"visible_attributes_device_result.json"),json.toString(2));
        visibleHistoryContract(photo,json);
        File referenceFile=new File(getTargetContext().getCacheDir(),"visible_attributes_cpu_reference.json");
        if(!referenceFile.isFile()){
            skip("visible attribute exact-input CPU parity","Missing cache/visible_attributes_cpu_reference.json; use exported deep_visible_attributes_input.f32");
            return;
        }
        JSONObject reference=new JSONObject(read(referenceFile));JSONArray expected=reference.getJSONArray("logits");
        double tolerance=reference.getDouble("atol");
        check("CPU visible reference uses exact model and Android tensor",reference.getString("model_sha256").equals(visible.modelSha256)
                &&reference.getString("input_tensor_sha256").equals(visible.inputTensorSha256)&&expected.length()==3
                &&Double.isFinite(tolerance)&&tolerance>0&&tolerance<=1e-4);
        double maximum=0;boolean decisionsMatch=true;
        for(int i=0;i<3;i++){
            double expectedLogit=expected.getDouble(i);if(!Double.isFinite(expectedLogit))throw new AssertionError("Nonfinite CPU reference");
            maximum=Math.max(maximum,Math.abs(visible.logits[i]-expectedLogit));
            decisionsMatch&=(SelfTrainedVisibleAttributes.sigmoid(expectedLogit)>=visible.thresholds[i])==visible.detected(i);
        }
        check("visible attribute exact-input CPU parity",maximum<=tolerance&&decisionsMatch);
        checks.getJSONObject(checks.length()-1).put("maxLogitDifference",maximum).put("atol",tolerance).put("thresholdDecisionsMatch",decisionsMatch);
    }
    private void visibleHistoryContract(Bitmap photo,JSONObject visible)throws Exception{
        File files=new File(getTargetContext().getCacheDir(),"contract-visible-history-"+System.nanoTime()+"/files");
        if(!files.mkdirs())throw new IllegalStateException("Cannot create visible history test folder");
        Context isolated=new ContextWrapper(getTargetContext()){@Override public File getFilesDir(){return files;}};
        HistoryStore history=new HistoryStore(isolated);AnalysisResult result=new AnalysisResult();result.scorable=true;result.score=6.0;
        result.primaryScore=sampleScore(3.25);result.primaryScoreStatus="success";
        result.deepAnalysis=new JSONObject().put("selfTrainedVisibleAttributes",visible).put("visibleAttributes",new JSONArray())
                .put("mouthActions",new JSONObject().put("available",false)).put("unknownModelField","preserve-me");
        history.save(photo,result);List<HistoryStore.Record> records=history.readAll();
        JSONObject deep=records.get(0).json.getJSONObject("deepAnalysis");
        check("trained visible result history round trip",records.size()==1&&deep.getJSONObject("selfTrainedVisibleAttributes").toString().equals(visible.toString())
                &&deep.has("visibleAttributes")&&deep.has("mouthActions")&&"preserve-me".equals(deep.getString("unknownModelField")));
    }
    private PrimaryScore sampleScore(double raw){
        return new PrimaryScore(raw,"contract-test",String.format(java.util.Locale.ROOT,"%064d",0),PrimaryScore.PREPROCESS_VERSION,
                3,new Rect(0,0,40,40),40,40);
    }
    private void check(String name,boolean passed)throws Exception{
        checks.put(new JSONObject().put("name",name).put("status",passed?"PASS":"FAIL").put("passed",passed));
        if(!passed)throw new AssertionError(name);
    }
    private void skip(String name,String reason)throws Exception{
        skippedCount++;checks.put(new JSONObject().put("name",name).put("status","SKIP").put("reason",reason));
    }
    private static void write(File file,String value)throws Exception{
        try(FileOutputStream out=new FileOutputStream(file)){out.write(value.getBytes(StandardCharsets.UTF_8));}
    }
    private static String read(File file)throws Exception{
        try(FileInputStream in=new FileInputStream(file);java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream()){
            byte[] bytes=new byte[4096];int n;while((n=in.read(bytes))!=-1)out.write(bytes,0,n);return out.toString("UTF-8");
        }
    }
    private static String hash(byte[] input)throws Exception{
        byte[] bytes=java.security.MessageDigest.getInstance("SHA-256").digest(input);StringBuilder result=new StringBuilder();
        for(byte value:bytes)result.append(String.format(java.util.Locale.ROOT,"%02x",value&255));return result.toString();
    }
}
