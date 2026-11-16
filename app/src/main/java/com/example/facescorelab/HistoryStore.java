package com.example.facescorelab;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.AtomicFile;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/** 只保存用户主动选择的记录，图片与索引均在应用内部目录。 */
public class HistoryStore {
    public static class Record {
        public String id,imagePath,referenceMode;
        public long timestamp;
        public double score;
        public JSONObject json;
        public int schemaVersion;
        public JSONObject primaryScore;
        public boolean hasPrimaryScore(){return schemaVersion==PrimaryScore.SCHEMA_VERSION&&PrimaryScore.isSupported(primaryScore);}
        public double displayScore(){return hasPrimaryScore()?primaryScore.optDouble("displayValue"):score;}
        public String scoreLabel(){return hasPrimaryScore()?"CNN 参考评分":"几何参考评分";}
    }
    private final File directory;
    private final AtomicFile index;
    private int unreadableCount;

    public HistoryStore(Context context){
        directory=new File(context.getFilesDir(),"history");
        if(!directory.exists()) directory.mkdirs();
        index=new AtomicFile(new File(directory,"records.json"));
    }
    public synchronized List<Record> readAll() throws Exception {
        List<Record> list=new ArrayList<>();
        unreadableCount=0;
        JSONArray array=readRaw();
        for(int i=0;i<array.length();i++){
            Record record=readRecord(array.optJSONObject(i));
            if(record!=null)list.add(record);else unreadableCount++;
        }
        return list;
    }
    public synchronized int getUnreadableCount(){return unreadableCount;}
    private JSONArray readRaw()throws Exception{
        byte[] bytes;
        // AtomicFile must get the first chance to restore a committed legacy .bak.
        // A missing base alone can mean an interrupted write, not an empty history.
        try(FileInputStream stream=index.openRead(); ByteArrayOutputStream data=new ByteArrayOutputStream()){
            byte[] buffer=new byte[4096]; int n;
            while((n=stream.read(buffer))!=-1) data.write(buffer,0,n);
            bytes=data.toByteArray();
        }catch(FileNotFoundException missing){
            File base=index.getBaseFile(),backup=new File(base.getPath()+".bak");
            // Only a genuinely absent index is the first-run empty state. Keep
            // failed recovery, unreadable paths and other I/O errors visible so a
            // later save cannot silently replace recoverable or inaccessible data.
            if(base.exists()||backup.exists()||!directory.isDirectory()||!directory.canRead())throw missing;
            return new JSONArray();
        }
        return new JSONArray(new String(bytes,StandardCharsets.UTF_8));
    }
    private Record readRecord(JSONObject o){
        if(o==null)return null;
        try{
            Record r=new Record();
            r.id=o.getString("id"); r.imagePath=o.getString("imagePath");
            r.referenceMode=o.optString("referenceMode","general"); r.timestamp=o.getLong("timestamp");
            r.score=o.getDouble("score"); r.json=o;
            r.schemaVersion=o.optInt("schemaVersion",1);r.primaryScore=o.optJSONObject("primaryScore");
            if(!Double.isFinite(r.score)||r.id.isEmpty()||!isInternalImage(r.imagePath)||!new File(r.imagePath).isFile())return null;
            return r;
        }catch(Exception invalid){return null;}
    }
    public synchronized void save(Bitmap bitmap,AnalysisResult result) throws Exception {
        JSONArray previous=readRaw();
        ensureWritable(previous);
        // Unknown/unreadable rows are preserved verbatim. Never apply retention to them silently.
        if(previous.length()>=50)for(int i=0;i<previous.length();i++){
            if(readRecord(previous.optJSONObject(i))==null)throw new IOException("历史中有无法显示的记录，已停止自动清理并保护原文件");
        }
        String id=UUID.randomUUID().toString(); File image=new File(directory,id+".jpg");
        try(FileOutputStream output=new FileOutputStream(image)){
            if(!bitmap.compress(Bitmap.CompressFormat.JPEG,90,output)) throw new IOException("图片保存失败");
        }
        JSONObject o=new JSONObject(); o.put("id",id); o.put("imagePath",image.getAbsolutePath());
        o.put("timestamp",System.currentTimeMillis()); o.put("score",result.score);
        // Keep top-level score and algorithmVersion as geometry for older app versions.
        o.put("schemaVersion",PrimaryScore.SCHEMA_VERSION);o.put("legacyScoreKind","geometry");
        o.put("primaryScoreStatus",result.primaryScoreStatus);
        if(result.primaryScore!=null)o.put("primaryScore",result.primaryScore.toJson());
        o.put("referenceMode",result.referenceMode); o.put("algorithmVersion",result.algorithmVersion);
        if(result.deepAnalysis!=null)o.put("deepAnalysis",result.deepAnalysis);
        if(result.scoringBasis!=null)o.put("scoringBasis",result.scoringBasis);
        JSONArray metrics=new JSONArray();
        for(AnalysisResult.Metric m:result.metrics){
            JSONObject metric=new JSONObject(); metric.put("name",m.name); metric.put("value",m.value);
            metric.put("unit",m.unit); metric.put("description",m.description); metrics.put(metric);
        }
        o.put("metrics",metrics); o.put("reasons",new JSONArray(result.reasons));
        JSONArray array=new JSONArray(); array.put(o);
        for(int i=0;i<Math.min(previous.length(),49);i++) array.put(previous.get(i));
        try{write(array);}catch(Exception e){image.delete();throw e;}
        for(int i=49;i<previous.length();i++){
            Record expired=readRecord(previous.optJSONObject(i));if(expired!=null)deleteImage(expired.imagePath);
        }
    }
    public synchronized void delete(String id) throws Exception {
        JSONArray previous=readRaw(),array=new JSONArray();String path=null;
        ensureWritable(previous);
        for(int i=0;i<previous.length();i++){
            Record record=readRecord(previous.optJSONObject(i));
            if(record!=null&&record.id.equals(id))path=record.imagePath;else array.put(previous.get(i));
        }
        write(array); if(path!=null) deleteImage(path);
    }
    public synchronized void clear() throws Exception {
        JSONArray previous=readRaw();ensureWritable(previous);
        List<Record> list=readAll();
        if(unreadableCount>0)throw new IOException("存在无法显示的历史记录，已保护原文件；请先检查这些记录");
        write(new JSONArray());for(Record r:list) deleteImage(r.imagePath);
    }
    private void ensureWritable(JSONArray records)throws IOException{
        for(int i=0;i<records.length();i++){
            JSONObject record=records.optJSONObject(i);if(record==null||!record.has("schemaVersion"))continue;
            Object value=record.opt("schemaVersion");
            if(!(value instanceof Number)||((Number)value).doubleValue()!=((Number)value).intValue()
                    ||((Number)value).intValue()<1||((Number)value).intValue()>PrimaryScore.SCHEMA_VERSION)
                throw new IOException("历史记录包含无法安全修改的格式版本，已保护原文件，请先更新或检查记录");
        }
    }
    private boolean isInternalImage(String path) throws IOException {
        File f=new File(path);
        return directory.getCanonicalFile().equals(f.getCanonicalFile().getParentFile())&&f.getName().endsWith(".jpg");
    }
    private void deleteImage(String path)throws IOException {if(isInternalImage(path))new File(path).delete();}
    private void write(JSONArray array)throws Exception {
        FileOutputStream output=null;
        try{output=index.startWrite();output.write(array.toString().getBytes(StandardCharsets.UTF_8));index.finishWrite(output);}
        catch(Exception e){if(output!=null)index.failWrite(output);throw e;}
    }
}
