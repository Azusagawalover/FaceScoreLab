package com.example.facescorelab;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

/** History recovery contracts. Every fixture has a separate cache-backed files directory. */
public final class HistoryBackupContractChecks {
    private HistoryBackupContractChecks() {}

    /** A new installation has no index; reading it must not create one. */
    public static boolean verifyMissingIndexIsEmpty(Context context) throws Exception {
        Fixture fixture=fixture(context,"absent");
        require(fixture.store.readAll().isEmpty(),"A genuinely absent index should read as empty");
        require(fixture.store.getUnreadableCount()==0,"A new index has no unreadable rows");
        require(!fixture.base.exists()&&!fixture.backup.exists(),"Reading empty history must not write files");
        return true;
    }

    /** Android's legacy backup is committed history even when the base file is missing. */
    public static boolean verifyBackupOnlyRecovery(Context context) throws Exception {
        Fixture fixture=fixture(context,"backup-only");
        JSONObject legacy=legacy(fixture,"committed-backup",6.75);
        String committed=new JSONArray().put(legacy).toString();
        write(fixture.backup,committed);
        require(!fixture.base.exists(),"Fixture must start with only a backup index");
        List<HistoryStore.Record> records=fixture.store.readAll();
        require(records.size()==1&&"committed-backup".equals(records.get(0).id),"Backup history was not restored");
        require(records.get(0).score==6.75&&!records.get(0).hasPrimaryScore(),"Recovery changed legacy score semantics");
        require(committed.equals(read(fixture.base)),"Recovery must preserve the committed JSON bytes");
        require(!fixture.backup.exists(),"AtomicFile should complete its backup-to-base recovery");
        return true;
    }

    /** An interrupted partial base must not take precedence over a valid committed backup. */
    public static boolean verifyBackupOverridesInterruptedBase(Context context) throws Exception {
        Fixture fixture=fixture(context,"interrupted-base");
        JSONObject legacy=legacy(fixture,"previous-commit",5.5);
        String committed=new JSONArray().put(legacy).toString();
        write(fixture.backup,committed);
        write(fixture.base,"[{\"id\":\"unfinished-write\"");
        List<HistoryStore.Record> records=fixture.store.readAll();
        require(records.size()==1&&"previous-commit".equals(records.get(0).id),"Interrupted base hid the committed backup");
        require(committed.equals(read(fixture.base)),"Backup recovery produced different bytes");
        require(!fixture.backup.exists(),"Recovered backup should no longer be pending");
        return true;
    }

    /** Malformed committed JSON must cause a visible error; saving must not reset it. */
    public static boolean verifyCorruptionIsNotReset(Context context) throws Exception {
        String broken="[{broken committed history";
        Fixture baseFixture=fixture(context,"corrupt-base");
        write(baseFixture.base,broken);
        assertReadFails(baseFixture.store,"Malformed base history became an empty list");
        assertSaveFailsWithoutChangingIndex(baseFixture,broken);

        Fixture backupFixture=fixture(context,"corrupt-backup");
        write(backupFixture.backup,broken);
        assertReadFails(backupFixture.store,"Malformed backup history became an empty list");
        require(broken.equals(read(backupFixture.base)),"Recovered malformed bytes were not preserved");
        assertSaveFailsWithoutChangingIndex(backupFixture,broken);
        return true;
    }

    /** FileNotFoundException can also mean an invalid/unreadable path, not first use. */
    public static boolean verifyReadIoFailureIsNotEmpty(Context context) throws Exception {
        Fixture fixture=fixture(context,"index-is-directory");
        require(fixture.base.mkdir(),"Cannot create an isolated invalid index path");
        boolean rejected=false;
        try{fixture.store.readAll();}catch(IOException expected){rejected=true;}
        require(rejected,"An existing non-file index must report an I/O failure");
        require(fixture.base.isDirectory(),"A failed read must not replace the inaccessible path");
        return true;
    }

    /** Saving directly from a backup-only state must retain all old rows and unknown fields. */
    public static boolean verifySavePreservesRecoveredRows(Context context) throws Exception {
        Fixture fixture=fixture(context,"save-after-recovery");
        JSONObject legacy=legacy(fixture,"old-record",6.25);
        JSONObject opaque=new JSONObject().put("futureExtra","keep this undisplayable row");
        write(fixture.backup,new JSONArray().put(legacy).put(opaque).toString());
        Bitmap bitmap=Bitmap.createBitmap(40,40,Bitmap.Config.ARGB_8888);
        try{fixture.store.save(bitmap,newResult());}finally{bitmap.recycle();}
        JSONArray saved=new JSONArray(read(fixture.base));
        require(saved.length()==3,"Saving must prepend once, not discard recovered history");
        require(saved.getJSONObject(1).toString().equals(legacy.toString()),"Recovered legacy fields were rewritten");
        require(saved.getJSONObject(2).toString().equals(opaque.toString()),"Recovered opaque row was dropped");
        List<HistoryStore.Record> records=fixture.store.readAll();
        require(records.size()==2&&fixture.store.getUnreadableCount()==1,"Readable and opaque rows lost their meanings");
        require(records.get(1).score==6.25&&!records.get(1).hasPrimaryScore(),"Old geometry score was reinterpreted");
        require(!fixture.backup.exists(),"Successful save should finish the AtomicFile transaction");
        return true;
    }

    /** Recovering a future schema must not remove its write protection. */
    public static boolean verifyRecoveredFutureSchemaStaysProtected(Context context) throws Exception {
        Fixture fixture=fixture(context,"future-schema");
        JSONObject future=legacy(fixture,"future-record",7.0).put("schemaVersion",99).put("opaqueFutureField","preserve");
        String committed=new JSONArray().put(future).toString();
        write(fixture.backup,committed);
        Bitmap bitmap=Bitmap.createBitmap(40,40,Bitmap.Config.ARGB_8888);
        boolean rejected=false;
        try{fixture.store.save(bitmap,newResult());}catch(IOException expected){rejected=true;}finally{bitmap.recycle();}
        require(rejected,"A recovered future schema must remain read-only");
        require(committed.equals(read(fixture.base)),"Rejected save changed the recovered future record");
        File[] photos=fixture.folder.listFiles((dir,name)->name.endsWith(".jpg"));
        require(photos!=null&&photos.length==1,"Rejected save must not create another history photo");
        return true;
    }

    private static void assertReadFails(HistoryStore store,String message) throws Exception {
        boolean rejected=false;
        try{store.readAll();}catch(Exception expected){rejected=true;}
        require(rejected,message);
    }

    private static void assertSaveFailsWithoutChangingIndex(Fixture fixture,String committed) throws Exception {
        Bitmap bitmap=Bitmap.createBitmap(40,40,Bitmap.Config.ARGB_8888);
        boolean rejected=false;
        try{fixture.store.save(bitmap,newResult());}catch(Exception expected){rejected=true;}finally{bitmap.recycle();}
        require(rejected,"Saving malformed committed JSON must fail rather than start a fresh history");
        require(committed.equals(read(fixture.base)),"Failed save overwrote malformed committed JSON");
        File[] photos=fixture.folder.listFiles((dir,name)->name.endsWith(".jpg"));
        require(photos!=null&&photos.length==0,"Malformed index must be rejected before creating a photo");
    }

    private static AnalysisResult newResult(){
        AnalysisResult result=new AnalysisResult();
        result.scorable=true;result.score=7.25;result.referenceMode="general";
        result.algorithmVersion="geometry-history-recovery-contract";
        return result;
    }

    private static JSONObject legacy(Fixture fixture,String id,double score) throws Exception {
        File image=new File(fixture.folder,id+".jpg");
        Bitmap bitmap=Bitmap.createBitmap(40,40,Bitmap.Config.ARGB_8888);
        try(FileOutputStream output=new FileOutputStream(image)){
            require(bitmap.compress(Bitmap.CompressFormat.JPEG,90,output),"Cannot encode an isolated history fixture");
        }finally{bitmap.recycle();}
        return new JSONObject().put("id",id).put("imagePath",image.getAbsolutePath()).put("timestamp",123456789L)
                .put("score",score).put("referenceMode","general").put("algorithmVersion","geometry-legacy-contract")
                .put("unknownField","preserve-me").put("deepAnalysis",new JSONObject().put("unknownNested",true));
    }

    private static Fixture fixture(Context context,String name) throws Exception {
        File root=new File(context.getCacheDir(),"contract-history-backup-"+name+"-"+System.nanoTime());
        final File files=new File(root,"files");
        require(files.mkdirs(),"Cannot create isolated history fixture");
        Context isolated=new ContextWrapper(context){@Override public File getFilesDir(){return files;}};
        HistoryStore store=new HistoryStore(isolated);
        return new Fixture(store,new File(files,"history"));
    }

    private static final class Fixture {
        final HistoryStore store;
        final File folder,base,backup;
        Fixture(HistoryStore store,File folder){
            this.store=store;this.folder=folder;
            base=new File(folder,"records.json");backup=new File(folder,"records.json.bak");
        }
    }

    private static void write(File file,String value) throws Exception {
        try(FileOutputStream output=new FileOutputStream(file)){output.write(value.getBytes(StandardCharsets.UTF_8));}
    }
    private static String read(File file) throws Exception {
        try(FileInputStream input=new FileInputStream(file);ByteArrayOutputStream output=new ByteArrayOutputStream()){
            byte[] bytes=new byte[4096];int count;
            while((count=input.read(bytes))!=-1)output.write(bytes,0,count);
            return new String(output.toByteArray(),StandardCharsets.UTF_8);
        }
    }
    private static void require(boolean condition,String message){if(!condition)throw new AssertionError(message);}
}
