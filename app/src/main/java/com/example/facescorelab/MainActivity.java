package com.example.facescorelab;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.content.res.ColorStateList;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** 主界面、图片输入、结果说明和本地记录。模型运算始终放在后台线程。 */
public class MainActivity extends AppCompatActivity {
    private static final int PICK_IMAGE=100, CAPTURE_IMAGE=101, DEEP_ANALYSIS=102;
    private static final int INK=VisualStyle.INK, MUTED=VisualStyle.MUTED;
    private static final int ACCENT=VisualStyle.ACCENT, SOFT=VisualStyle.SOFT;
    private static final int BORDER=VisualStyle.BORDER;
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private LinearLayout content;
    private ScrollView scroll;
    private TextView subtitle;
    private SharedPreferences preferences;
    private HistoryStore history;
    private LandmarkEngine engine;
    private GeometryAnalyzer analyzer;
    private OnnxLocalEngine cnnEngine;
    private GenderClassifier genderEngine;
    private JSONObject genderResult;
    private Bitmap image;
    private PointF[] points=new PointF[0];
    private AnalysisResult result;
    private JSONObject pendingDeep;
    private String currentPath, sourceName="", referenceMode="general", currentTab="home";
    private String inputIdentity="";
    private String status="";
    private Float hairline;
    private boolean busy=false, showPoints=true, calibrating=false;
    private volatile boolean destroyed=false;
    private volatile int generation=0;
    private long elapsedMs=0;
    private int faceCount=0;
    private FaceImageView faceView;
    private final java.util.Map<String,Bitmap> demoThumbnails=new java.util.HashMap<>();

    @Override protected void onCreate(Bundle state){
        super.onCreate(state);
        setContentView(R.layout.activity_main);
        preferences=getSharedPreferences("display_preferences",MODE_PRIVATE);
        boolean profileChoiceInitialized=preferences.getBoolean("manual_score_profiles_v1_initialized",false);
        if(!profileChoiceInitialized)preferences.edit().putBoolean("manual_score_profiles_v1_initialized",true)
                .putString("reference_mode",ScoreProfiles.GENERAL).apply();
        history=new HistoryStore(this);
        content=findViewById(R.id.page_content);
        scroll=findViewById(R.id.content_scroll);
        subtitle=findViewById(R.id.header_subtitle);
        referenceMode=ScoreProfiles.normalize(preferences.getString("reference_mode","general"));
        findViewById(R.id.tab_home).setOnClickListener(v->selectTab("home"));
        findViewById(R.id.tab_history).setOnClickListener(v->selectTab("history"));
        findViewById(R.id.tab_research).setOnClickListener(v->selectTab("research"));
        if(state!=null){
            currentPath=state.getString("image_path");
            inputIdentity=state.getString("input_identity","");
            sourceName=state.getString("source_name","");
            if(profileChoiceInitialized)referenceMode=ScoreProfiles.normalize(state.getString("reference_mode",referenceMode));
            showPoints=state.getBoolean("show_points",true);
            calibrating=state.getBoolean("calibrating",false);
            if(state.containsKey("hairline"))hairline=state.getFloat("hairline");
            currentTab=state.getString("tab","home");
            try{String saved=state.getString("deep_json");if(saved!=null)pendingDeep=new JSONObject(saved);}catch(Exception ignored){}
        }
        selectTab(currentTab);
        if(currentPath!=null&&new File(currentPath).isFile())loadFile(currentPath,sourceName,false);
    }

    private void selectTab(String tab){
        currentTab=tab;
        setTabStyle(R.id.tab_home,"home".equals(tab));
        setTabStyle(R.id.tab_history,"history".equals(tab));
        setTabStyle(R.id.tab_research,"research".equals(tab));
        faceView=null;
        content.removeAllViews();
        if("history".equals(tab)){subtitle.setText("保存在这台设备上的分析记录");showHistory();}
        else if("research".equals(tab)){subtitle.setText("方法、实验与文献");showResearch();}
        else{currentTab="home";subtitle.setText("面部分析与评分实验室");showHome();}
        scroll.scrollTo(0,0);
    }
    private void setTabStyle(int id,boolean selected){
        TextView view=findViewById(id);view.setTextColor(selected?ACCENT:MUTED);
        view.setTypeface(Typeface.DEFAULT,selected?Typeface.BOLD:Typeface.NORMAL);
        for(android.graphics.drawable.Drawable icon:view.getCompoundDrawablesRelative())if(icon!=null)icon.mutate().setTint(selected?ACCENT:MUTED);
        view.setSelected(selected);
    }
    private void refreshHome(){if("home".equals(currentTab)){int y=scroll.getScrollY();content.removeAllViews();showHome();scroll.post(()->scroll.scrollTo(0,y));}}

    private void showHome(){
        if(image==null&&!busy)showHero();
        LinearLayout actions=card();
        label(actions,image==null?"开始分析":"选择另一张照片",16,INK,true);
        LinearLayout sources=row();marginTop(sources,12);actions.addView(sources);
        addRowButton(sources,"相册",R.id.gallery_button,true,v->openGallery()).setContentDescription("从相册选择照片");
        addRowButton(sources,"拍照",R.id.camera_button,false,v->openCamera(false)).setContentDescription("拍照并分析");
        addRowButton(sources,"实时",R.id.live_button,false,v->openCamera(true)).setContentDescription("打开实时相机分析");
        LinearLayout demos=row();marginTop(demos,10);actions.addView(demos);
        addDemoTile(demos,"portrait_a.jpg","示例 A",R.id.demo_one_button);
        addDemoTile(demos,"portrait_b.jpg","示例 B",R.id.demo_two_button);
        for(int i=0;i<sources.getChildCount();i++)sources.getChildAt(i).setEnabled(!busy);
        for(int i=0;i<demos.getChildCount();i++){demos.getChildAt(i).setEnabled(!busy);demos.getChildAt(i).setAlpha(busy?.5f:1f);}
        if(busy){
            LinearLayout loading=card();
            TextView text=label(loading,status.isEmpty()?"正在分析，请稍候…":status,14,ACCENT,true);
            text.setId(R.id.analysis_status);
            ProgressBar bar=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);bar.setIndeterminate(true);
            LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,dp(5));p.topMargin=dp(12);loading.addView(bar,p);
        }
        if(image!=null){
            LinearLayout photo=card();photo.setPadding(dp(10),dp(10),dp(10),dp(12));
            faceView=new FaceImageView(this);faceView.setId(R.id.face_image);faceView.setImage(image,points);
            faceView.setShowPoints(showPoints);faceView.setHairline(calibrating?hairline:null);
            // 按图像比例调整预览高度。
            photo.addView(faceView,new LinearLayout.LayoutParams(-1,previewHeight(image)));
            LinearLayout imageInfo=row();imageInfo.setGravity(Gravity.CENTER_VERTICAL);marginTop(imageInfo,8);photo.addView(imageInfo);
            TextView caption=text(sourceName.isEmpty()?"当前图片":sourceName,12,MUTED,false);
            imageInfo.addView(caption,new LinearLayout.LayoutParams(0,-2,1));
            Switch toggle=new Switch(this);toggle.setId(R.id.points_toggle);toggle.setText("关键点  ");toggle.setTextSize(12);toggle.setTextColor(MUTED);toggle.setChecked(showPoints);
            toggle.setOnCheckedChangeListener((button,checked)->{showPoints=checked;if(faceView!=null)faceView.setShowPoints(checked);});imageInfo.addView(toggle);
            if(!busy&&!status.isEmpty()){
                TextView statusText=label(photo,status,12,result!=null&&result.scorable?MUTED:VisualStyle.WARN,false);
                statusText.setId(R.id.analysis_status);marginTop(statusText,8);
            }
            if(!busy&&result==null){
                Button clear=button("清除照片",R.id.reset_button,false);marginTop(clear,10);photo.addView(clear);clear.setOnClickListener(v->clearCurrent());
            }
        }
        if(result!=null&&!busy)showAnalysis();
        if(image==null&&!busy){
            showReferenceCard();
            LinearLayout note=card();label(note,"你的照片由你掌握",15,INK,true);
            label(note,"分析在设备上完成。点击“保存记录”后，才会加入历史。",13,MUTED,false);
            label(note,"分数反映特定样本中的主观评分偏好，用于课程研究。",12,MUTED,false);
        }
    }

    private void showHero(){
        LinearLayout hero=card();
        // 首页介绍卡使用统一渐变背景。
        hero.setBackground(VisualStyle.heroBackground(this));
        LinearLayout top=row();top.setGravity(Gravity.CENTER_VERTICAL);hero.addView(top);
        LinearLayout words=new LinearLayout(this);words.setOrientation(LinearLayout.VERTICAL);top.addView(words,new LinearLayout.LayoutParams(0,-2,1));
        label(words,"LOCAL FACE ANALYSIS",10,VisualStyle.CORAL_DEEP,true);
        TextView title=label(words,"看见面部结构",22,INK,true);marginTop(title,8);
        ImageView illustration=new ImageView(this);illustration.setImageResource(R.drawable.illustration_face_mesh);
        illustration.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        LinearLayout.LayoutParams illustrationParams=new LinearLayout.LayoutParams(dp(94),dp(104));illustrationParams.leftMargin=dp(8);top.addView(illustration,illustrationParams);
        // 说明文字独立占用整行宽度。
        TextView description=label(hero,"关键点、面部比例与图像评分依据，全部在本机完成。",12,MUTED,false);marginTop(description,10);
        LinearLayout badges=row();marginTop(badges,12);hero.addView(badges);
        badge(badges,"照片分析");badge(badges,"实时检测");badge(badges,"本地保存");
    }

    private void addDemoTile(LinearLayout parent,String asset,String title,int id){
        LinearLayout tile=row();tile.setGravity(Gravity.CENTER_VERTICAL);tile.setPadding(dp(8),dp(9),dp(8),dp(9));
        tile.setId(id);tile.setMinimumHeight(dp(62));tile.setClickable(true);tile.setFocusable(true);
        tile.setContentDescription("加载"+title+"人像进行分析");
        tile.setBackground(new RippleDrawable(ColorStateList.valueOf(0x164255c5),VisualStyle.rounded(this,Color.WHITE,12,true),VisualStyle.rounded(this,Color.WHITE,12,false)));
        ImageView thumb=new ImageView(this);thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);thumb.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        thumb.setBackground(VisualStyle.rounded(this,SOFT,8,false));thumb.setClipToOutline(true);
        Bitmap thumbnail=demoThumbnails.get(asset);
        if(thumbnail==null)try(InputStream stream=getAssets().open("demo/"+asset)){
            BitmapFactory.Options options=new BitmapFactory.Options();options.inSampleSize=4;thumbnail=BitmapFactory.decodeStream(stream,null,options);
            if(thumbnail!=null)demoThumbnails.put(asset,thumbnail);
        }catch(IOException ignored){}
        if(thumbnail!=null)thumb.setImageBitmap(thumbnail);
        tile.addView(thumb,new LinearLayout.LayoutParams(dp(36),dp(40)));
        LinearLayout words=new LinearLayout(this);words.setOrientation(LinearLayout.VERTICAL);LinearLayout.LayoutParams wordParams=new LinearLayout.LayoutParams(0,-2,1);wordParams.leftMargin=dp(8);tile.addView(words,wordParams);
        label(words,title,12,INK,true);TextView subtitle=label(words,"正面人像",10,MUTED,false);marginTop(subtitle,3);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(0,-2,1);if(parent.getChildCount()>0)params.leftMargin=dp(8);parent.addView(tile,params);
        tile.setOnClickListener(v->loadDemo(asset,title+" · 人像照片"));
    }

    /** 预览高度：按图片宽高比推算，并限制在合理区间内，仅影响显示布局。 */
    private int previewHeight(Bitmap image){
        int screenWidth=getResources().getDisplayMetrics().widthPixels;
        int availableWidth=screenWidth-dp(16)*2-dp(10)*2;
        if(availableWidth<=0||image==null||image.getWidth()<=0)return dp(250);
        float aspect=image.getHeight()/(float)image.getWidth();
        int height=Math.round(availableWidth*aspect);
        return Math.max(dp(180),Math.min(dp(330),height));
    }

    private void showAnalysis(){
        LinearLayout score=card();
        // 评分卡背景使用统一样式。
        score.setBackground(VisualStyle.scoreCardBackground(this));
        LinearLayout scoreRow=row();scoreRow.setGravity(Gravity.CENTER_VERTICAL);score.addView(scoreRow);
        LinearLayout numberBox=new LinearLayout(this);numberBox.setOrientation(LinearLayout.VERTICAL);
        scoreRow.addView(numberBox,new LinearLayout.LayoutParams(0,-2,1));
        boolean hasCnn=result.primaryScore!=null;
        label(numberBox,hasCnn?ScoreProfiles.label(result.primaryScore.referenceProfile):result.scorable?"CNN 暂无评分":"需要调整照片",13,MUTED,false);
        // 分数采用中性色显示。
        TextView number=label(numberBox,hasCnn?String.format(Locale.CHINA,"%.1f",result.primaryScore.displayValue):"—",43,INK,true);
        number.setId(R.id.score_text);
        if(hasCnn)label(numberBox,"/ 10 分",13,MUTED,false);
        if(hasCnn)scoreTrack(numberBox,result.primaryScore.displayValue,10);
        TextView mode=text("手动参照："+modeName(referenceMode)+"\n"+points.length+" 个关键点",12,MUTED,true);
        mode.setGravity(Gravity.END);scoreRow.addView(mode);
        if(result.scorable){
            if(hasCnn){
                label(score,String.format(Locale.CHINA,"原始输出 %.3f / 5 · CNN %d ms",result.primaryScore.rawValue,result.primaryScore.durationMillis),12,MUTED,false);
                label(score,"当前使用"+ScoreProfiles.label(result.primaryScore.referenceProfile)+"；原始 1～5 分按固定端点映射显示为 1～10 分。",12,MUTED,false);
                if(!ScoreProfiles.GENERAL.equals(referenceMode))label(score,profileNotice(referenceMode),12,MUTED,false);
            }else label(score,"multiple_faces".equals(result.primaryScoreStatus)?"请使用单人照片取得 CNN 评分。":"图像评分暂不可用，可重试；下方保留几何对照。",12,MUTED,false);
            TextView details=label(score,String.format(Locale.CHINA,"几何对照评分 %.1f / 10 · 依据面部比例计算",result.score),14,MUTED,false);marginTop(details,9);
            if(result.qualityMessage!=null&&!result.qualityMessage.isEmpty()){
                TextView quality=label(score,result.qualityMessage,12,MUTED,false);marginTop(quality,7);
            }
        }else{
            label(score,result.qualityMessage==null?"请使用清晰、正面的单人照片。":result.qualityMessage,14,VisualStyle.WARN,false);
        }
        showReferenceCard();
        showDeepCard();
        showScoringBasis();
        if(result.reasons!=null&&!result.reasons.isEmpty()){
            LinearLayout explanation=card();label(explanation,"几何对照依据",16,INK,true);
            for(String reason:result.reasons){TextView view=label(explanation,"•  "+reason,13,MUTED,false);marginTop(view,8);}
        }
        LinearLayout geometry=card();geometry.setId(R.id.metrics_container);label(geometry,"面部几何分析",16,INK,true);
        if(result.metrics!=null){
            for(AnalysisResult.Metric metric:result.metrics){
                LinearLayout metricRow=row();metricRow.setGravity(Gravity.CENTER_VERTICAL);marginTop(metricRow,13);geometry.addView(metricRow);
                TextView name=text(metric.name,13,INK,false);metricRow.addView(name,new LinearLayout.LayoutParams(0,-2,1));
                TextView value=text(String.format(Locale.CHINA,"%.3f %s",metric.value,metric.unit==null?"":metric.unit),13,INK,true);metricRow.addView(value);
                if(metric.description!=null&&!metric.description.isEmpty())label(geometry,metric.description,11,MUTED,false);
            }
        }
        LinearLayout observation=card();label(observation,"表情动作观察",16,INK,true);
        label(observation,String.format(Locale.CHINA,"眼睛开合比  %.3f     嘴部开合比  %.3f",result.eyeAspect,result.mouthAspect),13,MUTED,false);
        label(observation,"实时相机可连续观察可见动作变化。",12,MUTED,false);
        showCalibration();
        LinearLayout controls=row();content.addView(controls,new LinearLayout.LayoutParams(-1,-2));
        Button save=addRowButton(controls,"保存记录",R.id.save_button,true,v->saveRecord());save.setEnabled(result.scorable&&!busy);
        addRowButton(controls,"清除当前照片",R.id.reset_button,false,v->clearCurrent());
        TextView foot=text("主观评分仅用于课程研究。不同拍摄角度和光照会影响测量。",11,MUTED,false);
        LinearLayout.LayoutParams fp=new LinearLayout.LayoutParams(-1,-2);fp.topMargin=dp(12);fp.bottomMargin=dp(12);content.addView(foot,fp);
    }

    /** 展示参照组几何依据、皮肤亮度统计和表情开合测量。 */
    private void showScoringBasis(){
        JSONObject basis=result.scoringBasis;
        LinearLayout card=card();card.setId(R.id.scoring_basis_card);
        label(card,"打分依据",16,INK,true);
        label(card,"以下依据均由本机图像测量得到，用于说明分数的来源；不代表客观美丑或人的价值。",12,MUTED,false);
        if(basis==null)return;
        String mode=basis.optString("referenceMode","general");
        card.addView(divider());
        label(card,"分性别打分依据 · "+ScoringBasis.modeLabel(mode),14,INK,true);
        label(card,basis.optString("focusRationale",""),12,MUTED,false);
        showGenderReference(card);
        JSONArray focus=basis.optJSONArray("focus");
        if(focus!=null&&focus.length()>0){
            for(int i=0;i<focus.length();i++){
                JSONObject item=focus.optJSONObject(i);if(item==null)continue;
                String estimate=item.optString("estimate","");
                metricRow(card,
                        item.optString("name"),
                        String.format(Locale.CHINA,"%.3f %s",item.optDouble("value"),item.optString("unit")).trim(),
                        item.optString("description",""),
                        estimate);
            }
        }
        card.addView(divider());
        String skinText=basis.optString("skinUniformityText","");
        JSONObject deepSkin=result.deepAnalysis==null?null:result.deepAnalysis.optJSONObject("skinUniformity");
        if(deepSkin!=null&&deepSkin.optBoolean("available")){
            skinText=String.format(Locale.CHINA,"%.1f / 100（皮肤区亮度标准差 %.3f，皮肤占比 %.1f%%）",
                    deepSkin.optDouble("uniformity"),deepSkin.optDouble("luminanceStdDev"),deepSkin.optDouble("coverage")*100);
        }
        metricRow(card,"肤色均匀度",
                skinText.isEmpty()?"待分割":skinText,
                skinText.isEmpty()?"在深度分析页运行五官分割后，可查看肤色均匀度。":"只在分割出的皮肤区域统计亮度离散度，属图像统计量，不判断肤色深浅好坏，也不是健康指标。",
                null);
        card.addView(divider());
        String nature=basis.optString("expressionNaturalness","");
        label(card,"表情自然度",14,INK,true);
        label(card,nature.isEmpty()?"本次未取得表情开合测量。":nature,13,MUTED,false);
        noteBox(card,basis.optString("disclaimer",""));
        Button switchMode=button("切换参照组",R.id.scoring_basis_mode_button,false);
        switchMode.setOnClickListener(v->{
            String next=ScoringBasis.MODE_GENERAL.equals(referenceMode)?ScoringBasis.MODE_MALE
                    :ScoringBasis.MODE_MALE.equals(referenceMode)?ScoringBasis.MODE_FEMALE:ScoringBasis.MODE_GENERAL;
            chooseScoringProfile(next);
        });
        marginTop(switchMode,10);card.addView(switchMode);
        label(card,"参照组由你手动选择，App 不从照片推断性别。",11,MUTED,false);
    }
    /** 显示可选性别参考，参照组由用户确认后切换。 */
    private void showGenderReference(LinearLayout card){
        label(card,"性别参考（可选）",14,INK,true);
        if(genderResult==null){
            label(card,"用本项目训练的人脸性别模型给出一个参考判断；不使用该结果也不影响打分。",12,MUTED,false);
            Button run=button(busy?"分析中…":"运行性别参考",R.id.gender_run_button,false);
            run.setEnabled(!busy);
            run.setOnClickListener(v->runGenderReference());
            marginTop(run,8);card.addView(run);
            return;
        }
        GenderClassifier.Result parsed=null;
        try{parsed=GenderClassifier.Result.fromJson(genderResult);}catch(Exception ignored){}
        TextView text=label(card,parsed==null?"性别参考结果无法读取。":parsed.describe(),13,INK,false);
        text.setId(R.id.gender_result_text);
        if(parsed==null||!parsed.available)return;
        String suggested=parsed.predictedMale()?"male":"female";
        if(parsed.uncertain()){
            label(card,"结果接近判定阈值，建议保持当前参照组。",11,VisualStyle.WARN,false);
        }else if(!suggested.equals(referenceMode)){
            Button apply=button("按模型建议改用"+ScoringBasis.modeLabel(suggested),R.id.gender_apply_button,false);
            apply.setEnabled(!busy);
            apply.setOnClickListener(v->{
                if(busy)return;
                chooseScoringProfile(suggested);
            });
            marginTop(apply,8);card.addView(apply);
            label(card,"这是你主动确认的切换，App 不会自动改变参照组。",11,MUTED,false);
        }else label(card,"与当前参照组一致，无需切换。",11,MUTED,false);
    }

    private void runGenderReference(){
        if(busy||image==null||points.length==0)return;
        final Bitmap selected=image;
        final float[] coordinates=new float[points.length*2];
        for(int i=0;i<points.length;i++){coordinates[2*i]=points[i].x;coordinates[2*i+1]=points[i].y;}
        final int token=begin("正在运行性别参考模型…");
        worker.execute(()->{
            try{
                if(destroyed||token!=generation)return;
                if(genderEngine==null)genderEngine=new GenderClassifier(this);
                JSONObject json=genderEngine.analyze(selected,coordinates).toJson();
                runOnUiThread(()->{
                    if(destroyed||token!=generation)return;
                    busy=false;genderResult=json;status="已取得性别参考";refreshHome();
                });
            }catch(Exception error){runOnUiThread(()->failed(token,"性别参考未完成，可稍后重试。",error));}
        });
    }

    private void showDeepCard(){        LinearLayout deep=card();label(deep,"深度模型对比",17,INK,true);
        label(deep,"对比自训练 CNN、FPEM 与几何评分，查看可见属性、五官分割和区域评分依据。",13,MUTED,false);
        if(result.deepAnalysis!=null){
            label(deep,String.format(Locale.CHINA,"CNN 参考评分 %.1f / 10",result.deepAnalysis.optDouble("cnnScore")),21,INK,true);
            if(result.deepAnalysis.has("fpemScore"))label(deep,String.format(Locale.CHINA,"FPEM 参考评分 %.1f / 10",result.deepAnalysis.optDouble("fpemScore")),18,INK,true);
            label(deep,"已附加 CNN 分数与分割统计，可一起保存到历史记录。",12,MUTED,false);
        }
        Button open=button(result.deepAnalysis==null?"CNN / 属性 / 五官分割":"再次查看深度分析",R.id.deep_analysis_button,true);
        open.setEnabled(result.scorable);open.setOnClickListener(v->openDeepAnalysis());marginTop(open,10);deep.addView(open);
    }
    private void openDeepAnalysis(){
        if(busy||result==null||!result.scorable||image==null||currentPath==null)return;
        float[] coordinates=new float[points.length*2];
        for(int i=0;i<points.length;i++){coordinates[2*i]=points[i].x;coordinates[2*i+1]=points[i].y;}
        Intent intent=new Intent(this,DeepAnalysisActivity.class);
        intent.putExtra("reference_mode",referenceMode);
        intent.putExtra("image_path",currentPath);intent.putExtra("landmarks",coordinates);intent.putExtra("geometry_score",result.score);
        intent.putExtra("source_generation",generation);
        intent.putExtra("source_identity",inputIdentity);
        if(result.deepAnalysis!=null)intent.putExtra("existing_deep",result.deepAnalysis.toString());
        intent.putExtra("image_width",image.getWidth());intent.putExtra("image_height",image.getHeight());startActivityForResult(intent,DEEP_ANALYSIS);
    }
    private void showReferenceCard(){
        LinearLayout reference=card();label(reference,"参考模式",15,INK,true);
        label(reference,"手动选择通用、男性或女性参照，切换对应的独立 CNN 评分模型；不从照片自动决定。",12,MUTED,false);
        label(reference,profileNotice(referenceMode),11,MUTED,false);
        LinearLayout modes=row();marginTop(modes,10);reference.addView(modes);
        // 等宽按钮使用短标签，推荐说明另行展示。
        
        String[] values={"general","male","female"}, labels={"通用","男性实验","女性实验"};
        int[] ids={R.id.general_button,R.id.male_button,R.id.female_button};
        for(int i=0;i<values.length;i++){
            final String value=values[i];Button button=addRowButton(modes,labels[i],ids[i],referenceMode.equals(value),v->{
                chooseScoringProfile(value);
            });button.setEnabled(!busy);
        }
    }
    private void chooseScoringProfile(String value){
        if(busy)return;
        String selected=ScoreProfiles.normalize(value);
        if(selected.equals(referenceMode)){refreshHome();return;}
        referenceMode=selected;preferences.edit().putString("reference_mode",selected).apply();
        // 复用照片和关键点，评分及遮挡结果须与当前模型匹配。
        
        pendingDeep=null;result=null;
        if(image!=null)analyzeCurrent(points.length>0);else refreshHome();
    }
    private static boolean deepMatchesProfile(JSONObject deep,String profile){
        return deep!=null&&PrimaryScore.matchesProfile(deep.optJSONObject("primaryScore"),profile);
    }
    private String profileNotice(String profile){
        try{return ScoreProfiles.resolve(new JSONObject(readAsset("deep_models.json")),profile).notice;}
        catch(Exception unavailable){return ScoreProfiles.GENERAL.equals(profile)?"通用模型保持默认推荐。":"专用实验模型，尚未证实更准；若模型不可用请切回通用。";}
    }
    private void showCalibration(){
        LinearLayout card=card();label(card,"三庭测量校准",16,INK,true);
        String calibrationText=hairline==null?"真实发际线需要手动指定。未校准时，相关比例只作近似参考。"
                :result!=null&&result.hairlineCalibrated?"已指定发际线位置，可继续微调。":"请把校准线移到眉部上方的实际发际线位置，当前仍按估算显示。";
        label(card,calibrationText,12,MUTED,false);
        Button button=button(calibrating?"关闭校准线":"调整发际线",R.id.calibration_button,false);marginTop(button,10);card.addView(button);
        button.setOnClickListener(v->{
            calibrating=!calibrating;
            if(calibrating&&hairline==null){
                hairline=points.length>10?Math.max(0f,Math.min(.9f,points[10].y/image.getHeight()-.04f)):.2f;
                analyzeCurrent(true);
            }else refreshHome();
        });
        if(calibrating){
            label(card,"移动滑块，让黄色线与照片中的发际线重合。",12,MUTED,false);
            SeekBar seek=new SeekBar(this);seek.setId(R.id.hairline_seek);seek.setMax(1000);seek.setProgress(Math.round((hairline==null?.2f:hairline)*1000));card.addView(seek,new LinearLayout.LayoutParams(-1,dp(48)));
            seek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
                @Override public void onProgressChanged(SeekBar bar,int progress,boolean fromUser){if(fromUser){hairline=progress/1000f;if(faceView!=null)faceView.setHairline(hairline);}}
                @Override public void onStartTrackingTouch(SeekBar bar){}
                @Override public void onStopTrackingTouch(SeekBar bar){analyzeCurrent(true);}
            });
        }
    }

    private void openGallery(){
        if(busy)return;
        Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("image/*").addCategory(Intent.CATEGORY_OPENABLE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try{startActivityForResult(intent,PICK_IMAGE);}catch(ActivityNotFoundException e){toast("这台设备没有可用的图片选择器。");}
    }
    private void openCamera(boolean live){
        if(busy)return;
        try{Intent intent=new Intent(this,CameraActivity.class);intent.putExtra("live_mode",live);intent.putExtra("reference_mode",referenceMode);startActivityForResult(intent,CAPTURE_IMAGE);}
        catch(Exception e){error("相机暂时无法启动",e);}
    }
    @Override protected void onActivityResult(int request,int code,Intent data){
        super.onActivityResult(request,code,data);
        if(code!=Activity.RESULT_OK){if(request==PICK_IMAGE||request==CAPTURE_IMAGE)toast("已取消选择");return;}
        if(request==DEEP_ANALYSIS&&data!=null){
            // 输入标识随页面重建恢复，请求代号仅用于当前实例。
            if(currentPath==null||!currentPath.equals(data.getStringExtra("source_path"))||inputIdentity.isEmpty()
                    ||!inputIdentity.equals(data.getStringExtra("source_identity")))return;
            try{pendingDeep=new JSONObject(data.getStringExtra("deep_json"));
                if(!deepMatchesProfile(pendingDeep,referenceMode)){pendingDeep=null;toast("该结果来自另一评分参照组，请重新分析");return;}
                if(result!=null){
                    result.deepAnalysis=pendingDeep;
                    JSONObject primary=pendingDeep.optJSONObject("primaryScore");
                    if(faceCount==1&&PrimaryScore.isSupported(primary)){result.primaryScore=PrimaryScore.fromJson(primary);result.primaryScoreStatus="success";}
                    pendingDeep=null;
                }refreshHome();toast("深度分析已附加，点击保存记录可保留结果");}
            catch(Exception e){error("深度分析结果读取失败",e);}return;
        }
        if(request==PICK_IMAGE&&data!=null&&data.getData()!=null){
            Uri uri=data.getData();loadUri(uri);
        }else if(request==CAPTURE_IMAGE&&data!=null){
            String path=data.getStringExtra("image_path");
            if(path!=null&&new File(path).isFile())loadFile(path,"相机照片",true);else toast("没有收到可读取的照片，请重试。");
        }
    }
    private void loadUri(Uri uri){
        if(busy)return;
        final int token=begin("正在读取照片…");
        worker.execute(()->{
            try{
                Bitmap decoded=decodeUri(uri);String path=cacheBitmap(decoded);
                Bitmap canonical=decodeFile(path);if(canonical!=decoded)decoded.recycle();
                runOnUiThread(()->acceptImage(token,canonical,path,"相册图片",true));
            }catch(Exception e){runOnUiThread(()->failed(token,"照片读取失败，请选择其他图片。",e));}
        });
    }
    private void loadDemo(String asset,String label){
        if(busy)return;
        final int token=begin("正在读取官方示例…");
        worker.execute(()->{
            try(InputStream stream=getAssets().open("demo/"+asset)){
                Bitmap decoded=BitmapFactory.decodeStream(stream);if(decoded==null)throw new IOException("示例图片无效");
                String path=cacheBitmap(decoded);Bitmap canonical=decodeFile(path);if(canonical!=decoded)decoded.recycle();
                runOnUiThread(()->acceptImage(token,canonical,path,label,true));
            }catch(Exception e){runOnUiThread(()->failed(token,"示例读取失败，请尝试相册选图。",e));}
        });
    }
    private void loadFile(String path,String label,boolean reset){
        if(busy)return;
        final int token=begin("正在读取照片…");
        worker.execute(()->{
            try{
                Bitmap decoded=decodeFile(path);runOnUiThread(()->acceptImage(token,decoded,path,label,reset));
            }catch(Exception e){runOnUiThread(()->failed(token,"照片无法读取。",e));}
        });
    }
    private int begin(String message){busy=true;status=message;int token=++generation;refreshHome();return token;}
    private void acceptImage(int token,Bitmap decoded,String path,String label,boolean reset){
        if(destroyed||token!=generation){decoded.recycle();return;}
        image=decoded;currentPath=path;sourceName=label;points=new PointF[0];result=null;
        if(reset||inputIdentity.isEmpty())inputIdentity=java.util.UUID.randomUUID().toString();
        if(reset){hairline=null;calibrating=false;pendingDeep=null;genderResult=null;}
        busy=false;
        if(reset)selectTab("home");else refreshHome();
        analyzeCurrent(false);
    }
    private void analyzeCurrent(boolean reuse){
        if(image==null||busy)return;
        final Bitmap selected=image;final PointF[] cached=points;final String mode=referenceMode;final Float line=hairline;
        final JSONObject previousDeep=reuse&&result!=null&&deepMatchesProfile(result.deepAnalysis,mode)?result.deepAnalysis:null;
        final PrimaryScore previousPrimary=reuse&&result!=null&&result.primaryScore!=null
                &&mode.equals(result.primaryScore.referenceProfile)?result.primaryScore:null;
        final int token=begin(reuse?"正在更新测量结果…":"正在检测人脸与关键点…");
        worker.execute(()->{
            long start=System.nanoTime();
            try{
                if(destroyed||token!=generation)return;
                if(analyzer==null)analyzer=new GeometryAnalyzer(this);
                PointF[] detected;
                if(reuse&&cached.length>0)detected=cached;
                else{if(engine==null)engine=new LandmarkEngine(this);detected=engine.detect(selected);}
                if(detected==null)detected=new PointF[0];
                final PointF[] output=detected;
                final int detectedCount=reuse?faceCount:(engine==null?0:engine.lastFaceCount);
                final AnalysisResult measured=output.length==0?null:analyzer.analyze(output,selected.getWidth(),selected.getHeight(),mode,line);
                if(measured!=null){
                    measured.deepAnalysis=previousDeep;
                    measured.primaryScoreStatus=detectedCount>1?"multiple_faces":"quality_rejected";
                    if(measured.scorable&&detectedCount==1&&!destroyed&&token==generation){
                        try{
                            if(previousPrimary!=null)measured.primaryScore=previousPrimary;
                            else{
                                if(cnnEngine!=null&&!mode.equals(cnnEngine.getReferenceProfile())){cnnEngine.close();cnnEngine=null;}
                                if(cnnEngine==null)cnnEngine=new OnnxLocalEngine(this,mode);
                                float[] coordinates=new float[output.length*2];
                                for(int i=0;i<output.length;i++){coordinates[2*i]=output[i].x;coordinates[2*i+1]=output[i].y;}
                                measured.primaryScore=cnnEngine.scoreOnly(selected,coordinates);
                            }
                            measured.primaryScoreStatus="success";
                        }catch(Exception cnnError){measured.primaryScoreStatus="unavailable";android.util.Log.w("FaceScoreLab","Home CNN unavailable",cnnError);}
                    }
                    if((getApplicationInfo().flags&android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE)!=0&&!destroyed&&token==generation){
                        try(FileOutputStream log=new FileOutputStream(new File(getCacheDir(),"home_last.json"))){
                            JSONObject snapshot=new JSONObject();snapshot.put("schemaVersion",PrimaryScore.SCHEMA_VERSION);
                            snapshot.put("score",measured.score);snapshot.put("legacyScoreKind","geometry");snapshot.put("primaryScoreStatus",measured.primaryScoreStatus);
                            if(measured.primaryScore!=null)snapshot.put("primaryScore",measured.primaryScore.toJson());
                            log.write(snapshot.toString(2).getBytes(StandardCharsets.UTF_8));
                        }catch(Exception ignored){}
                    }
                }
                final long duration=(System.nanoTime()-start)/1000000;
                runOnUiThread(()->{
                    if(destroyed||token!=generation)return;
                    busy=false;points=output;result=measured;elapsedMs=duration;faceCount=detectedCount;
                    if(result!=null&&pendingDeep!=null){if(deepMatchesProfile(pendingDeep,referenceMode))result.deepAnalysis=pendingDeep;pendingDeep=null;}
                    if(output.length==0)status="没有检测到人脸，请使用清晰、正面的单人照片。";
                    else if(measured!=null&&!measured.scorable)status=measured.qualityMessage;
                    else if(detectedCount>1)status="检测到 "+detectedCount+" 张人脸，当前分析已标出的最大人脸。建议使用单人照片。";
                    else status="已检测 "+output.length+" 个关键点 · 本次处理 "+duration+" ms";
                    refreshHome();
                });
            }catch(Exception e){runOnUiThread(()->failed(token,"分析未完成，请重试或换一张照片。",e));}
        });
    }
    private void failed(int token,String message,Exception error){
        if(destroyed||token!=generation)return;
        busy=false;result=null;points=new PointF[0];status=message;refreshHome();
        error(message,error);
    }
    private void clearCurrent(){
        if(busy)return;
        generation++;image=null;points=new PointF[0];result=null;pendingDeep=null;currentPath=null;inputIdentity="";sourceName="";status="";hairline=null;calibrating=false;genderResult=null;
        selectTab("home");
    }
    private Bitmap decodeUri(Uri uri)throws IOException{
        BitmapFactory.Options bounds=new BitmapFactory.Options();bounds.inJustDecodeBounds=true;
        try(InputStream stream=getContentResolver().openInputStream(uri)){BitmapFactory.decodeStream(stream,null,bounds);}
        if(bounds.outWidth<=0||bounds.outHeight<=0)throw new IOException("图片格式无法识别");
        BitmapFactory.Options options=sampleOptions(bounds.outWidth,bounds.outHeight);
        Bitmap bitmap;try(InputStream stream=getContentResolver().openInputStream(uri)){bitmap=BitmapFactory.decodeStream(stream,null,options);}
        if(bitmap==null)throw new IOException("图片解码失败");
        int orientation=ExifInterface.ORIENTATION_NORMAL;
        try(InputStream stream=getContentResolver().openInputStream(uri)){if(stream!=null)orientation=new ExifInterface(stream).getAttributeInt(ExifInterface.TAG_ORIENTATION,ExifInterface.ORIENTATION_NORMAL);}catch(Exception ignored){}
        return rotateExif(bitmap,orientation);
    }
    private Bitmap decodeFile(String path)throws IOException{
        BitmapFactory.Options bounds=new BitmapFactory.Options();bounds.inJustDecodeBounds=true;BitmapFactory.decodeFile(path,bounds);
        if(bounds.outWidth<=0||bounds.outHeight<=0)throw new IOException("图片不存在或已损坏");
        Bitmap bitmap=BitmapFactory.decodeFile(path,sampleOptions(bounds.outWidth,bounds.outHeight));
        if(bitmap==null)throw new IOException("图片解码失败");
        int orientation=ExifInterface.ORIENTATION_NORMAL;try{orientation=new ExifInterface(path).getAttributeInt(ExifInterface.TAG_ORIENTATION,ExifInterface.ORIENTATION_NORMAL);}catch(Exception ignored){}
        return rotateExif(bitmap,orientation);
    }
    private BitmapFactory.Options sampleOptions(int width,int height){
        BitmapFactory.Options options=new BitmapFactory.Options();options.inPreferredConfig=Bitmap.Config.ARGB_8888;options.inSampleSize=1;
        while(Math.max(width,height)/options.inSampleSize>1600)options.inSampleSize*=2;
        return options;
    }
    private Bitmap rotateExif(Bitmap bitmap,int orientation){
        Matrix m=new Matrix();
        switch(orientation){
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:m.setScale(-1,1);break;
            case ExifInterface.ORIENTATION_ROTATE_180:m.setRotate(180);break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:m.setScale(1,-1);break;
            case ExifInterface.ORIENTATION_TRANSPOSE:m.setRotate(90);m.postScale(-1,1);break;
            case ExifInterface.ORIENTATION_ROTATE_90:m.setRotate(90);break;
            case ExifInterface.ORIENTATION_TRANSVERSE:m.setRotate(270);m.postScale(-1,1);break;
            case ExifInterface.ORIENTATION_ROTATE_270:m.setRotate(270);break;
            default:return bitmap;
        }
        return Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),m,true);
    }
    private String cacheBitmap(Bitmap bitmap)throws IOException{
        File file=File.createTempFile("analysis-",".jpg",getCacheDir());
        try(FileOutputStream stream=new FileOutputStream(file)){if(!bitmap.compress(Bitmap.CompressFormat.JPEG,95,stream))throw new IOException("临时照片保存失败");}
        return file.getAbsolutePath();
    }

    private void saveRecord(){
        if(busy||image==null||result==null||!result.scorable)return;
        Bitmap bitmap=image;AnalysisResult measured=result;final String previousStatus=status;
        busy=true;status="正在保存记录…";refreshHome();
        worker.execute(()->{
            try{history.save(bitmap,measured);runOnUiThread(()->{if(destroyed)return;busy=false;status=previousStatus;refreshHome();toast("已保存到历史记录");});}
            catch(Exception e){runOnUiThread(()->{if(destroyed)return;busy=false;status=previousStatus;refreshHome();error("保存失败，请重试。",e);});}
        });
    }
    private void showHistory(){
        LinearLayout intro=card();label(intro,"分析记录",22,INK,true);
        label(intro,"仅保存你主动保留的照片和结果，最多 50 条。",13,MUTED,false);
        try{
            List<HistoryStore.Record> records=history.readAll();
            if(history.getUnreadableCount()>0)label(intro,"有 "+history.getUnreadableCount()+" 条记录暂无法显示，原始记录已保留。",13,VisualStyle.WARN,false);
            if(records.isEmpty()){
                LinearLayout empty=card();label(empty,"还没有记录",18,INK,true);label(empty,"分析照片后，点击“保存记录”即可在这里查看。",14,MUTED,false);return;
            }
            // 记录数量显示在列表前。
            label(intro,String.format(Locale.CHINA,"共 %d 条记录",records.size()),12,VisualStyle.OK,true);
            int position=0;
            for(HistoryStore.Record record:records){
                position++;final int index=position;
                LinearLayout item=card(), head=row();head.setGravity(Gravity.CENTER_VERTICAL);item.addView(head);
                ImageView thumb=new ImageView(this);thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
                BitmapFactory.Options options=new BitmapFactory.Options();options.inSampleSize=4;thumb.setImageBitmap(BitmapFactory.decodeFile(record.imagePath,options));
                head.addView(thumb,new LinearLayout.LayoutParams(dp(62),dp(62)));
                LinearLayout lines=new LinearLayout(this);lines.setOrientation(LinearLayout.VERTICAL);LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,-2,1);lp.leftMargin=dp(13);head.addView(lines,lp);
                label(lines,String.format(Locale.CHINA,"%.1f / 10",record.displayScore()),24,INK,true);
                label(lines,record.scoreLabel(),12,MUTED,false);
                String savedProfile=record.hasPrimaryScore()?ScoreProfiles.label(PrimaryScore.referenceProfileFromJson(record.primaryScore))
                        :"几何参照："+modeName(record.referenceMode);
                label(lines,new SimpleDateFormat("MM月dd日 HH:mm",Locale.CHINA).format(new Date(record.timestamp))+" · "+savedProfile,12,MUTED,false);
                LinearLayout buttons=row();marginTop(buttons,13);item.addView(buttons);
                Button view=addRowButton(buttons,"查看详情",View.NO_ID,false,v->showRecord(record));view.setContentDescription("查看记录 "+index);
                Button delete=addRowButton(buttons,"删除",View.NO_ID,false,v->new AlertDialog.Builder(this).setTitle("删除这条记录？").setNegativeButton("取消",null)
                        .setPositiveButton("删除",(dialog,which)->{try{history.delete(record.id);selectTab("history");toast("已删除");}catch(Exception e){error("删除失败",e);}}).show());
                delete.setContentDescription("删除记录 "+index);
            }
            // 清空入口位于列表末尾。
            LinearLayout manage=card();
            label(manage,"管理",15,INK,true);
            label(manage,"清空会删除本机保存的全部照片与分析结果，且无法恢复。",12,MUTED,false);
            Button clear=button("清空全部记录",R.id.clear_history_button,false);
            marginTop(clear,10);manage.addView(clear);
            clear.setOnClickListener(v->new AlertDialog.Builder(this).setTitle("清空记录？").setMessage("已保存的照片和分析结果将从这台设备删除。")
                    .setNegativeButton("取消",null).setPositiveButton("删除",(dialog,which)->{try{history.clear();selectTab("history");toast("记录已清空");}catch(Exception e){error("删除失败",e);}}).show());
        }catch(Exception e){label(intro,"记录暂时无法读取，请稍后重试。",14,MUTED,false);}
    }
    private void showRecord(HistoryStore.Record record){
        LinearLayout details=new LinearLayout(this);details.setOrientation(LinearLayout.VERTICAL);details.setPadding(dp(20),dp(6),dp(20),dp(12));
        ImageView view=new ImageView(this);view.setScaleType(ImageView.ScaleType.FIT_CENTER);view.setImageBitmap(BitmapFactory.decodeFile(record.imagePath));details.addView(view,new LinearLayout.LayoutParams(-1,dp(230)));
        label(details,String.format(Locale.CHINA,"保存时%s：%.1f / 10",record.scoreLabel(),record.displayScore()),24,INK,true);
        if(record.hasPrimaryScore()){
            label(details,String.format(Locale.CHINA,"CNN 原始 %.3f / 5 · 几何对照 %.1f / 10",record.primaryScore.optDouble("rawValue"),record.score),13,MUTED,false);
            label(details,"CNN："+record.primaryScore.optString("modelVersion")+" · "+ScoreProfiles.label(PrimaryScore.referenceProfileFromJson(record.primaryScore)),11,MUTED,false);
            if(!record.primaryScore.has("referenceProfile"))label(details,"旧记录未保存专用模型标识，按原通用评分模型展示；不由当时几何参照组反推模型。",11,MUTED,false);
        }
        label(details,new SimpleDateFormat("yyyy年MM月dd日 HH:mm",Locale.CHINA).format(new Date(record.timestamp)),12,MUTED,false);
        label(details,"算法："+record.json.optString("algorithmVersion",""),11,MUTED,false);
        JSONObject deep=record.json.optJSONObject("deepAnalysis");
        if(deep!=null){
            label(details,String.format(Locale.CHINA,"CNN 参考评分 %.1f / 10（原始 %.3f / 5）",deep.optDouble("cnnScore"),deep.optDouble("cnnRawScore")),17,INK,true);
            label(details,"CNN："+deep.optString("cnnModel")+" · "+ScoreProfiles.label(PrimaryScore.referenceProfileFromJson(deep.optJSONObject("primaryScore")))+"；分割："+deep.optString("segmentationModel"),11,MUTED,false);
            if(deep.has("fpemScore"))label(details,String.format(Locale.CHINA,"FPEM 参考评分 %.1f / 10（原始 %.3f / 5）",deep.optDouble("fpemScore"),deep.optDouble("fpemRawScore")),17,INK,true);
            JSONObject actions=deep.optJSONObject("mouthActions");
            if(actions!=null&&actions.optBoolean("available")){
                label(details,"微笑与嘴部动作",14,INK,true);
                label(details,String.format(Locale.CHINA,"微笑（嘴角上扬）：%s  强度 %.3f",actions.optBoolean("smileDetected")?"检出":"未检出",actions.optDouble("smileCoefficient")),12,MUTED,false);
                label(details,String.format(Locale.CHINA,"下颌张开：%s  强度 %.3f",actions.optBoolean("mouthOpenDetected")?"检出":"未检出",actions.optDouble("jawOpenCoefficient")),12,MUTED,false);
                label(details,"动作系数不代表情绪或校准后的概率。",11,MUTED,false);
                String micro=MicroExpressionStats.lineFromJson(actions.optJSONObject("microExpression"));
                if(!micro.isEmpty()){
                    label(details,"面部可见动作统计",14,INK,true);
                    label(details,micro,12,MUTED,false);
                    label(details,"仅描述单张照片的可见动作分布，不是情绪识别，也不是任何疾病的检查。",11,MUTED,false);
                }
            }
            JSONArray attributes=deep.optJSONArray("visibleAttributes");
            if(attributes!=null){
                label(details,"可见属性参考概率",14,INK,true);
                for(int i=0;i<attributes.length();i++){
                    JSONObject item=attributes.optJSONObject(i);if(item!=null)label(details,String.format(Locale.CHINA,"%s  %.1f%%",item.optString("name"),item.optDouble("probability")*100),12,MUTED,false);
                }
            }
            JSONObject trained=deep.optJSONObject("selfTrainedVisibleAttributes");
            if(trained!=null){
                label(details,"自训练可见属性",14,INK,true);
                try{
                    SelfTrainedVisibleAttributes.Result visible=SelfTrainedVisibleAttributes.Result.fromJson(trained);
                    if(visible.available){
                        for(int i=0;i<3;i++)label(details,String.format(Locale.CHINA,"%s：%s  参考值 %.3f（阈值 %.3f）",i==2?"嘴微张":SelfTrainedVisibleAttributes.DISPLAY_LABELS[i],visible.detected(i)?"检出":"未检出",visible.scores[i],visible.thresholds[i]),12,MUTED,false);
                        label(details,"保存时的模型输出，参考值未经概率校准。",11,MUTED,false);
                    }else label(details,"该记录未取得自训练可见属性结果。",12,MUTED,false);
                }catch(Exception invalid){label(details,"该记录的可见属性格式暂不支持，原始记录仍保留。",12,MUTED,false);}
            }
            JSONArray ablations=deep.optJSONArray("regionAblation");
            if(ablations!=null){
                label(details,"区域依据 原图分减去遮挡后分",14,INK,true);
                for(int i=0;i<ablations.length();i++){
                    JSONObject item=ablations.optJSONObject(i);if(item!=null)label(details,String.format(Locale.CHINA,"%s  %+.3f 分",item.optString("name"),item.optDouble("deltaScore")),12,MUTED,false);
                }
            }
            JSONArray regions=deep.optJSONArray("regions");
            if(regions!=null)for(int i=1;i<regions.length();i++){
                JSONObject region=regions.optJSONObject(i);
                if(region!=null&&region.optInt("pixels")>0)label(details,region.optString("name")+"  "+String.format(Locale.CHINA,"%.2f%%",region.optDouble("percent")),12,MUTED,false);
            }
        }
        JSONObject basis=record.json.optJSONObject("scoringBasis");
        if(basis!=null){
            label(details,"打分依据 · "+ScoringBasis.modeLabel(basis.optString("referenceMode","general")),14,INK,true);
            JSONArray focus=basis.optJSONArray("focus");
            if(focus!=null)for(int i=0;i<focus.length();i++){
                JSONObject item=focus.optJSONObject(i);if(item==null)continue;
                String estimate=item.optString("estimate","");
                String line=String.format(Locale.CHINA,"%s  %.3f %s",item.optString("name"),item.optDouble("value"),item.optString("unit")).trim();
                if(!estimate.isEmpty())line+="  ·  "+estimate;
                label(details,line,12,INK,false);
            }
            String skin=basis.optString("skinUniformityText","");if(!skin.isEmpty())label(details,skin,12,MUTED,false);
            String nature=basis.optString("expressionNaturalness","");if(!nature.isEmpty())label(details,nature,12,MUTED,false);
        }
        JSONObject savedSkin=deep==null?null:deep.optJSONObject("skinUniformity");
        if(savedSkin!=null&&savedSkin.optBoolean("available"))label(details,String.format(Locale.CHINA,"肤色均匀度（深度分析） %.1f / 100",savedSkin.optDouble("uniformity")),12,MUTED,false);
        JSONArray reasons=record.json.optJSONArray("reasons");if(reasons!=null)for(int i=0;i<reasons.length();i++){TextView line=label(details,"• "+reasons.optString(i),13,MUTED,false);marginTop(line,8);}
        JSONArray metrics=record.json.optJSONArray("metrics");if(metrics!=null)for(int i=0;i<metrics.length();i++){JSONObject metric=metrics.optJSONObject(i);if(metric!=null)label(details,metric.optString("name")+"  "+String.format(Locale.CHINA,"%.3f",metric.optDouble("value"))+" "+metric.optString("unit"),12,MUTED,false);}
        ScrollView holder=new ScrollView(this);holder.addView(details);
        new AlertDialog.Builder(this).setTitle("历史详情").setView(holder).setNegativeButton("关闭",null).setPositiveButton("重新分析",(dialog,which)->{selectTab("home");loadFile(record.imagePath,"历史照片",true);}).show();
    }

    private void showResearch(){
        LinearLayout intro=card();label(intro,"研究与实验",22,INK,true);
        label(intro,"把检测结果、评分依据和数据来源放在一起查看。",14,MUTED,false);
        LinearLayout method=card();label(method,"分析流程",16,INK,true);
        label(method,"图像输入 → 人脸关键点 → 几何特征 → 本地评分 → 结果说明",14,ACCENT,true);
        label(method,"参考模式由用户选择。分数反映研究样本的主观评分偏好。",12,MUTED,false);
        LinearLayout deepModels=card();label(deepModels,"深度模型与推理核对",16,INK,true);
        label(deepModels,"CNN：本项目在 SCUT-FBP5500 上迁移训练 MobileNetV3 Small。\n属性：FaceAttribNet，分析眼睛睁开、眼镜、口罩与太阳镜。\n可见动作：MediaPipe 检测左右嘴角上扬与下颌张开，展示微笑和张嘴系数。\n分割：BiSeNet ResNet18，输出19类像素标签。\n评分依据：真实运行五个区域的遮挡实验，展示评分变化。\nFPEM：官方公开检查点本地推理。",13,MUTED,false);
        label(deepModels,"模型均在本机运行。通用CNN保持推荐；男性/女性专用实验模型只由你手动选择，尚未证实更准。SCUT成绩是官方第1折历史基准，各模型和各组分别报告；数值核对不等于真机准确率。",12,MUTED,false);
        LinearLayout basisCard=card();label(basisCard,"打分依据的构成",16,INK,true);
        label(basisCard,"界面上的“打分依据”包含四类信息：",13,INK,false);
        label(basisCard,"1. 面部对称性：以眼睛轴校正倾斜后比较左右对应点。\n2. 五官比例：三庭五眼、眉眼距、鼻宽/嘴宽、外唇高度等二维测量。\n3. 肤色均匀度：只在分割出的皮肤区域统计亮度离散度，指数映射为 0～100。\n4. 表情自然度：由可见的眼部与嘴部开合比判断是否接近中性放松。",13,MUTED,false);
        label(basisCard,"分性别依据按你手动选择的参照组展示：男性参考组侧重轮廓、眉眼间距与下颌线；女性参考组侧重眼型、唇形与面部柔和度。这些侧重来自数据集组内统计，App 不从照片推断性别。",12,MUTED,false);
        label(basisCard,"颜值分数只是人群主观打分的统计结果，不代表客观美丑或人的价值。",12,MUTED,true);
        showExtensionDirections();
        try{
            LinearLayout trained=card();label(trained,"SCUT 自训练与独立测试",16,INK,true);
            label(trained,readAsset("scut_experiment_summary.txt"),13,MUTED,false);
        }catch(IOException ignored){}
        try{
            String summary=readAsset("experiment_summary.txt");
            LinearLayout experiment=card();label(experiment,"模型实验记录",16,INK,true);label(experiment,summary,12,MUTED,false);
        }catch(IOException ignored){}
        paper("SCUT-FBP5500","正面人脸评分基准，比较数据、人工标注和模型评价方式。","https://arxiv.org/abs/1801.06345");
        paper("MEBeauty","多样化图像与主观评分，是本地几何模型的数据来源。","https://github.com/fbplab/MEBeauty-database");
        paper("ICCV 2025 · FPEM","直播场景中的吸引力预测研究，关注画面变化、修饰和泛化问题。","https://openaccess.thecvf.com/content/ICCV2025/html/Li_FPEM_Face_Prior_Enhanced_Facial_Attractiveness_Prediction_for_Live_Videos_ICCV_2025_paper.html");
        paper("FER2013","了解表情数据的标签来源和局限。表情标签不能代替疾病诊断标签。","https://arxiv.org/abs/1307.0414");
        paper("群体差异与识别公平性","阅读唐氏群体人脸识别性能研究，分析数据覆盖与公平性。","https://arxiv.org/abs/2405.11240");
        paper("隐私与研究边界","阅读性取向研究的复现与局限；本应用不根据照片推断这类个人属性。","https://arxiv.org/abs/1902.10739");
        LinearLayout sources=card();label(sources,"示例照片来源",16,INK,true);
        label(sources,"示例照片仅用于体验实际分析流程，分数由当前模型运行生成。",12,MUTED,false);
        Button credits=button("查看图片来源与授权",View.NO_ID,false);marginTop(credits,10);sources.addView(credits);
        credits.setOnClickListener(v->{try{new AlertDialog.Builder(this).setTitle("示例照片来源").setMessage(readAsset("demo/portraits_sources.txt")).setPositiveButton("关闭",null).show();}catch(IOException missing){toast("图片来源信息暂不可用");}});
        LinearLayout data=card();label(data,"数据与照片",16,INK,true);
        label(data,"照片在设备本地分析。历史记录由你主动保存，可随时删除。内置示例与研究资料用于非商业课程学习。",13,MUTED,false);
    }
    /** 展示扩展研究方向及使用边界。 */
    private void showExtensionDirections(){
        LinearLayout card=card();card.setId(R.id.extended_directions_card);
        label(card,"可选扩展方向",16,INK,true);
        label(card,ExtensionEthics.overview(),13,MUTED,false);
        for(ExtensionEthics.Direction direction:ExtensionEthics.directions()){
            label(card,direction.title,14,INK,true);
            label(card,"当前状态："+direction.status,12,MUTED,true);
            label(card,"技术思路："+direction.technique,12,MUTED,false);
            label(card,"伦理边界："+direction.boundary,12,MUTED,false);
        }
        final boolean hasPhoto=image!=null&&points.length>=478&&currentPath!=null;
        Button audit=button(hasPhoto?"打开方法审计与鲁棒性演示（带上当前照片）":"打开方法审计与鲁棒性演示",R.id.open_audit_button,true);
        audit.setOnClickListener(v->{
            Intent intent=new Intent(this,MethodologyAuditActivity.class).putExtra("reference_mode",referenceMode);
            if(hasPhoto){
                float[] coordinates=new float[points.length*2];
                for(int i=0;i<points.length;i++){coordinates[2*i]=points[i].x;coordinates[2*i+1]=points[i].y;}
                intent.putExtra("image_path",currentPath);
                intent.putExtra("landmarks",coordinates);
            }
            startActivity(intent);
        });
        marginTop(audit,10);card.addView(audit);
        label(card,hasPhoto
                ?"将携带首页当前的照片与 478 个关键点；进去后可直接运行三个方向的可见测量。"
                :"可见测量需要人脸关键点。请先到首页选一张照片（或官方示例 A/B），等关键点检测完成后再回来。",
                11,MUTED,false);
        label(card,"可用两种方式核查这类推断的可靠性：对公开报告数值做统计审计，"
                +"或实测同一张脸在拍摄条件变化下的输出漂移。",11,MUTED,false);
        LinearLayout ethics=card();ethics.setId(R.id.ethics_review_card);
        label(ethics,"伦理要求自检",16,INK,true);
        label(ethics,"任务书 §八 的要求与本应用的对应做法：",12,MUTED,false);
        String[] requirements=ExtensionEthics.requirements();
        String[] compliance=ExtensionEthics.compliance();
        for(int i=0;i<requirements.length;i++){
            label(ethics,"要求 "+(i+1)+"："+requirements[i],12,INK,false);
            if(i<compliance.length)label(ethics,"　做法："+compliance[i],12,MUTED,false);
        }
        label(ethics,"人脸数据默认本地处理，不上传服务器；最终 Manifest 不含互联网权限。",12,ACCENT,true);
    }

    private void paper(String title,String description,String url){
        LinearLayout paper=card();label(paper,title,16,INK,true);TextView body=label(paper,description,13,MUTED,false);marginTop(body,6);
        Button open=button("查看原始资料",View.NO_ID,false);marginTop(open,11);paper.addView(open);open.setOnClickListener(v->openLink(url));
    }
    private String readAsset(String name)throws IOException{
        try(InputStream stream=getAssets().open(name);ByteArrayOutputStream out=new ByteArrayOutputStream()){
            byte[] buffer=new byte[4096];int n;while((n=stream.read(buffer))!=-1)out.write(buffer,0,n);return new String(out.toByteArray(),StandardCharsets.UTF_8);
        }
    }
    private void openLink(String url){try{startActivity(new Intent(Intent.ACTION_VIEW,Uri.parse(url)));}catch(ActivityNotFoundException e){toast("没有可用浏览器，资料地址："+url);}}

    /** 按 1～10 分量表绘制中性色刻度条。 */
    private void scoreTrack(LinearLayout parent,double value,double max){
        if(!Double.isFinite(value)||max<=0)return;
        double clamped=Math.max(0,Math.min(max,value));
        LinearLayout wrap=new LinearLayout(this);wrap.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams wrapParams=new LinearLayout.LayoutParams(-1,-2);
        wrapParams.topMargin=dp(8);
        parent.addView(wrap,wrapParams);

        LinearLayout bar=new LinearLayout(this);bar.setOrientation(LinearLayout.HORIZONTAL);
        GradientDrawable track=new GradientDrawable();track.setColor(VisualStyle.SOFT);track.setCornerRadius(dp(4));
        bar.setBackground(track);
        wrap.addView(bar,new LinearLayout.LayoutParams(-1,dp(7)));

        View filled=new View(this);
        GradientDrawable fill=new GradientDrawable();fill.setColor(VisualStyle.CORAL);fill.setCornerRadius(dp(4));
        filled.setBackground(fill);
        float ratio=(float)(clamped/max);
        bar.addView(filled,new LinearLayout.LayoutParams(0,dp(7),Math.max(0.02f,ratio)));
        View rest=new View(this);
        bar.addView(rest,new LinearLayout.LayoutParams(0,dp(7),Math.max(0.0001f,1f-ratio)));

        LinearLayout ticks=new LinearLayout(this);ticks.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams tickParams=new LinearLayout.LayoutParams(-1,-2);
        tickParams.topMargin=dp(4);
        wrap.addView(ticks,tickParams);
        TextView low=text("1",11,MUTED,false);
        ticks.addView(low,new LinearLayout.LayoutParams(0,-2,1));
        TextView high=text("10",11,MUTED,false);
        high.setGravity(Gravity.END);
        ticks.addView(high,new LinearLayout.LayoutParams(0,-2,1));
    }

    /** 卡片内容分隔线。 */
    private View divider(){
        View line=new View(this);
        line.setBackgroundColor(0xfff0ece4);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,Math.max(1,dp(1)));
        params.topMargin=dp(11);params.bottomMargin=dp(4);
        line.setLayoutParams(params);
        return line;
    }

    /** 指标名称、数值与补充说明。 */
    private void metricRow(LinearLayout parent,String name,String value,String description,String trailing){
        LinearLayout row=new LinearLayout(this);row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.TOP);
        LinearLayout.LayoutParams rowParams=new LinearLayout.LayoutParams(-1,-2);
        rowParams.topMargin=dp(9);
        parent.addView(row,rowParams);
        TextView nameView=text(name,13,INK,false);
        row.addView(nameView,new LinearLayout.LayoutParams(0,-2,1));
        TextView valueView=text(value,13,INK,true);
        valueView.setGravity(Gravity.END);
        row.addView(valueView,new LinearLayout.LayoutParams(-2,-2));
        if(description!=null&&!description.isEmpty()){
            TextView note=text(description,11,MUTED,false);
            LinearLayout.LayoutParams noteParams=new LinearLayout.LayoutParams(-1,-2);
            noteParams.topMargin=dp(3);
            parent.addView(note,noteParams);
        }
        if(trailing!=null&&!trailing.isEmpty()){
            TextView hint=text(trailing,11,VisualStyle.TEAL,false);
            LinearLayout.LayoutParams hintParams=new LinearLayout.LayoutParams(-1,-2);
            hintParams.topMargin=dp(2);
            parent.addView(hint,hintParams);
        }
    }

    /** 浅色背景的说明区块。 */
    private void noteBox(LinearLayout parent,String content){
        if(content==null||content.isEmpty())return;
        TextView box=text(content,11,MUTED,false);
        box.setPadding(dp(11),dp(9),dp(11),dp(9));
        GradientDrawable shape=new GradientDrawable();shape.setColor(VisualStyle.CARD_TINT);shape.setCornerRadius(dp(12));
        box.setBackground(shape);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,-2);
        params.topMargin=dp(10);
        parent.addView(box,params);
    }

    private LinearLayout card(){        LinearLayout card=new LinearLayout(this);card.setOrientation(LinearLayout.VERTICAL);card.setPadding(dp(17),dp(17),dp(17),dp(17));
        GradientDrawable bg=new GradientDrawable();bg.setColor(VisualStyle.CARD);bg.setCornerRadius(dp(18));bg.setStroke(dp(1),BORDER);card.setBackground(bg);
        LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2);p.bottomMargin=dp(12);content.addView(card,p);return card;
    }
    private LinearLayout row(){LinearLayout row=new LinearLayout(this);row.setOrientation(LinearLayout.HORIZONTAL);return row;}
    /** 根据字号设置行距。 */
    private void applyLineSpacing(TextView view,float size){
        
        if(size>=VisualStyle.PAGE)view.setLineSpacing(dp(1),1.0f);
        else if(size>=VisualStyle.CARD_TITLE)view.setLineSpacing(dp(2),1.0f);
        else view.setLineSpacing(dp(3),1.0f);
    }

    private TextView label(LinearLayout parent,String value,float size,int color,boolean bold){
        TextView text=text(value,size,color,bold);parent.addView(text,new LinearLayout.LayoutParams(-1,-2));return text;
    }
    private TextView text(String value,float size,int color,boolean bold){
        TextView text=new TextView(this);text.setText(value==null?"":value);
        // 使用统一的字号档位。
        float snapped=VisualStyle.size(size);
        text.setTextSize(snapped);text.setTextColor(color);text.setIncludeFontPadding(true);
        applyLineSpacing(text,snapped);if(bold)text.setTypeface(Typeface.DEFAULT,Typeface.BOLD);return text;
    }
    private void badge(LinearLayout parent,String value){
        TextView badge=text(value,10,VisualStyle.TEAL,true);badge.setPadding(dp(9),dp(5),dp(9),dp(5));
        GradientDrawable bg=new GradientDrawable();bg.setColor(0xddffffff);bg.setCornerRadius(dp(12));badge.setBackground(bg);
        LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-2,-2);p.rightMargin=dp(7);parent.addView(badge,p);
    }
    private Button button(String title,int id,boolean primary){
        AppCompatButton button=VisualStyle.button(this,title,primary);if(id!=View.NO_ID)button.setId(id);
        int icon=id==R.id.gallery_button?R.drawable.ic_action_gallery:id==R.id.camera_button?R.drawable.ic_action_camera:id==R.id.live_button?R.drawable.ic_action_live:0;
        if(icon!=0)VisualStyle.icon(this,button,icon,primary);
        return button;
    }
    private Button addRowButton(LinearLayout row,String title,int id,boolean primary,View.OnClickListener click){
        Button b=button(title,id,primary);b.setOnClickListener(click);LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(0,-2,1);
        if(row.getChildCount()>0)p.leftMargin=dp(7);row.addView(b,p);return b;
    }
    private void marginTop(View view,int value){
        android.view.ViewGroup.LayoutParams existing=view.getLayoutParams();
        LinearLayout.LayoutParams p=existing instanceof LinearLayout.LayoutParams?(LinearLayout.LayoutParams)existing:new LinearLayout.LayoutParams(-1,-2);
        p.topMargin=dp(value);view.setLayoutParams(p);
    }
    private String modeName(String mode){return "male".equals(mode)?"男性参考":"female".equals(mode)?"女性参考":"通用参考";}
    private int dp(float value){return Math.round(value*getResources().getDisplayMetrics().density);}
    private void toast(String value){Toast.makeText(this,value,Toast.LENGTH_SHORT).show();}
    private void error(String message,Exception e){
        android.util.Log.e("FaceScoreLab",message,e);
        if(!isFinishing()&&!destroyed)new AlertDialog.Builder(this).setTitle("操作提示").setMessage(message+"\n你可以重新选择图片或稍后再试。")
                .setPositiveButton("好的",null).show();
    }
    @Override protected void onSaveInstanceState(Bundle out){
        super.onSaveInstanceState(out);out.putString("image_path",currentPath);out.putString("source_name",sourceName);out.putString("reference_mode",referenceMode);
        out.putString("input_identity",inputIdentity);
        out.putString("tab",currentTab);out.putBoolean("show_points",showPoints);out.putBoolean("calibrating",calibrating);if(hairline!=null)out.putFloat("hairline",hairline);
        JSONObject deep=result==null?pendingDeep:result.deepAnalysis;if(deep!=null)out.putString("deep_json",deep.toString());
    }
    @Override protected void onDestroy(){
        destroyed=true;generation++;
        worker.execute(()->{if(engine!=null)try{engine.close();}catch(Exception ignored){}if(cnnEngine!=null)cnnEngine.close();if(genderEngine!=null)genderEngine.close();});worker.shutdown();
        super.onDestroy();
    }
}
