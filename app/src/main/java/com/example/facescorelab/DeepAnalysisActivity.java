package com.example.facescorelab;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** 本地深度模型分析页面。 */
public final class DeepAnalysisActivity extends AppCompatActivity {
    private static final int INK=VisualStyle.INK,MUTED=VisualStyle.MUTED,PURPLE=VisualStyle.ACCENT,TEAL=VisualStyle.TEAL;
    private static final Object INFERENCE_LOCK=new Object();
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private LinearLayout body;
    private TextView status;
    private Bitmap photo;
    private float[] points;
    private OnnxLocalEngine.Result result;
    private volatile boolean destroyed;
    private boolean busy;
    private double geometryScore;
    private org.json.JSONObject existingDeep;
    private String referenceProfile=ScoreProfiles.GENERAL;
    private OnnxLocalEngine localEngine;
    @Override protected void onCreate(Bundle state){
        super.onCreate(state);getWindow().setStatusBarColor(VisualStyle.SURFACE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        String path=getIntent().getStringExtra("image_path");
        referenceProfile=ScoreProfiles.normalize(getIntent().getStringExtra("reference_mode"));
        points=getIntent().getFloatArrayExtra("landmarks");geometryScore=getIntent().getDoubleExtra("geometry_score",Double.NaN);
        photo=path==null?null:BitmapFactory.decodeFile(path);
        try{
            String saved=state==null?getIntent().getStringExtra("existing_deep"):state.getString("existing_deep");
            if(saved!=null)existingDeep=new org.json.JSONObject(saved);
            if(existingDeep!=null&&!PrimaryScore.matchesProfile(existingDeep.optJSONObject("primaryScore"),referenceProfile))existingDeep=null;
        }catch(Exception ignored){}
        if(photo!=null&&points!=null){
            int width=getIntent().getIntExtra("image_width",photo.getWidth()),height=getIntent().getIntExtra("image_height",photo.getHeight());
            for(int i=0;i<points.length;i+=2){points[i]*=photo.getWidth()/(float)width;points[i+1]*=photo.getHeight()/(float)height;}
        }
        show();
        if(photo!=null&&points!=null&&points.length>=956)analyze();
        else status.setText("照片或人脸关键点不可用，请返回首页重新分析。");
    }
    private void show(){
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setBackgroundResource(R.drawable.bg_app_gradient);
        LinearLayout bar=new LinearLayout(this);bar.setGravity(Gravity.CENTER_VERTICAL);bar.setPadding(dp(16),dp(10),dp(16),dp(10));
        TextView back=text("‹ 返回",16,PURPLE,false);back.setPadding(0,dp(10),dp(16),dp(10));back.setOnClickListener(v->finish());bar.addView(back);
        bar.addView(text("深度模型与五官分割",20,INK,true));root.addView(bar);
        ScrollView scroll=new ScrollView(this);body=new LinearLayout(this);body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(dp(16),0,dp(16),dp(22));scroll.addView(body);root.addView(scroll,new LinearLayout.LayoutParams(-1,0,1));
        LinearLayout intro=card();label(intro,"深度学习分析",22,INK,true);
        label(intro,"自训练图像评分、眼镜与微笑等可见属性、五官分割及区域评分依据。",13,MUTED,false);
        status=label(intro,"准备本地模型…",13,TEAL,true);status.setId(R.id.deep_status);
        if(photo!=null){SegmentationView preview=new SegmentationView(this,photo,null);intro.addView(preview,new LinearLayout.LayoutParams(-1,dp(220)));}
        setContentView(root);
    }
    private void analyze(){
        if(busy||destroyed)return;busy=true;result=null;status.setText("正在加载本地模型，首次分析需要稍候…");
        worker.execute(()->{
            try{
                OnnxLocalEngine.Result measured;
                synchronized(INFERENCE_LOCK){
                    if(destroyed)return;
                    if(localEngine==null)localEngine=new OnnxLocalEngine(this,referenceProfile);
                    measured=localEngine.analyze(photo,points,message->{
                        if(destroyed||Thread.currentThread().isInterrupted())throw new java.util.concurrent.CancellationException("Activity closed");
                        runOnUiThread(()->{if(!destroyed)status.setText(message);});
                    });
                }
                if(destroyed)return;
                if(existingDeep!=null&&OnnxLocalEngine.FPEM_VERSION.equals(existingDeep.optString("fpemModel"))
                        &&Double.isFinite(existingDeep.optDouble("fpemRawScore"))){
                    measured.fpemRawScore=existingDeep.optDouble("fpemRawScore");measured.fpemScore=existingDeep.optDouble("fpemScore");
                    measured.fpemMillis=existingDeep.optLong("fpemMillis");
                }
                // 保存最近一次推理结果。
                try(FileOutputStream output=new FileOutputStream(new File(getCacheDir(),"deep_last.json"))){
                    output.write(measured.toJson().toString(2).getBytes(StandardCharsets.UTF_8));
                }
                runOnUiThread(()->{if(destroyed)return;busy=false;result=measured;showResults();});
            }catch(Exception error){
                android.util.Log.e("FaceScoreLab","Deep model inference failed",error);
                runOnUiThread(()->{
                    if(destroyed)return;busy=false;status.setText("深度分析未完成，请重试。首页的几何分析仍可使用。");
                    Button retry=button("重新运行",R.id.deep_retry_button,true);retry.setOnClickListener(v->{show();analyze();});body.addView(retry);
                });
            }
        });
    }
    private void showResults(){
        body.removeAllViews();
        LinearLayout score=card();label(score,ScoreProfiles.label(referenceProfile),15,MUTED,false);
        TextView number=label(score,String.format(Locale.CHINA,"%.1f / 10",result.score),36,INK,true);number.setId(R.id.cnn_score);
        label(score,String.format(Locale.CHINA,"模型原始输出 %.3f / 5",result.rawScore),13,INK,false);
        if(Double.isFinite(geometryScore))label(score,String.format(Locale.CHINA,"几何参考评分 %.1f / 10",geometryScore),13,MUTED,false);
        label(score,"两个模型学习的数据和特征不同，适合做方法比较；CNN 原始 1～5 分按端点线性映射为 1～10 分。",12,MUTED,false);
        status=label(score,busy?"正在运行 FPEM 论文模型，请稍候…":String.format(Locale.CHINA,"评分 %d ms · 分割 %d ms · 属性 %d ms · 依据 %d ms",result.cnnMillis,result.parsingMillis,result.attributeMillis,result.explanationMillis),12,TEAL,false);status.setId(R.id.deep_status);

        // 方法详情默认折叠。
        final LinearLayout detail=new LinearLayout(this);detail.setOrientation(LinearLayout.VERTICAL);
        label(detail,"SCUT 自训练 MobileNetV3 Small · "+ScoreProfiles.label(referenceProfile),12,MUTED,false);
        label(detail,ScoreProfiles.SELECTION_NOTE,11,MUTED,false);
        try{label(detail,ScoreProfiles.resolve(new org.json.JSONObject(assetText("deep_models.json")),referenceProfile).notice,12,MUTED,false);}
        catch(Exception ignored){if(!ScoreProfiles.GENERAL.equals(referenceProfile))label(detail,"专用实验模型，尚未证实更准；默认推荐通用。",12,MUTED,false);}
        try{
            String experiment=ScoreProfiles.GENERAL.equals(referenceProfile)?assetText("scut_experiment_summary.txt")
                    :"当前为组内适配的专用评分模型；其组内验证和历史测试结果单列，不以通用模型的指标代替。";
            TextView metric=label(detail,experiment,12,MUTED,false);metric.setId(R.id.training_metrics);
        }catch(Exception ignored){}
        detail.setVisibility(View.GONE);
        final TextView detailToggle=text("方法说明与实验数据  ▾",13,VisualStyle.ACCENT,true);
        detailToggle.setPadding(0,dp(8),0,dp(4));
        detailToggle.setOnClickListener(v->{
            boolean open=detail.getVisibility()==View.VISIBLE;
            detail.setVisibility(open?View.GONE:View.VISIBLE);
            detailToggle.setText(open?"方法说明与实验数据  ▾":"方法说明与实验数据  ▴");
        });
        score.addView(detailToggle);
        score.addView(detail,new LinearLayout.LayoutParams(-1,-2));

        LinearLayout attributes=card();attributes.setId(R.id.visible_attributes);label(attributes,"可见属性分析",18,INK,true);
        label(attributes,"FaceAttribNet 独立输出各项参考概率，可同时出现多项。",12,MUTED,false);
        if(result.attributes!=null)for(int i=0;i<result.attributes.length;i++){
            double value=result.attributes[i];
            label(attributes,String.format(Locale.CHINA,"%s   %.1f%%   %s",OnnxLocalEngine.ATTRIBUTE_NAMES[i],value*100,value>=.5?"检出":"未检出"),13,INK,false);
            ProgressBar probability=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);
            probability.setMax(1000);probability.setProgress((int)Math.round(value*1000));
            probability.setProgressTintList(android.content.res.ColorStateList.valueOf(TEAL));
            attributes.addView(probability,new LinearLayout.LayoutParams(-1,dp(8)));
        }
        label(attributes,"这些数值是模型输出，尚未进行独立属性准确率评测；遮挡、光照和角度可能导致误判。",11,MUTED,false);

        if(result.skinUniformity!=null){
            LinearLayout skin=card();label(skin,"肤色均匀度",18,INK,true);
            TextView skinLine=label(skin,result.skinUniformity.line(),13,INK,false);
            skinLine.setId(R.id.skin_uniformity_text);
            if(result.skinUniformity.available){
                ProgressBar uniformity=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);
                uniformity.setMax(1000);uniformity.setProgress((int)Math.round(result.skinUniformity.uniformity*10));
                uniformity.setProgressTintList(android.content.res.ColorStateList.valueOf(TEAL));
                skin.addView(uniformity,new LinearLayout.LayoutParams(-1,dp(8)));
            }
            label(skin,"只在分割出的皮肤类像素上统计相对亮度的离散程度，是图像统计量；不判断肤色深浅的好坏，也不是健康或医学指标。",11,MUTED,false);
        }

        LinearLayout trained=card();trained.setId(R.id.self_trained_visible_attributes);label(trained,"自训练可见属性",18,INK,true);
        SelfTrainedVisibleAttributes.Result visible=result.selfTrainedVisibleAttributes;
        if(visible!=null&&visible.available){
            label(trained,"微笑、眼镜和嘴微张分别判断，可以同时检出。",12,MUTED,false);
            int[] ids={R.id.self_trained_smiling,R.id.self_trained_eyeglasses,R.id.self_trained_mouth_open};
            for(int i=0;i<3;i++){
                TextView row=label(trained,String.format(Locale.CHINA,"%s   %s   参考值 %.3f",i==2?"嘴微张":SelfTrainedVisibleAttributes.DISPLAY_LABELS[i],visible.detected(i)?"检出":"未检出",visible.scores[i]),14,INK,true);
                row.setId(ids[i]);
                label(trained,String.format(Locale.CHINA,"验证集判定阈值 %.3f",visible.thresholds[i]),11,MUTED,false);
            }
            label(trained,String.format(Locale.CHINA,"本地分析 %d ms",visible.durationMillis),12,TEAL,false).setId(R.id.self_trained_visible_status);
            label(trained,"参考值未经概率校准；判断结果仅表示照片中的可见特征。",11,MUTED,false);
            label(trained,"嘴微张表示唇部略有开口；下方另行测量下颌动作强度。",11,MUTED,false);
        }else{
            String message=visible==null||"not_installed".equals(visible.status)?"本版本未安装自训练可见属性模型。":"自训练可见属性暂不可用，请更换照片后重试。";
            label(trained,message,13,MUTED,false).setId(R.id.self_trained_visible_status);
        }

        LinearLayout actions=card();actions.setId(R.id.visible_mouth_actions);label(actions,"微笑与嘴部动作",18,INK,true);
        if(result.mouthActions!=null&&result.mouthActions.available){
            VisibleExpressionEngine.Result action=result.mouthActions;
            actionRow(actions,"微笑（嘴角上扬）",action.smile,VisibleExpressionEngine.SMILE_THRESHOLD,R.id.smile_score);
            actionRow(actions,"下颌张开",action.jawOpen,VisibleExpressionEngine.MOUTH_OPEN_THRESHOLD,R.id.mouth_open_score);
            label(actions,String.format(Locale.CHINA,"左嘴角 %.3f · 右嘴角 %.3f · 分析 %d ms",action.smileLeft,action.smileRight,result.expressionMillis),12,MUTED,false);
            label(actions,"微笑强度取两侧嘴角系数的平均值；判定阈值为 0.50，张嘴阈值为 0.25。",12,MUTED,false);
        }else label(actions,"暂未取得嘴部动作系数，请使用清晰正面照片重新分析。",13,MUTED,false);
        label(actions,"MediaPipe 模型输出的是可见动作强度，不是情绪标签或准确率，也不是校准后的概率。",11,MUTED,false);

        if(result.mouthActions!=null&&result.mouthActions.microExpression!=null){
            LinearLayout micro=card();micro.setId(R.id.micro_expression_card);
            label(micro,"面部可见动作统计",18,INK,true);
            label(micro,"把 52 项可见动作系数汇总为分布统计，用于技术演示与伦理讨论。",12,MUTED,false);
            try{
                org.json.JSONObject stats=result.mouthActions.microExpression;
                TextView line=label(micro,MicroExpressionStats.lineFromJson(stats),13,INK,false);
                line.setId(R.id.micro_expression_summary);
                String dominant=stats.optJSONArray("dominant")==null?"":dominantText(stats.optJSONArray("dominant"));
                if(!dominant.isEmpty())label(micro,dominant,12,MUTED,false);
                org.json.JSONArray groups=stats.optJSONArray("groups");
                if(groups!=null)for(int i=0;i<groups.length();i++){
                    org.json.JSONObject group=groups.optJSONObject(i);if(group==null)continue;
                    label(micro,String.format(Locale.CHINA,"%s  平均系数 %.3f",group.optString("name"),group.optDouble("mean")),12,MUTED,false);
                }
            }catch(Exception ignored){}
            label(micro,"这不是情绪识别，也不是任何疾病的检查。本应用不输出“正常/异常”判断，不设临床阈值。",11,VisualStyle.WARN,false);
            label(micro,"任务书要求：不得宣称可诊断唐氏综合征、阿尔茨海默病或其他疾病。该功能仅用于技术演示与伦理讨论。",11,MUTED,false);
        }

        LinearLayout explanations=card();explanations.setId(R.id.region_explanations);label(explanations,"CNN 评分依据",18,INK,true);
        label(explanations,"逐一将模型分割出的区域替换为灰色，再用同一个 CNN 重新评分。数值为原图分减去遮挡后分。",12,MUTED,false);
        for(int i=0;i<result.occlusion.length();i++){
            org.json.JSONObject region=result.occlusion.optJSONObject(i);if(region==null)continue;
            label(explanations,String.format(Locale.CHINA,"%s    %+.3f 分   遮挡后 %.2f / 10",region.optString("name"),region.optDouble("deltaScore"),region.optDouble("occludedScore")),13,INK,false);
        }
        label(explanations,"这反映模型对当前照片区域的敏感程度，不能解释为改变某个五官就能获得同样分数变化。几何评分的特征贡献在首页查看。",11,MUTED,false);

        LinearLayout fpem=card();label(fpem,"FPEM 论文模型",18,INK,true);
        if(Double.isFinite(result.fpemScore)){
            TextView fpemNumber=label(fpem,String.format(Locale.CHINA,"%.1f / 10",result.fpemScore),33,INK,true);fpemNumber.setId(R.id.fpem_score);
            label(fpem,String.format(Locale.CHINA,"原始输出 %.3f / 5 · %d ms",result.fpemRawScore,result.fpemMillis),13,INK,false);
        }
        label(fpem,"官方检查点的本地推理：融合视觉特征、人脸先验与五档文本先验。原始 1～5 分使用相同端点映射显示。",12,MUTED,false);
        label(fpem,"模型较大，按需运行。已核对推理转换；未复现论文整套训练和数据集评测。",12,MUTED,false);
        Button fpemButton=button(Double.isFinite(result.fpemScore)?"重新运行 FPEM":"运行 FPEM 论文模型",R.id.fpem_button,true);
        fpemButton.setEnabled(!busy);fpemButton.setOnClickListener(v->runFpem());fpem.addView(fpemButton);

        LinearLayout parsing=card();label(parsing,"五官与头发区域",18,INK,true);
        label(parsing,"BiSeNet ResNet18 · 19 类像素分割",12,MUTED,false);
        SegmentationView segmentation=new SegmentationView(this,photo,result);segmentation.setId(R.id.segmentation_image);
        LinearLayout.LayoutParams imageParams=new LinearLayout.LayoutParams(-1,dp(290));imageParams.topMargin=dp(12);parsing.addView(segmentation,imageParams);
        Switch toggle=new Switch(this);toggle.setId(R.id.segmentation_toggle);toggle.setText("显示分割叠层");toggle.setChecked(true);toggle.setTextSize(13);
        toggle.setOnCheckedChangeListener((view,enabled)->segmentation.setShow(enabled));parsing.addView(toggle);
        label(parsing,"叠层透明度",12,MUTED,false);
        SeekBar opacity=new SeekBar(this);opacity.setId(R.id.segmentation_opacity);opacity.setMax(90);opacity.setProgress(55);
        opacity.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
            public void onProgressChanged(SeekBar bar,int value,boolean fromUser){segmentation.setOpacity(value);}
            public void onStartTrackingTouch(SeekBar bar){} public void onStopTrackingTouch(SeekBar bar){}
        });parsing.addView(opacity,new LinearLayout.LayoutParams(-1,dp(40)));
        LinearLayout filters=new LinearLayout(this);parsing.addView(filters);
        String[] names={"全部","眼眉","鼻子","嘴唇","头发"};int[] ids={-1,20,21,22,23};
        for(int i=0;i<names.length;i++){
            final int id=ids[i];Button filter=button(names[i],View.NO_ID,false);filter.setTextSize(12);
            filter.setOnClickListener(v->segmentation.setFilter(id));LinearLayout.LayoutParams param=new LinearLayout.LayoutParams(0,dp(42),1);
            if(i>0)param.leftMargin=dp(4);filters.addView(filter,param);
        }
        label(parsing,"绿色鼻子 · 青色眼睛 · 紫色眉毛 · 粉色嘴唇 · 蓝紫色头发",11,MUTED,false);
        label(parsing,"色块来自逐像素模型预测；边界可能受遮挡、光照和裁剪影响。",12,MUTED,false);
        LinearLayout statistics=card();label(statistics,"区域像素统计",17,INK,true);
        label(statistics,"占裁剪区域像素的比例",12,MUTED,false);
        for(int i=1;i<result.counts.length;i++){
            if(result.counts[i]==0)continue;
            label(statistics,String.format(Locale.CHINA,"●  %s     %.2f%%",OnnxLocalEngine.LABELS[i],100.0*result.counts[i]/(result.size*result.size)),13,INK,false);
        }
        LinearLayout note=card();label(note,"模型来源与实验范围",16,INK,true);
        label(note,"CNN 在本项目中从 ImageNet 通用权重迁移训练，使用 SCUT 官方第1折历史基准复测。FaceAttribNet 与分割采用公开预训练模型；自训练可见属性是单独的三标签模型。基准结果不等于真机拍照准确率，各模块应分别评测。",12,MUTED,false);
        Button audit=button("方法审计与鲁棒性演示",R.id.open_audit_button,false);
        audit.setOnClickListener(v->{
            Intent intent=new Intent(this,MethodologyAuditActivity.class);
            intent.putExtra("image_path",getIntent().getStringExtra("image_path"));
            intent.putExtra("landmarks",points);
            intent.putExtra("reference_mode",referenceProfile);
            startActivity(intent);
        });
        note.addView(audit);
        label(note,"对公开报告的统计审计，以及同一张脸在 9 种拍摄条件下的输出漂移演示。不含人群分类模型。",11,MUTED,false);
        Button attach=button("附加到当前结果",R.id.deep_save_button,true);
        attach.setEnabled(!busy);
        attach.setOnClickListener(v->{
            try{setResult(Activity.RESULT_OK,new Intent().putExtra("deep_json",result.toJson().toString())
                    .putExtra("source_path",getIntent().getStringExtra("image_path"))
                    .putExtra("source_identity",getIntent().getStringExtra("source_identity"))
                    .putExtra("reference_mode",referenceProfile)
                    .putExtra("source_generation",getIntent().getIntExtra("source_generation",-1)));finish();}
            catch(Exception e){Toast.makeText(this,"结果整理失败，请重试",Toast.LENGTH_SHORT).show();}
        });body.addView(attach);
        label(body,"返回首页后可点击“保存记录”，一并保留模型评分和区域统计。",12,MUTED,false);
    }
    private void runFpem(){
        if(busy||result==null)return;busy=true;showResults();
        worker.execute(()->{
            try{
                synchronized(INFERENCE_LOCK){if(destroyed)return;if(localEngine==null)localEngine=new OnnxLocalEngine(this,referenceProfile);localEngine.analyzeFpem(photo,result);}
                if(destroyed)return;
                try(FileOutputStream output=new FileOutputStream(new File(getCacheDir(),"deep_last.json"))){output.write(result.toJson().toString(2).getBytes(StandardCharsets.UTF_8));}
                runOnUiThread(()->{if(destroyed)return;busy=false;showResults();});
            }catch(Exception error){
                android.util.Log.e("FaceScoreLab","FPEM inference failed",error);
                runOnUiThread(()->{if(destroyed)return;busy=false;showResults();status.setText("FPEM 未完成，可重试或保留 CNN 与分割结果。");});
            }
        });
    }
    private LinearLayout card(){
        LinearLayout box=new LinearLayout(this);box.setOrientation(LinearLayout.VERTICAL);box.setPadding(dp(16),dp(16),dp(16),dp(16));
        GradientDrawable background=new GradientDrawable();background.setColor(VisualStyle.CARD);background.setCornerRadius(dp(18));background.setStroke(dp(1),VisualStyle.BORDER);box.setBackground(background);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,-2);params.bottomMargin=dp(12);body.addView(box,params);return box;
    }
    private TextView text(String value,float size,int color,boolean bold){
        TextView view=new TextView(this);view.setText(value);
        float snapped=VisualStyle.size(size);view.setTextSize(snapped);view.setTextColor(color);
        applyLineSpacing(view,snapped);
        if(bold)view.setTypeface(Typeface.DEFAULT,Typeface.BOLD);return view;
    }
    /** 根据字号设置行距。 */
    private void applyLineSpacing(TextView view,float size){
        
        if(size>=VisualStyle.PAGE)view.setLineSpacing(dp(1),1.0f);
        else if(size>=VisualStyle.CARD_TITLE)view.setLineSpacing(dp(2),1.0f);
        else view.setLineSpacing(dp(3),1.0f);
    }

    private TextView label(LinearLayout box,String value,float size,int color,boolean bold){
        TextView view=text(value,size,color,bold);LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,-2);params.bottomMargin=dp(8);box.addView(view,params);return view;
    }
    private Button button(String value,int id,boolean primary){
        Button button=VisualStyle.button(this,value,primary);if(id!=View.NO_ID)button.setId(id);
        button.setPadding(dp(5),0,dp(5),0);
        button.setLayoutParams(new LinearLayout.LayoutParams(-1,dp(48)));return button;
    }
    private int dp(float value){return Math.round(value*getResources().getDisplayMetrics().density);}
    private void actionRow(LinearLayout box,String name,double coefficient,double threshold,int id){
        TextView row=label(box,String.format(Locale.CHINA,"%s   %s   强度 %.3f",name,coefficient>=threshold?"检出":"未检出",coefficient),14,INK,true);
        row.setId(id);
        ProgressBar strength=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);
        strength.setMax(1000);strength.setProgress((int)Math.round(coefficient*1000));
        strength.setProgressTintList(android.content.res.ColorStateList.valueOf(TEAL));
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,dp(10));params.bottomMargin=dp(14);box.addView(strength,params);
    }
    /** 把 JSON 中“较明显动作”数组整理成一行文字。 */
    private String dominantText(org.json.JSONArray dominant){
        if(dominant==null||dominant.length()==0)return "";
        StringBuilder builder=new StringBuilder("较明显的可见动作：");
        for(int i=0;i<dominant.length();i++){
            org.json.JSONObject item=dominant.optJSONObject(i);if(item==null)continue;
            if(i>0)builder.append(" · ");
            builder.append(String.format(Locale.CHINA,"%s %.3f",MicroExpressionStats.displayName(item.optString("name")),item.optDouble("coefficient")));
        }
        return builder.toString();
    }
    private String assetText(String name)throws Exception{        try(java.io.InputStream input=getAssets().open(name);java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream()){
            byte[] bytes=new byte[4096];int n;while((n=input.read(bytes))!=-1)out.write(bytes,0,n);
            return new String(out.toByteArray(),StandardCharsets.UTF_8);
        }
    }
    @Override protected void onSaveInstanceState(Bundle out){
        super.onSaveInstanceState(out);
        try{if(result!=null)out.putString("existing_deep",result.toJson().toString());else if(existingDeep!=null)out.putString("existing_deep",existingDeep.toString());}catch(Exception ignored){}
    }
    @Override protected void onDestroy(){
        destroyed=true;
        worker.execute(()->{if(localEngine!=null)localEngine.close();if(photo!=null&&!photo.isRecycled())photo.recycle();});
        worker.shutdown();super.onDestroy();
    }
}
