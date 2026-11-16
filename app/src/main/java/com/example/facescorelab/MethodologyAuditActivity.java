package com.example.facescorelab;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.util.AtomicFile;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** 统计方法审计、图像扰动对照及可见特征测量页面。
 * 本页面不执行人群分类。
 */
public final class MethodologyAuditActivity extends AppCompatActivity {
    private static final int INK=VisualStyle.INK,MUTED=VisualStyle.MUTED,PURPLE=VisualStyle.ACCENT,TEAL=VisualStyle.TEAL,WARN=VisualStyle.WARN;
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private LinearLayout body;
    private volatile boolean destroyed;
    private Bitmap photo;
    private float[] points;
    private OnnxLocalEngine engine;
    private String referenceMode="general";

    @Override protected void onCreate(Bundle state){
        super.onCreate(state);
        getWindow().setStatusBarColor(VisualStyle.SURFACE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        String path=getIntent().getStringExtra("image_path");
        photo=path==null?null:BitmapFactory.decodeFile(path);
        points=getIntent().getFloatArrayExtra("landmarks");
        String selected=getIntent().getStringExtra("reference_mode");
        if("male".equals(selected)||"female".equals(selected))referenceMode=selected;
        show();
    }

    private void show(){
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setBackgroundResource(R.drawable.bg_app_gradient);
        LinearLayout bar=new LinearLayout(this);bar.setGravity(Gravity.CENTER_VERTICAL);bar.setPadding(dp(16),dp(10),dp(16),dp(10));
        TextView back=text("‹ 返回",16,PURPLE,false);back.setPadding(0,dp(10),dp(16),dp(10));back.setOnClickListener(v->finish());bar.addView(back);
        bar.addView(text("方法审计与鲁棒性",20,INK,true));root.addView(bar);
        ScrollView scroll=new ScrollView(this);body=new LinearLayout(this);body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(dp(16),0,dp(16),dp(24));scroll.addView(body);root.addView(scroll,new LinearLayout.LayoutParams(-1,0,1));
        setContentView(root);

        LinearLayout intro=card();
        label(intro,"这个页面在做什么",18,INK,true);
        label(intro,ExtensionEthics.overview(),13,MUTED,false);
        noteBox(intro,"适用范围",
                "任务书把“唐氏综合征面部特征识别”和“性取向识别复现”定位为技术探索与伦理讨论，"
                        +"并明确禁止作为医学诊断或检测工具。因此本应用不做这两类人群推断，"
                        +"而是把它们变成可核查的统计审计与鲁棒性实验。");
        label(intro,"本页面不包含任何人群分类模型，不对任何个人或群体输出判断。",12,WARN,true);

        buildAuditSection();
        buildRobustnessSection();
        buildExtensionLabSection();
    }

    // 方法审计

    private void buildAuditSection(){
        LinearLayout card=card();
        label(card,"一、方法论审计",18,INK,true);
        label(card,"下面用明确标出的教学假设演示基率、置信区间与类别不平衡。案例数字不是对论文原表格的摘录；没有适用风险比时不计算E-value。",13,MUTED,false);

        List<MethodologyAudit.Case> cases=MethodologyAudit.presetCases();
        String[] names=new String[cases.size()];
        for(int i=0;i<cases.size();i++)names[i]=cases.get(i).name;
        label(card,"选择案例",14,INK,true);
        Spinner spinner=new Spinner(this);spinner.setId(R.id.audit_case_spinner);
        ArrayAdapter<String> adapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_item,names);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);card.addView(spinner,new LinearLayout.LayoutParams(-1,dp(48)));

        TextView result=label(card,"选择案例后会显示审计结果。",12,MUTED,false);
        result.setId(R.id.audit_result);

        Button run=button("运行审计",R.id.audit_run_button,true);card.addView(run);
        run.setOnClickListener(v->{
            int index=spinner.getSelectedItemPosition();
            if(index<0||index>=cases.size())return;
            result.setText(MethodologyAudit.auditCase(cases.get(index)));
            result.setTextColor(INK);
            saveAuditSnapshot(cases.get(index));
        });
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener(){
            @Override public void onItemSelected(AdapterView<?> parent,View view,int position,long id){
                if(position>=0&&position<cases.size()){
                    result.setText(MethodologyAudit.auditCase(cases.get(position)));
                    result.setTextColor(INK);
                }
            }
            @Override public void onNothingSelected(AdapterView<?> parent){}
        });

        LinearLayout manual=card();
        label(manual,"自己算一次基率效应",20,INK,true);
        label(manual,"假设某系统敏感度 90%、特异度 90%。改变目标人群的先验比例，看阳性预测值如何变化。",12,MUTED,false);
        label(manual,"目标人群先验比例（%）",13,INK,true);
        SeekBar prevalence=new SeekBar(this);prevalence.setId(R.id.audit_prevalence_input);
        prevalence.setMax(1000);prevalence.setProgress(10);
        manual.addView(prevalence,new LinearLayout.LayoutParams(-1,dp(44)));
        TextView prevalenceText=label(manual,"0.1%",13,INK,true);
        TextView manualResult=label(manual,"",13,INK,false);
        manualResult.setId(R.id.audit_result);
        Runnable recompute=()->{
            double p=Math.max(0.0001,prevalence.getProgress()/1000.0);
            MethodologyAudit.PredictiveValues values=MethodologyAudit.baseRate(p,0.90,0.90);
            prevalenceText.setText(String.format(Locale.CHINA,"先验比例 %.2f%%",p*100));
            manualResult.setText(values.summary());
        };
        prevalence.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){
            @Override public void onProgressChanged(SeekBar bar,int progress,boolean fromUser){recompute.run();}
            @Override public void onStartTrackingTouch(SeekBar bar){}
            @Override public void onStopTrackingTouch(SeekBar bar){}
        });
        recompute.run();

        label(manual,"样本量与置信区间",14,INK,true);
        label(manual,"在 90% 准确率下，要拿到不同宽度的 95% 置信区间需要多少样本：",12,MUTED,false);
        int[] totals={50,200,1000,7000};
        StringBuilder builder=new StringBuilder();
        for(int total:totals){
            MethodologyAudit.Interval interval=MethodologyAudit.wilson((int)Math.round(0.9*total),total,0.95);
            builder.append(String.format(Locale.CHINA,"n=%-5d → [%.1f%%, %.1f%%]，宽 %.1f 个百分点%n",
                    total,interval.lower*100,interval.upper*100,interval.width*100));
        }
        label(manual,builder.toString().trim(),12,MUTED,false);
        label(manual,"样本越少，区间通常越宽；比较两个准确率还需要对应样本设计与统计检验，不能只比较单点。",12,WARN,false);
    }

    private void saveAuditSnapshot(MethodologyAudit.Case item){
        try{
            JSONObject json=MethodologyAudit.toJson(item);
            if(json==null)return;
            try(FileOutputStream output=new FileOutputStream(new File(getCacheDir(),"audit_last.json"))){
                output.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        }catch(Exception ignored){}
    }

    // 图像扰动对照

    private void buildRobustnessSection(){
        LinearLayout card=card();
        label(card,"二、鲁棒性演示",18,INK,true);
        label(card,"当前评分档位："+ScoreProfiles.label(referenceMode),12,MUTED,true);
        label(card,"把同一张脸在统一裁剪与预处理下进行 9 项图像处理，重新运行普通颜值CNN并观察输出漂移。"
                +"结果只说明当前评分模型的输入敏感性。",13,MUTED,false);
        if(photo==null){
            label(card,"当前没有可用照片。请先在首页选择照片，再从深度分析页进入本页面。",13,WARN,false);
            return;
        }
        if(points==null||points.length<956){
            label(card,"缺少人脸关键点，无法限定人脸区域。请从深度分析页重新进入。",13,WARN,false);
            return;
        }
        TextView status=label(card,"点击下方按钮开始演示。",12,MUTED,false);
        status.setId(R.id.robustness_status);
        TextView result=label(card,"",12,INK,false);
        result.setId(R.id.robustness_result);
        final LinearLayout rows=new LinearLayout(this);
        rows.setOrientation(LinearLayout.VERTICAL);
        card.addView(rows,new LinearLayout.LayoutParams(-1,-2));
        Button run=button("开始鲁棒性演示（9 次推理）",R.id.robustness_run_button,true);
        card.addView(run);
        run.setOnClickListener(v->{
            run.setEnabled(false);
            status.setText("正在运行…");
            result.setText("");
            rows.removeAllViews();
            android.graphics.Rect crop=OnnxLocalEngine.faceCrop(photo,points);
            final float[] box={crop.left,crop.top,crop.right,crop.bottom};
            worker.execute(()->{
                RobustnessProbe.Result measured;
                try{
                    if(destroyed)return;
                    if(engine==null)engine=new OnnxLocalEngine(this,referenceMode);
                    measured=RobustnessProbe.run(photo,box,value->{
                        if(destroyed)throw new IllegalStateException("页面已关闭");
                        return engine.scoreFaceBitmap(value);
                    },(name,index,total)->runOnUiThread(()->{
                        if(!destroyed)status.setText(String.format(Locale.CHINA,"%d / %d · %s",index,total,name));
                    }));
                }catch(Exception error){
                    android.util.Log.e("FaceScoreLab","Robustness probe failed",error);
                    runOnUiThread(()->{
                        if(destroyed)return;
                        run.setEnabled(true);
                        status.setText("演示未完成，请重试。");
                    });
                    return;
                }
                final RobustnessProbe.Result finalResult=measured;
                runOnUiThread(()->{
                    if(destroyed)return;
                    run.setEnabled(true);
                    status.setText(String.format(Locale.CHINA,"演示结束：%d / %d 项有效。",finalResult.validCount,finalResult.totalCount));
                    result.setText(finalResult.summary);
                    renderRobustnessRows(rows,finalResult);
                    try{writeSnapshot("robustness_last.json",finalResult.toJson().put("referenceProfile",referenceMode));}
                    catch(Exception error){android.util.Log.w("FaceScoreLab","Robustness snapshot not saved",error);}
                });
            });
        });
    }

    /** 显示各扰动条件的分数与相对变化。 */
    private void renderRobustnessRows(LinearLayout host,RobustnessProbe.Result result){
        host.removeAllViews();
        if(result.samples.isEmpty()){
            metricRow(host,"逐项结果","无有效样本",null,true);
            return;
        }
        host.addView(divider());
        TextView caption=new TextView(this);
        caption.setText("逐项结果（原始 1～5 分，括号内为相对原图的偏移）");
        caption.setTextSize(11f);caption.setTextColor(MUTED);
        LinearLayout.LayoutParams captionParams=new LinearLayout.LayoutParams(-1,-2);
        captionParams.bottomMargin=dp(2);
        host.addView(caption,captionParams);
        for(RobustnessProbe.Sample sample:result.samples){
            String value;
            if(!sample.valid){
                value="推理失败";
            }else if(Double.isFinite(result.baseline)){
                value=String.format(Locale.CHINA,"%.3f（%+.3f）",sample.value,sample.value-result.baseline);
            }else{
                value=String.format(Locale.CHINA,"%.3f",sample.value);
            }
            metricRow(host,sample.name,value,null,!sample.valid);
        }
        noteBox(host,"如何解读",
                "漂移不是错误率：本演示没有真值，不能由此得到分类准确率，也不能外推其他模型。"
                        +"Leuner 2019 讨论形态、呈现和生活方式因素贡献仍不明确；本演示不声称已复现论文分类器。"
                        +"本演示只衡量当前颜值模型对拍摄条件的敏感性。");
    }

    private String describeRobustness(RobustnessProbe.Result result){        if(result.samples.isEmpty())return result.summary;
        StringBuilder builder=new StringBuilder();
        builder.append(result.summary).append("\n\n");
        builder.append("逐项结果（原始 1～5 分）：\n");
        for(RobustnessProbe.Sample sample:result.samples){
            if(!sample.valid){
                builder.append(String.format(Locale.CHINA,"· %s：推理失败%n",sample.name));
                continue;
            }
            if(Double.isFinite(result.baseline))builder.append(String.format(Locale.CHINA,"· %s：%.3f（%+.3f）%n",sample.name,sample.value,sample.value-result.baseline));
            else builder.append(String.format(Locale.CHINA,"· %s：%.3f（原图基准不可用）%n",sample.name,sample.value));
        }
        builder.append("\n漂移不是错误率：本演示没有真值，不能由此得到分类准确率，也不能外推其他模型。\n");
        builder.append("Leuner 2019讨论形态、呈现和生活方式因素贡献仍不明确；本演示不声称已复现论文分类器。\n");
        builder.append("本演示只衡量当前颜值模型对拍摄条件的敏感性。");
        return builder.toString();
    }

    // 可见特征测量

    private void buildExtensionLabSection(){
        LinearLayout card=card();
        label(card,"三个研究方向：论文与可见测量",18,INK,true);
        label(card,"保留三个研究方向，当前照片只显示有明确定义的二维比例、眼口开合与图像统计。"
                +"论文没有为旧规则提供疾病/性取向阈值；本项目测量定义与原论文结论分开列出。",12,MUTED,false);
        label(card,"方向一：唐氏相关面部研究 → 二维几何观察\n"
                +"方向二：阿尔茨海默相关研究 → 眼睑与嘴部可见测量\n"
                +"方向三：性取向论文复现讨论 → 拍摄条件与像素对照",12,INK,false);
        if(photo!=null&&points!=null){
            FaceImageView preview=new FaceImageView(this);preview.setImage(photo,toPoints(points));
            card.addView(preview,new LinearLayout.LayoutParams(-1,dp(210)));
            label(card,"绿色点为当前照片的MediaPipe网格；全部比例使用同一照片坐标，未标定真实长度。",11,MUTED,false);
        }
        CheckBox blurToggle=new CheckBox(this);
        blurToggle.setText("增加真实3×3均值模糊像素对照");
        blurToggle.setTextSize(13);
        blurToggle.setTextColor(INK);
        card.addView(blurToggle);

        TextView out=label(card,photo==null||points==null
                ?"当前没有携带照片，无法计算。请按此路径进入：首页 → 选一张照片（或官方示例 A/B）→ 等提示“已检测 478 个关键点” → 深度模型对比 → CNN/属性/五官分割 → 页面最下方“方法审计与鲁棒性演示”。"
                :"点击下方按钮提取当前照片的可见测量。",12,MUTED,false);
        out.setId(R.id.extension_measurement_result);

        final LinearLayout results=new LinearLayout(this);
        results.setOrientation(LinearLayout.VERTICAL);
        card.addView(results,new LinearLayout.LayoutParams(-1,-2));

        Button runExt=button("运行三个方向的可见测量",R.id.extension_measurement_run,true);
        runExt.setOnClickListener(v->{
            android.graphics.PointF[] pts=toPoints(points);
            if(pts==null||pts.length<468){
                out.setText("当前照片缺少完整 478 点人脸网格，请返回首页先完成人脸识别。");
                return;
            }
            boolean blurCtrl=blurToggle.isChecked();
            runExt.setEnabled(false);out.setText("正在提取二维几何、可见开合与图像统计…");
            results.removeAllViews();
            worker.execute(()->{
                try{
                    if(destroyed)return;
                    ExtensionAnalyzer.ExtensionReport geometry=ExtensionAnalyzer.measureGeometry(pts,photo);
                    ExtensionAnalyzer.ExtensionReport actions=ExtensionAnalyzer.measureVisibleActions(pts,measureSymmetry(pts));
                    ExtensionAnalyzer.ExtensionReport quality=ExtensionAnalyzer.measureCaptureQuality(pts,photo,blurCtrl);
                    JSONObject snapshot=new JSONObject().put("methodVersion",ExtensionAnalyzer.VERSION)
                            .put("geometry",geometry.toJson()).put("visibleActions",actions.toJson()).put("captureQuality",quality.toJson())
                            .put("diagnosticUse",false).put("personalAttributeInference",false);
                    writeSnapshot("visible_measurements_last.json",snapshot);
                    runOnUiThread(()->{
                        if(destroyed)return;
                        runExt.setEnabled(true);
                        out.setText("测量完成。以下为当前照片的可见测量。");
                        results.removeAllViews();
                        renderReport(results,geometry,"方向一",PURPLE);
                        renderReport(results,actions,"方向二",TEAL);
                        renderReport(results,quality,"方向三",VisualStyle.CORAL_DEEP);
                    });
                }catch(Exception error){
                    android.util.Log.e("FaceScoreLab","Visible measurement failed",error);
                    runOnUiThread(()->{if(!destroyed){runExt.setEnabled(true);out.setText("测量未完成，请检查照片/关键点后重试；未生成个人属性结论。");}});
                }
            });
        });
        card.addView(runExt);
    }

    private void writeSnapshot(String name,JSONObject json)throws Exception{
        byte[] bytes=json.toString(2).getBytes(StandardCharsets.UTF_8);
        AtomicFile file=new AtomicFile(new File(getCacheDir(),name));FileOutputStream output=null;
        try{output=file.startWrite();output.write(bytes);file.finishWrite(output);}
        catch(Exception error){if(output!=null)file.failWrite(output);throw error;}
    }

    /** 用几何分析器算出的左右几何相似度（0–100）；失败返回 NaN。 */
    private double measureSymmetry(android.graphics.PointF[] pts){
        if(photo==null||pts==null||pts.length<468)return Double.NaN;
        try{
            GeometryAnalyzer analyzer=new GeometryAnalyzer(this);
            AnalysisResult measured=analyzer.analyze(pts,photo.getWidth(),photo.getHeight(),"general",null);
            double value=measured.symmetry;
            return Double.isFinite(value)?value:Double.NaN;
        }catch(Exception error){
            android.util.Log.w("FaceScoreLab","Symmetry measurement unavailable",error);
            return Double.NaN;
        }
    }

    private android.graphics.PointF[] toPoints(float[] values){
        if(values==null)return null;
        android.graphics.PointF[] output=new android.graphics.PointF[values.length/2];
        for(int i=0;i<output.length;i++)output[i]=new android.graphics.PointF(values[2*i],values[2*i+1]);
        return output;
    }

    // 界面辅助方法

    private LinearLayout card(){
        LinearLayout box=new LinearLayout(this);box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16),dp(16),dp(16),dp(16));
        GradientDrawable background=new GradientDrawable();background.setColor(VisualStyle.CARD);background.setCornerRadius(dp(18));
        background.setStroke(dp(1),VisualStyle.BORDER);box.setBackground(background);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,-2);params.bottomMargin=dp(12);
        body.addView(box,params);return box;
    }

    /** 区块之间的细分隔线。 */
    private View divider(){
        View line=new View(this);
        line.setBackgroundColor(0xfff0ece4);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,Math.max(1,dp(1)));
        params.topMargin=dp(10);params.bottomMargin=dp(10);
        line.setLayoutParams(params);
        return line;
    }

    /** 圆角小标签，用于“方向一/方向二”这类分类标识。 */
    private TextView chip(String value,int fg,int bg){
        TextView chip=new TextView(this);chip.setText(value);chip.setTextSize(11);
        chip.setTextColor(fg);chip.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        chip.setPadding(dp(8),dp(3),dp(8),dp(3));
        GradientDrawable shape=new GradientDrawable();shape.setColor(bg);shape.setCornerRadius(dp(9));
        chip.setBackground(shape);
        return chip;
    }

    /** 单条测量：名称左、数值右，定义在下方小字。 */
    private void metricRow(LinearLayout box,String name,String value,String definition,boolean valueMuted){
        LinearLayout row=new LinearLayout(this);row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(android.view.Gravity.TOP);
        LinearLayout.LayoutParams rowParams=new LinearLayout.LayoutParams(-1,-2);
        rowParams.topMargin=dp(9);
        box.addView(row,rowParams);

        TextView nameView=new TextView(this);nameView.setText(name);nameView.setTextSize(13f);nameView.setTextColor(INK);
        row.addView(nameView,new LinearLayout.LayoutParams(0,-2,1));

        TextView valueView=new TextView(this);valueView.setText(value);valueView.setTextSize(14f);
        valueView.setTextColor(valueMuted?MUTED:INK);
        valueView.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        valueView.setGravity(android.view.Gravity.END);
        row.addView(valueView,new LinearLayout.LayoutParams(-2,-2));

        if(definition!=null&&!definition.isEmpty()){
            TextView note=new TextView(this);note.setText(definition);note.setTextSize(11f);note.setTextColor(MUTED);
            note.setLineSpacing(dp(2),1f);
            LinearLayout.LayoutParams noteParams=new LinearLayout.LayoutParams(-1,-2);
            noteParams.topMargin=dp(3);
            box.addView(note,noteParams);
        }
    }

    /** 浅底提示块，用于“文献关系/测量限制”这类附注。 */
    private void noteBox(LinearLayout box,String title,String content){
        LinearLayout wrap=new LinearLayout(this);wrap.setOrientation(LinearLayout.VERTICAL);
        wrap.setPadding(dp(11),dp(10),dp(11),dp(10));
        GradientDrawable shape=new GradientDrawable();shape.setColor(VisualStyle.CARD_TINT);shape.setCornerRadius(dp(12));
        wrap.setBackground(shape);
        LinearLayout.LayoutParams wrapParams=new LinearLayout.LayoutParams(-1,-2);
        wrapParams.topMargin=dp(10);
        box.addView(wrap,wrapParams);

        TextView head=new TextView(this);head.setText(title);head.setTextSize(11f);
        head.setTextColor(MUTED);head.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        wrap.addView(head);
        TextView value=new TextView(this);value.setText(content);value.setTextSize(11.5f);value.setTextColor(MUTED);
        value.setLineSpacing(dp(2),1f);
        LinearLayout.LayoutParams valueParams=new LinearLayout.LayoutParams(-1,-2);
        valueParams.topMargin=dp(4);
        wrap.addView(value,valueParams);
    }

    /** 显示一组可见测量及其定义。 */
    private void renderReport(LinearLayout host,ExtensionAnalyzer.ExtensionReport report,String tag,int accent){
        // 复用外层卡片，仅添加测量分组。
        LinearLayout block=new LinearLayout(this);block.setOrientation(LinearLayout.VERTICAL);
        block.setPadding(dp(12),dp(12),dp(12),dp(12));
        GradientDrawable shape=new GradientDrawable();shape.setColor(VisualStyle.SURFACE);shape.setCornerRadius(dp(14));
        shape.setStroke(dp(1),VisualStyle.BORDER);
        block.setBackground(shape);
        LinearLayout.LayoutParams blockParams=new LinearLayout.LayoutParams(-1,-2);
        blockParams.topMargin=dp(12);
        host.addView(block,blockParams);

        LinearLayout head=new LinearLayout(this);head.setOrientation(LinearLayout.HORIZONTAL);
        head.setGravity(android.view.Gravity.CENTER_VERTICAL);
        block.addView(head);
        head.addView(chip(tag,accent,(0x1c<<24)|(accent&0x00ffffff)));
        TextView title=new TextView(this);title.setText(report.title);title.setTextSize(15f);
        title.setTextColor(INK);title.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        LinearLayout.LayoutParams titleParams=new LinearLayout.LayoutParams(0,-2,1);
        titleParams.leftMargin=dp(8);
        head.addView(title,titleParams);

        if(!report.available){
            TextView bad=new TextView(this);bad.setText("本次不可用："+report.status);
            bad.setTextSize(12f);bad.setTextColor(WARN);bad.setLineSpacing(dp(2),1f);
            LinearLayout.LayoutParams badParams=new LinearLayout.LayoutParams(-1,-2);
            badParams.topMargin=dp(8);
            block.addView(bad,badParams);
            return;
        }
        block.addView(divider());
        for(ExtensionAnalyzer.Measurement row:report.measurements){
            String value=row.available
                    ?String.format(Locale.CHINA,"%.4f",row.value)+row.unit
                    :"不可用";
            metricRow(block,row.name,value,row.definition,!row.available);
        }
        noteBox(block,"文献关系",report.source);
        noteBox(block,"测量限制",report.limitation);
    }

    private TextView text(String value,float size,int color,boolean bold){
        TextView view=new TextView(this);view.setText(value);
        float snapped=VisualStyle.size(size);view.setTextSize(snapped);view.setTextColor(color);
        applyLineSpacing(view,snapped);if(bold)view.setTypeface(Typeface.DEFAULT,Typeface.BOLD);return view;
    }
    /** 根据字号设置行距。 */
    private void applyLineSpacing(TextView view,float size){
        
        if(size>=VisualStyle.PAGE)view.setLineSpacing(dp(1),1.0f);
        else if(size>=VisualStyle.CARD_TITLE)view.setLineSpacing(dp(2),1.0f);
        else view.setLineSpacing(dp(3),1.0f);
    }

    private TextView label(LinearLayout box,String value,float size,int color,boolean bold){
        TextView view=text(value,size,color,bold);LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,-2);
        params.bottomMargin=dp(8);box.addView(view,params);return view;
    }
    private Button button(String value,int id,boolean primary){
        Button button=VisualStyle.button(this,value,primary);if(id!=View.NO_ID)button.setId(id);
        button.setPadding(dp(5),0,dp(5),0);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,dp(48));
        params.topMargin=dp(6);button.setLayoutParams(params);return button;
    }
    private int dp(float value){return Math.round(value*getResources().getDisplayMetrics().density);}

    @Override protected void onDestroy(){
        destroyed=true;
        worker.execute(()->{if(engine!=null)engine.close();if(photo!=null&&!photo.isRecycled())photo.recycle();});
        worker.shutdown();
        super.onDestroy();
    }
}
