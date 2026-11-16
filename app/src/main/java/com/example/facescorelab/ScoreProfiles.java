package com.example.facescorelab;

import org.json.JSONObject;

/** Explicit, immutable scoring profiles selected by the user, never inferred from a photo. */
public final class ScoreProfiles {
    public static final String GENERAL="general",MALE="male",FEMALE="female";
    public static final String SELECTION="manual";
    public static final String SELECTION_NOTE="评分参照组由用户手动选择；不代表对照片中人物身份的判断。";
    private ScoreProfiles() {}

    public static String normalize(String profile){return MALE.equals(profile)?MALE:FEMALE.equals(profile)?FEMALE:GENERAL;}
    public static boolean isKnown(String profile){return GENERAL.equals(profile)||MALE.equals(profile)||FEMALE.equals(profile);}
    public static String label(String profile){return MALE.equals(profile)?"男性专用实验模型":FEMALE.equals(profile)?"女性专用实验模型":GENERAL.equals(profile)?"通用评分模型（推荐）":"未知评分模型组";}

    public static final class Profile {
        public final String id,file,modelVersion,sha256,preprocessVersion,label,notice,validationRecommendation;
        private Profile(String id,String file,String modelVersion,String sha256,String preprocessVersion,String label,String notice,String recommendation){
            this.id=id;this.file=file;this.modelVersion=modelVersion;this.sha256=sha256;
            this.preprocessVersion=preprocessVersion;this.label=label;this.notice=notice;this.validationRecommendation=recommendation;
        }
    }

    public static Profile resolve(JSONObject manifest,String selected) throws Exception {
        String id=normalize(selected);
        JSONObject profiles=manifest.optJSONObject("score_profiles");
        JSONObject item=profiles==null?null:profiles.optJSONObject(id);
        String file,version,hash,preprocess,title;
        if(item==null){
            if(!GENERAL.equals(id))throw new IllegalStateException("所选专用评分模型未安装，请切回通用或更新安装包");
            JSONObject cnn=manifest.getJSONObject("cnn");
            file="beauty_mobilenetv3_small.onnx";version=cnn.getString("model_version");
            hash=manifest.getJSONObject("assets").getJSONObject(file).getString("sha256");
            preprocess=cnn.optString("preprocess_version",PrimaryScore.PREPROCESS_VERSION);title=label(id);
        }else{
            file=item.getString("file");version=item.getString("model_version");hash=item.getString("sha256");
            preprocess=item.getString("preprocess_version");title=item.optString("label",label(id));
        }
        String expected=GENERAL.equals(id)?"beauty_mobilenetv3_small.onnx":MALE.equals(id)?"beauty_male.onnx":"beauty_female.onnx";
        if(!expected.equals(file)||version.trim().isEmpty()||!hash.matches("[0-9a-f]{64}")
                ||!PrimaryScore.PREPROCESS_VERSION.equals(preprocess))throw new IllegalStateException("评分模型配置无效");
        String assetHash=manifest.getJSONObject("assets").getJSONObject(file).getString("sha256");
        if(!hash.equals(assetHash))throw new IllegalStateException("评分配置与模型资产校验不一致");
        String recommendation=item==null?"general_default_recommended":item.optString("validation_recommendation","");
        String notice=GENERAL.equals(id)?"通用模型保持默认推荐；专用实验档位只在你主动选择时使用。":"专用实验模型，尚未证实更准；默认推荐通用模型。";
        if(item!=null&&!GENERAL.equals(id)){
            JSONObject validation=item.optJSONObject("validation"),general=item.optJSONObject("general_validation");
            double specialized=validation==null?Double.NaN:validation.optDouble("mae",Double.NaN);
            double baseline=general==null?Double.NaN:general.optDouble("mae",Double.NaN);
            if(Double.isFinite(specialized)&&Double.isFinite(baseline))notice+=String.format(java.util.Locale.CHINA,
                    " 组内验证MAE：专用 %.6f，通用 %.6f（专用－通用 %+.6f；原始1～5分）。",specialized,baseline,specialized-baseline);
        }
        return new Profile(id,file,version,hash,preprocess,title,notice,recommendation);
    }
}
