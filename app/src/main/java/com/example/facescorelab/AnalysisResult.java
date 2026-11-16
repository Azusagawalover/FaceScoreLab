package com.example.facescorelab;

import java.util.ArrayList;
import java.util.List;
import org.json.JSONObject;

/** Observable geometry and the output of the versioned course regression model. */
public final class AnalysisResult {
    public static final int SCHEMA_VERSION=PrimaryScore.SCHEMA_VERSION;
    /** Primary CNN measurement; score below always retains the legacy geometry meaning. */
    public PrimaryScore primaryScore;
    public String primaryScoreStatus="not_requested";
    /** NaN means that quality checks did not permit scoring. */
    public double score = Double.NaN;
    public double baseline = Double.NaN;
    /** 0-100 geometric similarity index, not an attractiveness probability. */
    public double symmetry;
    public double eyeAspect;
    public double mouthAspect;
    /** Approximate left/right displacement derived from geometry, not calibrated head pose. */
    public double yaw;
    public boolean scorable;
    public boolean hairlineCalibrated;
    public int referenceSampleCount;
    public String qualityMessage = "";
    public String algorithmVersion = "";
    public String referenceMode = "general";
    public JSONObject deepAnalysis;
    /** 分性别打分依据、肤色均匀度、表情自然度；见 ScoringBasis。 */
    public JSONObject scoringBasis;
    public final List<Metric> metrics = new ArrayList<>();
    public final List<String> reasons = new ArrayList<>();

    public static final class Metric {
        public String name;
        public String unit;
        public String description;
        public double value;

        public Metric(String name, double value, String unit, String description) {
            this.name = name;
            this.value = value;
            this.unit = unit;
            this.description = description;
        }
    }
}
