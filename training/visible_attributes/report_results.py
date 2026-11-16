"""Audit frozen visible-attribute predictions and report a paired image bootstrap.

This script performs no training, model selection, threshold fitting or inference.
It preserves all source artifacts and reports only the already selected candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
from evaluation_common import paired_group_bootstrap, read_records, sha256, write_csv, write_json
from evaluate import attribute_flat_metrics, binary_metrics

LABELS = ("Smiling", "Eyeglasses", "Mouth_Slightly_Open")
CHINESE = ("微笑", "眼镜", "嘴微张")
MODEL_NAMES = ("仅训练分类头", "分类头＋后部特征层微调8轮")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, context):
    need(math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12), context)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def local_time(value):
    return datetime.fromisoformat(value).astimezone(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def verify_lock(path):
    lock = read_json(path)
    need(canonical_digest({k: v for k, v in lock.items() if k != "lock_id"}) == lock["lock_id"], "Modified lock")
    run = Path(lock["run"])
    for name, field in (("best.pt", "checkpoint_sha256"), ("visible_attributes.onnx", "onnx_sha256"),
                        ("android_metadata.json", "android_metadata_sha256")):
        need(sha256(run/name) == lock[field], f"Locked artifact changed: {run/name}")
    return lock


def index_rows(path):
    rows = read_records(path)
    result = {r["name"]: r for r in rows}
    need(len(result) == len(rows), f"Duplicate names: {path}")
    return result


def per_image_counts(rows, names, thresholds, raw_truth):
    # Each image is one bootstrap unit. Missing outputs contribute no confusion
    # count, but retain their known-label denominator and unavailable status.
    counts = np.zeros((len(names), 3, 8), np.int64)
    detection = []
    for i, name in enumerate(names):
        row = rows[name]
        need(row["split"] == "test", "Non-test prediction")
        need(row["detected"] in ("0", "1"), "Invalid detection flag")
        available = row["detected"] == "1"
        detection.append(available)
        for j, label in enumerate(LABELS):
            need(row[label+"_known"] in ("0", "1"), "Invalid known mask")
            known = row[label+"_known"] == "1"
            counts[i, j, 4] = int(known)
            if known:
                target = int(row[label+"_label"])
                need(target in (0, 1), "Nonbinary known label")
                need(str(target) == raw_truth[name][label], "Prediction label differs from official-label manifest")
                counts[i, j, 6 if target else 7] = 1
            else:
                need(not raw_truth[name][label].strip(), "Known ground truth lost from predictions")
            if not available:
                need(row[label+"_score"] == "" and row[label+"_prediction"] == "", "Missing face was assigned a prediction")
                continue
            score = float(row[label+"_score"])
            need(math.isfinite(score) and 0 <= score <= 1, "Nonfinite/out-of-range score")
            prediction = int(score >= thresholds[j])
            need(prediction == int(row[label+"_prediction"]), "Saved prediction differs from frozen threshold")
            if known:
                counts[i, j, 5] = 1
                cell = 0 if prediction and target else 1 if prediction else 2 if target else 3
                counts[i, j, cell] = 1
    return counts, np.asarray(detection)


def flat_metrics(counts):
    result = attribute_flat_metrics(counts[:, :4], LABELS)
    for label, cells in zip(LABELS, counts):
        tp, fp, fn, tn, known, available, _, _ = cells
        result[label+".balanced_accuracy"] = float(.5*(tp/(tp+fn)+tn/(tn+fp))) if tp+fn and tn+fp else None
        result[label+".coverage"] = float(available/known) if known else None
    return result


def verify_evaluation(counts, detection, recorded, lock):
    need(recorded["lock_id"] == lock["lock_id"], "Evaluation/lock mismatch")
    need(recorded["onnx_sha256"] == lock["onnx_sha256"], "Evaluation/model mismatch")
    need(recorded["thresholds"] == lock["thresholds"], "Evaluation threshold mismatch")
    need(recorded["thresholds_retuned_on_test"] is False, "Test thresholds were retuned")
    aggregate = counts.sum(0)
    actual = flat_metrics(aggregate)
    original = recorded["heldout"]
    close(actual["macro_f1"], original["macro_f1"], "Macro F1 did not reproduce")
    need(original["coverage"]["images_total"] == len(detection), "Coverage total mismatch")
    need(original["coverage"]["images_with_output"] == int(detection.sum()), "Coverage outputs mismatch")
    rows = []
    for j, (label, chinese) in enumerate(zip(LABELS, CHINESE)):
        measured = binary_metrics(aggregate[j, :4])
        measured["balanced_accuracy"] = actual[label+".balanced_accuracy"]
        source = next(r for r in original["per_label"] if r["label"] == label)
        for key, value in measured.items():
            close(value, source[key], f"{label}.{key} did not reproduce")
        known, available, total_pos, total_neg = (int(x) for x in aggregate[j, 4:])
        need(known == source["known_total"] and available == source["known_available"], "Known denominator mismatch")
        rows.append({"label": label, "display_label": chinese, "threshold": lock["thresholds"][j],
                     **measured, "known_total": known, "unknown_total": len(detection)-known,
                     "available_known": available, "missing_outputs": known-available,
                     "coverage": available/known, "cohort_positive_support": total_pos,
                     "cohort_negative_support": total_neg})
    return rows, actual


def draw_chart(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#f6f8fb")
    x = np.arange(3)
    width = .30
    colors = ("#8294aa", "#227dba")
    for m, key in enumerate(("head", "tail8")):
        values = np.array([r["f1"] for r in result["models"][key]["per_label"]])
        bounds = [result["bootstrap"]["intervals"]["a" if m == 0 else "b"][label+".f1"] for label in LABELS]
        errors = np.array([[v-b["low"] for v,b in zip(values,bounds)], [b["high"]-v for v,b in zip(values,bounds)]])
        centers = x + (-width/2 if m == 0 else width/2)
        ax.bar(centers, values, width, color=colors[m], label=MODEL_NAMES[m], zorder=3)
        ax.errorbar(centers, values, yerr=errors, fmt="none", color="#243447", capsize=5, linewidth=1.2, zorder=4)
        for center, value, bound in zip(centers, values, bounds):
            ax.text(center, bound["high"]+.013, f"{value:.4f}", ha="center", fontsize=11, weight="bold")
    labels = [f"{ch}\n阳性 {r['positive_support']} / 阴性 {r['negative_support']}" for ch,r in zip(CHINESE,result["models"]["head"]["per_label"])]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.07)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_ylabel("F1（并非总体准确率）")
    ax.grid(axis="y", alpha=.18, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.16))
    fig.suptitle("可见属性识别：冻结模型在同一测试图片上的对照", x=.075, y=.97, ha="left", fontsize=18, weight="bold")
    a,b = (result["models"][key]["macro_f1"] for key in ("head", "tail8"))
    interval = result["bootstrap"]["intervals"]["delta_b_minus_a"]["macro_f1"]
    fig.text(.075, .12, f"Macro F1：{a:.4f} → {b:.4f}；差值 +{b-a:.4f}，配对95%区间 [{interval['low']:.4f}, {interval['high']:.4f}]", fontsize=12)
    fig.text(.075, .072, "官方测试集固定抽取2000张；两模型均有输出1992张（99.6%），8张检测失败单列。", fontsize=10, color="#465569")
    fig.text(.075, .035, "误差线：5000次图片级配对bootstrap的95%百分位区间；身份未核验，不代表真机效果。", fontsize=10, color="#465569")
    fig.subplots_adjust(left=.09, right=.97, top=.79, bottom=.26)
    fig.savefig(output/"f1_comparison.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(output/"f1_comparison.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_reports(output, result):
    a, b = (result["models"][k] for k in ("head", "tail8"))
    bootstrap = result["bootstrap"]["intervals"]
    macro_ci = bootstrap["delta_b_minus_a"]["macro_f1"]
    lines = ["FaceScoreLab 可见属性训练结果复核", "",
             f"报告生成：{local_time(result['created_utc'])}",
             f"仅训练分类头的测试集 Macro F1 为 {a['macro_f1']:.6f}；最终后部微调模型为 {b['macro_f1']:.6f}。",
             f"配对差值为 +{b['macro_f1']-a['macro_f1']:.6f}（+{100*(b['macro_f1']-a['macro_f1']):.3f} 个百分点），95%区间 [{macro_ci['low']:.6f}, {macro_ci['high']:.6f}]。",
             "这里的 F1 是精确率与召回率的调和平均，Macro F1 是三类 F1 的等权平均；均不能称为总体准确率。", "",
             "1. 数据与比较范围",
             "训练/验证分别固定抽取 CelebA 官方 train 20000 张、validation 2000 张，seed=42。检测成功数分别为19883和1988。",
             "测试集来自官方 test，固定2000张，seed=20261113；head与tail使用完全相同图片、官方标签和预处理。",
             "两模型都在1992张图片上给出输出，覆盖率99.6%；8张检测失败不作为负例或正确预测，保留在覆盖率分母中。",
             "当前指标条件于存在有效输出；身份信息未获得，官方图片划分不能证明人物身份独立。", "",
             "2. 训练方法",
             "两者均从ImageNet初始化的MobileNetV3 Small开始，使用三个独立sigmoid输出，无CelebA属性模型初始化。",
             "输入使用同一个MediaPipe模型选最大人脸，faceCrop v1裁剪、RGB 224×224、ImageNet归一化。",
             "分类头：LayerNorm、576→128线性层、Hardswish、Dropout 0.2，以及三个独立二分类输出头。",
             "head：冻结全部特征层，只训练分类头；AdamW lr=0.0008、weight_decay=0.01、batch=64、seed=42，最多50轮、耐心8。实际训练22轮，验证最佳为第14轮。",
             "tail8：重新执行同样的head规则，然后从该次最佳head继续微调features[9:]（索引9—12四个后部模块）；前缀固定、BN统计固定，分类头lr=0.0001、后部lr=0.000025。实际额外训练8轮，最终为tail第8轮。",
             "tail8中的8指训练轮数，不是解冻8层。两者都用按训练集负/正样本数加权的BCE，未知标签mask不计损失。",
             "新tail路线的缓存计算分批方式与旧head略有不同；遵循相同head规则，不声称浮点逐位相同。", "",
             "3. 选择、阈值与时间",]
    for row in result["timeline"]:
        lines.append(f"{row['event']}：{row['local_time']}（UTC {row['utc']}）")
    lines += [f"验证集Macro F1：{result['selection']['head_validation_macro_f1']:.6f} → {result['selection']['tail_validation_macro_f1']:.6f}；改善{result['selection']['gain']:.6f}，超过预登记门槛0.005。",
              "最终模型按验证集规则选定后才在本报告中比较测试成绩；不因测试成绩重新选模型或调整阈值。",
              "预登记记录声明：设计tail时模型选择者未查看已产生的head测试成绩；记录和文件时间不能独立证明人的读取历史。",
              "各类阈值仅在验证集候选网格0.05到0.95、步长0.025上选择。输出是未经概率校准的sigmoid分数，不能解释为真实置信概率。", "",
              "4. 各类测试结果（支持数指1992张有效输出中的标签数量）"]
    for j, label in enumerate(LABELS):
        ar,br = a["per_label"][j],b["per_label"][j]
        ci = bootstrap["delta_b_minus_a"][label+".f1"]
        lines += [f"{CHINESE[j]}（{label}）：阳性{ar['positive_support']}、阴性{ar['negative_support']}；全2000张阳性{ar['cohort_positive_support']}、阴性{ar['cohort_negative_support']}，未知0。",
                  f"  head：P={ar['precision']:.6f}, R={ar['recall']:.6f}, F1={ar['f1']:.6f}, balanced accuracy={ar['balanced_accuracy']:.6f}, 阈值={ar['threshold']:.3f}。",
                  f"  tail：P={br['precision']:.6f}, R={br['recall']:.6f}, F1={br['f1']:.6f}, balanced accuracy={br['balanced_accuracy']:.6f}, 阈值={br['threshold']:.3f}。",
                  f"  F1差值={br['f1']-ar['f1']:+.6f}；95%配对区间[{ci['low']:.6f}, {ci['high']:.6f}]。"]
    lines += ["", "5. 不确定性与边界",
              "使用5000次有放回图片级配对bootstrap，seed=20261113，每次从同一2000图片抽取2000项，两个模型共享抽样索引；失败图片作为无输出保留，指标仍对有效输出计算。区间为2.5%和97.5%分位数。",
              "身份未知，故没有按人物聚类抽样；同一人物的相关图片可能让区间偏窄。区间仅反映固定模型在本测试样本上的抽样波动，不涵盖训练随机性、调参选择或真实手机域偏移。",
              "眼镜阳性仅134个，需结合支持数和区间判断。Eyeglasses沿用CelebA原始标签含义，未声称普通眼镜/墨镜分别验证。",
              "仅训练种子42，没有多种子属性训练复验。未拿FaceAttribNet未评估的5属性作为对照。口罩、闭眼、分割mIoU/Dice、真机摄像头识别效果仍未测；已有功能存在不等于已有准确性证据。",
              "本次比较不支持疾病、认知健康、性取向等敏感属性照片推断。", "",
              "6. 复算与交付",
              "从两份原始逐图预测CSV按已冻结阈值重新生成混淆矩阵、支持数、P/R/F1、balanced accuracy和覆盖率，与原evaluation.json逐项核对通过。",
              "同时校验选模记录中的训练指标哈希、两份不可变锁、模型/阈值元数据哈希、固定cohort清单与原始标签；没有重新推理或改写训练产物。",
              "comparison.csv：验证与测试模型对照；per_label_metrics.csv：每类完整统计；paired_bootstrap.csv：模型与差值95%区间；coverage_support.csv：各划分覆盖率与支持数。",
              "f1_comparison.png / .svg：三类F1及区间；results.json：结构化结果及来源哈希；recomputation_audit.json：验证记录。", ""]
    text = "\n".join(lines)
    (output/"结果说明.txt").write_text(text, encoding="utf-8")
    table_rows = []
    for j, label in enumerate(LABELS):
        ar,br = a["per_label"][j],b["per_label"][j]
        table_rows.append(f"<tr><td>{CHINESE[j]}</td><td>{ar['positive_support']} / {ar['negative_support']}</td><td>{ar['f1']:.4f}</td><td>{br['f1']:.4f}</td><td>+{br['f1']-ar['f1']:.4f}</td></tr>")
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FaceScoreLab 可见属性训练结果</title>
<style>body{font-family:'Microsoft YaHei',sans-serif;color:#203249;background:#f3f6fa;max-width:1080px;margin:36px auto;padding:0 24px;line-height:1.8}h1{font-size:27px}article{background:white;padding:28px;border-radius:14px;margin:20px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #dce5ee;text-align:left}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.9 'Microsoft YaHei',sans-serif}.hero{font-size:24px;color:#176ca0;font-weight:bold}.muted{color:#627287;font-size:14px}a{color:#176ca0}</style>
<h1>FaceScoreLab · 可见属性训练结果</h1><article><div class="hero">测试集 Macro F1：HEAD → TAIL</div>
<p>同一2000张测试图片，1992张有效输出；最终模型及阈值已按验证集冻结。F1并非总体准确率。</p>
<table><tr><th>属性</th><th>有效阳性 / 阴性</th><th>仅分类头 F1</th><th>后部微调 F1</th><th>差值</th></tr>ROWS</table></article>
<article><img src="f1_comparison.png" alt="三个可见属性的F1及图片级bootstrap区间"></article>
<article><h2>完整说明</h2><pre>TEXT</pre></article><p class="muted">来源和完整统计见同目录 results.json、comparison.csv、per_label_metrics.csv、paired_bootstrap.csv、coverage_support.csv。</p></html>"""
    document = document.replace("HEAD", f"{a['macro_f1']:.4f}").replace("TAIL", f"{b['macro_f1']:.4f}").replace("ROWS", "".join(table_rows)).replace("TEXT", html.escape(text))
    (output/"report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    work = root/"visible-attributes"
    need(not (output/"results.json").exists(), "Preserve completed reports; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    paths = {"selection": work/"final_validation_selection.json", "preregistration": work/"tail_extension_protocol.json",
             "head_lock": work/"locked-candidate.json", "tail_lock": work/"locked-final-validation-winner.json",
             "selected_artifacts": work/"final_selected_artifacts.json", "cohort_reuse": work/"fixed_cohort_reuse.json",
             "head_metrics": work/"head-seed42"/"metrics.json", "tail_metrics": work/"tail8-seed42"/"metrics.json",
             "head_evaluation": work/"heldout-evaluation"/"evaluation.json", "tail_evaluation": work/"heldout-evaluation-final"/"evaluation.json",
             "head_predictions": work/"heldout-evaluation"/"heldout_predictions.csv", "tail_predictions": work/"heldout-evaluation-final"/"heldout_predictions.csv",
             "heldout_manifest": root/"datasets"/"celeba"/"heldout.csv", "heldout_protocol": root/"datasets"/"celeba"/"heldout_protocol.json"}
    selection, protocol = read_json(paths["selection"]), read_json(paths["preregistration"])
    head, tail = read_json(paths["head_metrics"]), read_json(paths["tail_metrics"])
    need(selection["protocol_sha256"] == sha256(paths["preregistration"]), "Preregistration changed")
    for key in ("head", "tail"):
        need(selection[key+"_metrics_sha256"] == sha256(paths[key+"_metrics"]), "Validation metrics changed after selection")
    need(selection["test_results_used_for_selection"] is False and protocol["test_results_used_to_design_this_recipe"] is False, "Selection declaration mismatch")
    close(selection["gain"], tail["best_validation_macro_f1"]-head["best_validation_macro_f1"], "Validation gain mismatch")
    need(selection["gain"] >= selection["required_gain"] == .005, "Registered selection rule not met")
    locks = [verify_lock(paths["head_lock"]), verify_lock(paths["tail_lock"])]
    selected = read_json(paths["selected_artifacts"])
    need(Path(selection["selected_run"]).resolve() == Path(locks[1]["run"]).resolve() == Path(selected["selected_run"]).resolve(), "Frozen winner mismatch")
    need(Path(selected["lock"]).resolve() == paths["tail_lock"].resolve() and Path(selected["evaluation"]).resolve() == paths["tail_evaluation"].resolve(), "Selected artifact path mismatch")
    need(datetime.fromisoformat(protocol["registered_utc"]) < datetime.fromisoformat(selection["created_utc"]) < datetime.fromisoformat(locks[1]["created_utc"]), "Invalid registration-selection-lock chronology")
    need(locks[0]["preprocessing"] == locks[1]["preprocessing"], "Unmatched preprocessing")
    for lock in locks:
        need(lock["heldout_selection"]["sample_count"] == 2000 and lock["heldout_selection"]["seed"] == 20261113, "Changed cohort selection")
    cohort_protocol = read_json(paths["heldout_protocol"])
    names = sorted(cohort_protocol["selected_names"]["test"])
    need(len(names) == len(set(names)) == 2000 and cohort_protocol["seed"] == 20261113, "Invalid fixed cohort")
    raw_truth = index_rows(paths["heldout_manifest"])
    need(set(raw_truth) == set(names) and all(r["split"] == "test" for r in raw_truth.values()), "Raw test manifest mismatch")
    reuse = read_json(paths["cohort_reuse"])
    need(reuse["original_protocol_sha256"] == sha256(paths["heldout_protocol"]), "Cohort protocol changed")
    need(reuse["heldout_manifest_sha256"] == sha256(paths["heldout_manifest"]) and reuse["final_lock_id"] == locks[1]["lock_id"], "Cohort reuse binding mismatch")
    predictions = [index_rows(paths[key+"_predictions"]) for key in ("head", "tail")]
    need(all(set(p) == set(names) for p in predictions), "Models do not share identical 2000 image cohort")
    counts, detections, measured, flat = [], [], [], []
    for key, rows, lock in zip(("head", "tail"), predictions, locks):
        c,d = per_image_counts(rows, names, lock["thresholds"], raw_truth)
        m,f = verify_evaluation(c,d,read_json(paths[key+"_evaluation"]),lock)
        counts.append(c); detections.append(d); measured.append(m); flat.append(f)
    need(np.array_equal(detections[0], detections[1]), "Different detection subsets require explicit matched-cohort handling")
    need(np.array_equal(counts[0][:,:,4:], counts[1][:,:,4:]), "Different target masks or support")
    bootstrap = paired_group_bootstrap(counts[0], flat_metrics, counts[1], repeats=5000, seed=20261113)
    bootstrap.update(resampling_unit="image; identity unknown", method="paired image percentile bootstrap, 95% interval",
                     group_definition="One filename per group, not one person", missing_output_policy="Included among 2000 draws; zero confusion contribution; retained coverage denominator",
                     limitation="Does not estimate identity-cluster dependence, training randomness or mobile-domain shift")
    models = {}
    for i, (key, metric) in enumerate(zip(("head", "tail8"),(head,tail))):
        models[key] = {"description": MODEL_NAMES[i], "run": locks[i]["run"], "lock_id": locks[i]["lock_id"],
                       "onnx_sha256": locks[i]["onnx_sha256"], "thresholds": locks[i]["thresholds"],
                       "validation_macro_f1": metric["best_validation_macro_f1"], "macro_f1": flat[i]["macro_f1"],
                       "per_label": measured[i], "coverage": {"total": 2000,"available": int(detections[i].sum()),"failed": int((~detections[i]).sum()),"rate": float(detections[i].mean())},
                       "training": {"best_stage": metric["best_stage"],"best_epoch": metric["best_epoch"],"epochs_completed": metric["epochs_completed"],
                                    "prepare_and_training_seconds": metric["phase_wall_seconds"],"preparation_seconds": metric["preparation_charged_seconds"]}}
    timeline = [{"event": label, "utc": when, "local_time": local_time(when)} for label,when in
                (("Head候选锁定",locks[0]["created_utc"]),("Tail对照预登记",protocol["registered_utc"]),
                 ("最终验证集选模记录",selection["created_utc"]),("最终候选锁定",locks[1]["created_utc"]))]
    sources = {key:{"path":str(path),"sha256":sha256(path)} for key,path in paths.items()}
    for path in (Path(__file__), EVALUATION/"evaluation_common.py", EVALUATION/"evaluate.py"):
        sources["report_code_"+path.name] = {"path":str(path),"sha256":sha256(path)}
    result = {"format":"facescorelab-visible-attributes-final-report-v1", "created_utc":datetime.now(timezone.utc).isoformat(),
              "frozen_selection_only":True,"selection":selection,"timeline":timeline,"models":models,"bootstrap":bootstrap,
              "difference_tail_minus_head_macro_f1":flat[1]["macro_f1"]-flat[0]["macro_f1"],
              "identity_independence_verified":False,"probability_calibration_performed":False,"real_phone_accuracy_measured":False,
              "unmeasured":["mask","closed_eye","segmentation_mIoU","segmentation_Dice","FaceAttribNet_5_attribute_accuracy"],"sources":sources}
    comparison, per_class = [], []
    for key,model in models.items():
        for split,value in (("validation",model["validation_macro_f1"]),("test",model["macro_f1"])):
            comparison.append({"method":key,"description":model["description"],"split":split,"macro_f1":value,
                               "images_total":2000,"images_with_output":1988 if split=="validation" else model["coverage"]["available"],
                               "coverage":.994 if split=="validation" else model["coverage"]["rate"],"selected_for_release":key=="tail8"})
        per_class.extend({"method":key,**r} for r in model["per_label"])
    write_csv(output/"comparison.csv",comparison)
    write_csv(output/"per_label_metrics.csv",per_class)
    bootstrap_rows = []
    for model_name,values in bootstrap["intervals"].items():
        for name,ci in values.items():
            estimate = flat[0][name] if model_name=="a" else flat[1][name] if model_name=="b" else flat[1][name]-flat[0][name]
            bootstrap_rows.append({"method":{"a":"head","b":"tail8","delta_b_minus_a":"tail8_minus_head"}[model_name],"metric":name,"estimate":estimate,
                                   "ci95_low":ci["low"],"ci95_high":ci["high"],"valid_replicates":ci["valid_replicates"],"total_replicates":ci["total_replicates"],
                                   "resampling_unit":"image_identity_unknown","seed":20261113})
    write_csv(output/"paired_bootstrap.csv",bootstrap_rows)
    support_rows = []
    for split,manifest,total in (("train",work/"prepared-dev"/"development.csv",20000),
                                 ("validation",work/"prepared-dev"/"development.csv",2000),
                                 ("test",work/"prepared-heldout-final"/"heldout.csv",2000)):
        rows = [r for r in read_records(manifest) if r["split"]==split]
        need(len(rows)==total,"Prepared support denominator changed")
        for label,chinese in zip(LABELS,CHINESE):
            known = [r for r in rows if r[label+"_mask"]=="1"]
            available = [r for r in known if r["detected"]=="1"]
            support_rows.append({"split":split,"label":label,"display_label":chinese,"total_images":total,"known_total":len(known),"unknown_total":total-len(known),
                                 "positive_total":sum(r[label]=="1" for r in known),"negative_total":sum(r[label]=="0" for r in known),
                                 "available_known":len(available),"positive_available":sum(r[label]=="1" for r in available),"negative_available":sum(r[label]=="0" for r in available),
                                 "unavailable_known":len(known)-len(available),"coverage":len(available)/len(known)})
        sources["support_"+split] = {"path":str(manifest),"sha256":sha256(manifest)}
    write_csv(output/"coverage_support.csv",support_rows)
    write_json(output/"recomputation_audit.json",{"passed":True,"checks":["Frozen selection metric hashes and rule","Canonical immutable lock IDs and artifact hashes",
               "Shared 2000-image cohort and official CSV labels","Frozen score threshold decisions","Detection subset and known masks identical",
               "Confusion matrix and P/R/F1/balanced-accuracy reproduce both evaluations","Source cohort provenance hashes","Preregistration-selection-final-lock timestamp order"],
               "max_numeric_tolerance":1e-12,"bootstrap_repeats":5000,"sources":sources,"no_training_or_inference_performed":True})
    draw_chart(output,result)
    make_reports(output,result)
    write_json(output/"results.json",result)
    write_json(output/"output_manifest.json",{p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in output.iterdir() if p.is_file() and p.name!="output_manifest.json"})
    print(json.dumps({"output":str(output),"head_macro_f1":flat[0]["macro_f1"],"tail_macro_f1":flat[1]["macro_f1"],
                      "macro_gain":result["difference_tail_minus_head_macro_f1"],"macro_gain_ci95":bootstrap["intervals"]["delta_b_minus_a"]["macro_f1"],
                      "per_label_gain_ci95":{label:bootstrap["intervals"]["delta_b_minus_a"][label+".f1"] for label in LABELS}},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
