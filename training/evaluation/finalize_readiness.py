"""Write a factual acceptance report from saved tool tests and source-audit evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from evaluation_common import sha256, write_json

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Existing data-audit directory")
    args = parser.parse_args()
    root = args.output.resolve()
    validation = load(root / "tool-validation/validation_summary.json")
    if not validation.get("all_checks_passed"):
        raise ValueError("Tool validation has not passed")
    changed = [name for name, expected in validation["source_sha256"].items() if not (HERE / name).exists() or sha256(HERE / name) != expected]
    if changed:
        raise ValueError(f"Validation predates source changes; rerun run_validation.py: {changed}")
    test_log = (root / "tool-validation/unit_tests.log.txt").read_text(encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", test_log)
    if not count or not test_log.rstrip().endswith("OK"):
        raise ValueError("Missing successful unittest completion evidence")
    models = load(root / "model_provenance_registry.json")
    model_checks = []
    for model in models["models"]:
        current = sha256(model["asset"])
        model_checks.append({"module": model["module"], "asset": model["asset"],
                             "matches_audited_sha256": current == model["sha256"],
                             "matches_existing_project_record": model["hash_matches_record"]})
    if any(not row["matches_audited_sha256"] for row in model_checks):
        raise ValueError("Model assets changed since audit; refresh provenance audit before reporting")
    metadata = load(root / "official_metadata/download_manifest.json")
    registry = load(root / "dataset_registry.json")
    split_audit = load(root / "celeba_official_split_audit.json")
    functional = load(root / "existing_functional_fixture_inventory.audit.json")
    annotation_counts = {}
    for name in ("attribute_annotations.csv", "segmentation_annotations.csv"):
        with (root / "annotation_templates" / name).open(encoding="utf-8-sig", newline="") as handle:
            annotation_counts[name] = sum(1 for _ in csv.DictReader(handle))
    unavailable = [{"name": row["name"], "reason": row.get("error"), "url": row["url"]}
                   for row in metadata["records"] if row["status"] != "downloaded"]
    facts = {
        "audit_date": "2026-11-13",
        "scope": "D0 provenance and data audit; D1 visible-attribute evaluation preparation; S0 segmentation evaluation preparation",
        "readiness": {"D0": "provenance_and_bounded_cache_audit_completed",
                      "D1": "tools_ready_independent_annotations_and_identity_metadata_pending",
                      "S0": "tools_ready_independent_ground_truth_masks_and_identity_metadata_pending"},
        "unit_tests_passed": int(count.group(1)),
        "cli_integration_checks_passed": len(validation["checks"]) - 1,
        "validation_source_hashes_current": True,
        "model_assets_checked": model_checks,
        "public_metadata_files_downloaded": sum(row["status"] == "downloaded" for row in metadata["records"]),
        "saved_public_metadata_bytes": metadata["total_bytes"],
        "full_dataset_images_downloaded": metadata["downloaded_images"],
        "existing_functional_fixtures_registered": functional["n_images"],
        "annotation_template_rows": annotation_counts,
        "real_people_labeled_by_this_work": 0,
        "real_attribute_accuracy_evaluated": False,
        "real_segmentation_accuracy_evaluated": False,
        "attribute_or_segmentation_improvement_claimed": False,
        "bootstrap": {"resampling_unit": "declared anonymous group", "paired_models_share_draws": True,
                      "interval": "95% percentile", "default_replicates": 2000,
                      "unknown_labels_counted_as_negative": False,
                      "missing_predictions_silently_dropped": False,
                      "paired_attribute_cohort": "common available predictions with both model coverage values reported"},
        "mask_rules": {"classes": 19, "ignore_id": 255, "both_absent_class": "null; excluded from macro", "false_positive_only_class": "zero; included", "implicit_resize": False},
        "official_identity_split_audit": split_audit,
        "unavailable_metadata": unavailable,
        "remaining_conditions": [
            "Acquire authorized actual images and labels; full image archives were not downloaded or verified complete.",
            "Official CelebA/CelebAMask-HQ text states identities/attributes are on request; obtain authorized identity metadata or explicitly use a weaker image-level protocol.",
            "Partition/identity download responses did not match known source MD5; they were rejected and not accepted as datasets.",
            "Public archive view pages were accessible; exact archive sizes, full direct download availability, and checksums remain unverified.",
            "Audit each pretrained checkpoint's training image/identity list; existing FaceAttribNet/BiSeNet overlap remains unknown.",
            "Complete independent visible-attribute labels and reviewed masks, freeze group-aware splits, then run real accuracy evaluation.",
            "Do not redistribute CelebA/CelebAMask-HQ images in APK, source bundle or public report attachments; official terms allow non-commercial research and limited internal use."
        ],
        "key_artifacts": {"model_registry": str(root / "model_provenance_registry.json"),
                          "dataset_registry": str(root / "dataset_registry.json"),
                          "local_cache_audit": str(root / "local_cache_audit.json"),
                          "label_guide": str(HERE / "annotation_guide.txt"),
                          "empty_review_templates": str(root / "annotation_templates"),
                          "tool_validation": str(root / "tool-validation/validation_summary.json"),
                          "functional_fixture_manifest": str(root / "existing_functional_fixture_inventory.csv")},
    }
    write_json(root / "data_readiness.json", facts)
    lines = [
        "FaceScoreLab 数据审计与评价准备实际验收报告",
        "日期：2026-11-13", "",
        "验收结论：数据登记、分组划分、可见属性评价及19类分割评价工具已可运行；真实独立标签数据和掩膜尚未准备完成。",
        f"本次通过 {facts['unit_tests_passed']} 项单元测试和 {facts['cli_integration_checks_passed']} 项命令行集成检查。测试输入为明确标记的合成样例，不代表模型识别准确率。", "",
        "一、实际完成",
        "1. 读取当前模型配置、说明和预处理代码，建立5个模型/任务资产来源及训练重叠已知/未知登记；5个资产均重新核对哈希。",
        "2. 在已指定的项目训练目录、旧交付缓存、Google目录及HuggingFace标准缓存位置做有深度和文件数上限的检查；结果不代表全电脑无其他副本。",
        f"3. 保存 {facts['public_metadata_files_downloaded']} 份公开官方/分发元数据，共 {facts['saved_public_metadata_bytes']} 字节；没有下载完整图像数据。",
        f"4. 登记已有 {facts['existing_functional_fixtures_registered']} 张演示照片为 functional_fixture，没有把它们当作准确率测试集。",
        "5. 提供双人独立标注、分歧裁决、unknown/遮挡处理、19类语义映射、空白CSV模板和JSON/CSV输入说明。模板目前无真实标注行。",
        "6. 身份编号HMAC假名化；身份、原文件重复、修正EXIF后的像素重复和已登记衍生图连接分组；冻结划分与新增困难样本仅入训练的检查已实现。",
        "7. 属性Precision/Recall/F1、支持数、未知标签数、推理缺失覆盖率；分割IoU/Dice、类支持、背景单列及混淆矩阵；按group配对bootstrap及差值区间已实现。", "",
        "二、关键正确性检查",
        "unknown、空白、不可判断标签不会变成负例；模型输出不能作为真实评测标签来源。",
        "人工agreed标签必须有两位不同复核者，且两人的原始值与最终值一致；分歧需裁决。",
        "两个模型使用相同group抽样；属性配对比较使用共同可用样本并列出各自覆盖率。样本或组过少时输出局限提示。",
        "掩膜255像素被忽略；双方都无某类时为null，假阳性独有类别为0；颜色图、错误尺寸和非法类别会报错，不静默缩放。",
        "重复图/同一身份跨划分、新增困难样本涉及锁定验证/测试、已冻结哈希被改动等情况会报错。", "",
        "三、数据来源的已验证情况和缺口",
        "CelebA官方页面及CelebAMask-HQ官方README明确为非商业研究用途，不允许向外再分发数据；CelebAMask-HQ来自CelebA，两者不可默认互相独立。",
        "官方说明写明身份/属性元数据需按研究用途申请。未发送申请，也没有假定取得授权或身份数据。",
        "Google Drive公开文件页能显示 CelebAMask-HQ.zip 与 img_align_celeba.zip；完整档案体积、直接下载成功和完整性尚未核实。",
        "尝试获取CelebA划分及身份小文件时，响应不符合torchvision记录的MD5，已拒绝将其接受为数据；尚不能据此完成官方身份交叉审计。",
        "Qualcomm来源的两个info.yaml路径返回404；本地保留的FaceAttribNet文档与当前分发README可用，精确训练图像和身份名单仍未知。",
        "完整图像数据需在来源、许可和体积确认后按计划取得；真实标签由官方可信标注或人工双审提供。", "",
        "四、准备状态",
        "D0：来源、资产、缓存与缺口审计完成。",
        "D1：可见属性评价工具和标注表已就绪；独立真实标签与身份划分未完成。",
        "S0：19类分割评价工具与映射规范已就绪；独立真实掩膜与身份划分未完成。",
        "本工作未训练/替换属性或分割模型，未产生新的真实属性/分割准确率，也未声称精度提高。", "",
        "五、证据位置",
    ]
    lines.extend(f"{name}：{path}" for name, path in facts["key_artifacts"].items())
    (root / "实际验收报告.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"readiness": str(root / "data_readiness.json"), "report": str(root / "实际验收报告.txt"),
                      "unit_tests_passed": facts["unit_tests_passed"], "cli_checks_passed": facts["cli_integration_checks_passed"]}))


if __name__ == "__main__":
    main()
