"""Summarize verified public source evidence and the bounded CelebA preparation state."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re

from evaluation_common import write_json
from prepare_celeba_subset import EXPECTED, RUN_ROOT, SAFE_ATTRIBUTES


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    root = RUN_ROOT / "data-audit"
    meta = root / "availability_metadata"
    dataset = RUN_ROOT / "datasets/celeba"
    request_log = load(meta / "request_log.json")
    requests = request_log["requests"]
    original = load(root / "availability_followup.json")
    raw_candidates = original.get("candidates", [])
    candidates = []
    for repository in ("flwrlabs/celeba", "huggan/CelebA-faces-with-attributes", "Yuehao/celeba", "tpremoli/CelebA-attrs"):
        slug = repository.replace("/", "__")
        tree_path = meta / (slug + "__data_tree.json")
        root_path = meta / (slug + "__root_tree.txt")
        files = load(tree_path if tree_path.exists() else root_path)
        parquet = [row for row in files if row.get("path", "").endswith(".parquet")]
        split_bytes = {split: sum(row["size"] for row in parquet if re.search(r"/" + split + "-", row["path"])) for split in ("train", "valid", "validation", "test")}
        row = {"repository": repository, "public_repository": True, "captured_file_list": str(tree_path if tree_path.exists() else root_path),
               "parquet_file_count": len(parquet), "parquet_total_bytes": sum(item["size"] for item in parquet),
               "parquet_bytes_by_split": {key: value for key, value in split_bytes.items() if value},
               "source_card": str(meta / (slug + "__readme.txt")) if (meta / (slug + "__readme.txt")).exists() else None}
        if repository == "flwrlabs/celeba":
            row.update({"license": "Explicit original CelebA Dataset Release Agreement, non-commercial research; identity labels remain on-request under original terms",
                        "license_evidence": str(meta / "flwrlabs__celeba__LICENSE.txt"),
                        "required_attributes_present": SAFE_ATTRIBUTES, "attribute_type": "bool",
                        "split_card_counts": {"train": 162770, "valid": 19867, "test": 19962},
                        "split_claim": "Mirror card says original splits and non-overlapping identities; not independently verified here",
                        "original_name_evidence": "First train Parquet footer contains original-style six-digit PNG names; full original filename/label/split mapping was not verified",
                        "public_range": next((item for item in requests if item["name"] == "flwrlabs__celeba__first_train_parquet_tail.bin"), None),
                        "selection": "Not selected: larger PNG Parquet transport and unverified exact original mapping; identity column not acquired for training"})
        elif repository == "huggan/CelebA-faces-with-attributes":
            row.update({"license": "No usable dataset card/license found in inspected root; original CelebA restrictions still apply",
                        "split_mapping": "Viewer exposes one train split; original split mapping not verified",
                        "required_attributes_verified": False,
                        "selection": "Not selected: insufficient provenance/split evidence"})
        elif repository == "tpremoli/CelebA-attrs":
            row.update({"license": "Mirror says MIT; this cannot replace original CelebA image restrictions",
                        "required_attributes_present": SAFE_ATTRIBUTES, "attribute_type": "int64",
                        "image_transform": "128x128 according to card",
                        "split_card_counts": {"train": 162770, "validation": 19962, "test": 19867},
                        "split_mapping": "Validation/test counts are opposite original official split counts; exact name mapping not verified",
                        "selection": "Not selected: license mismatch and split/transform differences"})
        else:
            row.update({"license": "Mirror links original CelebA source; original non-commercial research/internal-use terms are separately captured and apply, not a new mirror license",
                        "required_attributes_present": SAFE_ATTRIBUTES,
                        "public_range_verified": True,
                        "range_evidence": [item for item in requests if item["name"] in ("Yuehao__celeba__attribute_header.bin", "Yuehao__celeba__zip_eocd.bin")],
                        "original_names": "ZIP central directory sample matches img_align_celeba/000001.jpg, 000002.jpg etc; full selected mapping verified only after final archive checksum",
                        "exact_files": EXPECTED,
                        "identity_file": "Listed by repository, intentionally not downloaded; original request condition not bypassed",
                        "selection": "Selected original-file transport, conditional on original MD5 for full attributes and image ZIP"})
        candidates.append(row)
    startup = load(dataset / "startup_conditions.json") if (dataset / "startup_conditions.json").exists() else None
    prepared = load(dataset / "preparation_result.json") if (dataset / "preparation_result.json").exists() else None
    budget = load(dataset / "download_budget.json") if (dataset / "download_budget.json").exists() else None
    label_counts = {}
    development = dataset / "dev.csv"
    if prepared:
        seen = set()
        with development.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["name"] in seen or row["split"] not in ("train", "validation"):
                    raise ValueError("Duplicate name or test split in development CSV")
                seen.add(row["name"])
                counts = label_counts.setdefault(row["split"], {"images": 0, "positive": {name: 0 for name in SAFE_ATTRIBUTES}})
                counts["images"] += 1
                for attribute in SAFE_ATTRIBUTES:
                    if row[attribute] not in ("0", "1"):
                        raise ValueError("Development label is not known binary 0/1")
                    counts["positive"][attribute] += int(row[attribute])
                if not Path(row["image_path"]).is_file():
                    raise ValueError("Missing development image")
        if {split: values["images"] for split, values in label_counts.items()} != {"train": 20000, "validation": 2000}:
            raise ValueError("Development counts differ from protocol")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "verified_fixed_development_subset_prepared" if prepared else "original_labels_verified_image_archive_transfer_pending",
              "research_use_conditions": "Original CelebA terms permit non-commercial research and internal use within one organization; no redistribution. Mirror metadata does not grant broader rights.",
              "identity_request_condition": "No identity data requested, downloaded or used; image-level benchmark only",
              "source_authentication": "Public unauthenticated HTTP only; no payment, login, application or access-control bypass",
              "metadata_bytes_read_before_data_expansion": request_log["additional_bytes_read"],
              "authorization_expansion": {"original_metadata_budget_bytes": 20 * 1024**2, "approved_data_download_budget_bytes": 2 * 1024**3, "new_disk_budget_bytes": 25 * 1024**3},
              "candidates": candidates,
              "selected_source": {"repository": "Yuehao/celeba", "revision": startup["repository_revision"] if startup else None,
                                  "image_zip_exact_bytes": EXPECTED["img_align_celeba.zip"]["bytes"],
                                  "attribute_file_exact_bytes": EXPECTED["list_attr_celeba.txt"]["bytes"],
                                  "partition_file_exact_bytes": EXPECTED["list_eval_partition.txt"]["bytes"],
                                  "core_files_total_bytes": sum(value["bytes"] for value in EXPECTED.values()),
                                  "partition_original_md5_verified": bool(startup and startup["original_partition_md5_passed"]),
                                  "full_attributes_original_md5_verified": bool(startup and startup["original_attributes_md5_passed"]),
                                  "full_image_zip_original_md5_verified": bool(prepared),
                                  "public_base_path": f"https://huggingface.co/datasets/Yuehao/celeba/resolve/{startup['repository_revision']}/" if startup else None,
                                  "local_root": str(dataset)},
              "acquisition": prepared if prepared else {"startup": startup, "image_zip_pending": True},
              "download_accounting": budget,
              "development_interface": {"csv": str(development), "columns": ["name", "split", "image_path", *SAFE_ATTRIBUTES],
                                        "encoding": "UTF-8 CSV; 0=known negative, 1=known positive; no unknowns in selected official labels",
                                        "split_values": ["train", "validation"], "seed": 42, "identity_independent": False,
                                        "label_counts": label_counts, "preprocessing_performed": False, "test_images_extracted": 0},
              "overlap_limitations": ["No identity-isolation claim: identity metadata was not acquired.",
                                      "Existing FaceAttribNet/BiSeNet/MediaPipe pretraining-image overlap remains unknown.",
                                      "Future test-set results are public benchmark results, not independent real-phone evidence.",
                                      "CelebA Eyeglasses is retained as its original benchmark label; ordinary-vs-sunglasses semantics need separate validation."],
              "model_training_performed": False}
    write_json(root / "availability_followup.json", report)
    lines = ["CelebA 公开来源核查与下一阶段数据准备", "",
             "当前状态：" + ("原始文件与固定开发子集已校验准备完成。" if prepared else "官方属性/划分MD5通过，原图ZIP下载验真进行中。"),
             "采用公开 Yuehao/celeba 原始文件镜像，并固定仓库提交；以 torchvision 记录的原始文件 MD5 作验真。",
             "原始 CelebA 非商业研究与同组织内部使用条款继续适用；不把镜像许可标签当成更宽的图像授权，不公开再分发数据。",
             "未下载身份表，未登录、付费、提交申请或绕过访问控制；只建立 image-level 开发划分。", "",
             "文件：img_align_celeba.zip 1,443,490,838 字节；list_attr_celeba.txt 26,721,026 字节；list_eval_partition.txt 2,836,386 字节。",
             "属性 MD5：75e246fa4810816ffd6ee81facbd244c；划分 MD5：d32c9cbf5e040fd4025c592c306e6668。",
             "原图ZIP目标 MD5：00d2c5bc6d35e252742224ab0c1e8fcb；" + ("已通过。" if prepared else "尚待全量通过，当前不使用未验真的图片。"),
             "断点使用强ETag和精确Range，最终还核对镜像LFS SHA256；总下载预算2 GiB，新增磁盘预算25 GiB。", "",
             "开发接口：" + str(development),
             "列：name,split,image_path,Smiling,Eyeglasses,Mouth_Slightly_Open；0是已知阴性，1是已知阳性。",
             "选样：seed42，先从排序官方train池抽20000，再从排序官方validation池抽2000，均无放回。没有提取test图片，没有裁剪或训练。", "",
             "备选来源核查：",
             "flwrlabs/celeba 明确保留原协议及三属性，Parquet总大小11,734,694,689字节；原图使用PNG，完整原名/标签/划分映射未独立验真，未选择。",
             "huggan/CelebA-faces-with-attributes 仅一个train划分，缺少充分来源/许可/原划分证明，未选择。",
             "tpremoli/CelebA-attrs 的MIT标签不能覆盖原图条款；128×128变换且validation/test数量与官方相反，未选择。", "",
             "局限：现有预训练模型的图像重叠仍未知；公开基准结果不等于手机独立实测，也不能称身份独立。",
             "完整证据：" + str(root / "availability_followup.json")]
    (root / "公开数据可获得性与准备结果.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(root / "availability_followup.json"), "development_counts": label_counts}))


if __name__ == "__main__":
    main()
