"""Audit existing local model provenance; fetch only bounded public metadata when requested."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request
import urllib.parse

from evaluation_common import APP_CLASSES, ATTRIBUTES, MASK_SCHEMA, sha256, write_csv, write_json

PROJECT = Path(__file__).resolve().parents[2]
METADATA = {
    "celeba_official_page": "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
    "celebamask_hq_official_readme": "https://raw.githubusercontent.com/switchablenorms/CelebAMask-HQ/master/README.md",
    "celebamask_hq_official_tree": "https://api.github.com/repos/switchablenorms/CelebAMask-HQ/git/trees/master?recursive=1",
    "torchvision_celeba_loader": "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/celeba.py",
    "faceattrib_distribution_readme": "https://raw.githubusercontent.com/yakhyo/face-attribute/main/README.md",
    "face_parsing_distribution_readme": "https://raw.githubusercontent.com/yakhyo/face-parsing/main/README.md",
    "qualcomm_faceattrib_info": "https://raw.githubusercontent.com/qualcomm/ai-hub-models/main/src/qai_hub_models/models/face_attrib_net/info.yaml",
    "qualcomm_faceattrib_info_legacy_path": "https://raw.githubusercontent.com/qualcomm/ai-hub-models/main/qai_hub_models/models/face_attrib_net/info.yaml",
    "celeba_official_partition": "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pY0NSMzRuSXJEVkk",
    "celeba_official_identity_ids": "https://drive.google.com/uc?export=download&id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
    "celebamask_download_page": "https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view",
    "celeba_images_download_page": "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view",
    "celeba_official_partition_public_download": "https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk&export=download&confirm=t",
    "celeba_official_identity_ids_public_download": "https://drive.usercontent.google.com/download?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS&export=download&confirm=t",
}
EXPECTED_METADATA_MD5 = {"celeba_official_partition": "d32c9cbf5e040fd4025c592c306e6668", "celeba_official_identity_ids": "32bd1bd63d3c78cd57e08160ec5ed1e2"}
EXPECTED_METADATA_MD5.update({name + "_public_download": checksum for name, checksum in list(EXPECTED_METADATA_MD5.items())})
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_FILE_BYTES = 5 * 1024 * 1024


def fetch_metadata(output):
    output.mkdir(parents=True, exist_ok=True)
    total = 0
    records = []
    existing_manifest = output / "download_manifest.json"
    existing = {row["name"]: row for row in json.loads(existing_manifest.read_text(encoding="utf-8"))["records"]} if existing_manifest.exists() else {}
    queue = list(METADATA.items())
    for name, url in queue:
        record = {"name": name, "url": url, "role": "metadata_only"}
        old = existing.get(name)
        if old and old.get("url") == url:
            if old.get("status") != "downloaded" or (Path(old["path"]).exists() and sha256(old["path"]) == old["sha256"]):
                records.append(old)
                total += old.get("bytes", 0)
                continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FaceScoreLab-course-metadata-audit/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                limit = min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total)
                declared = response.headers.get("Content-Length")
                content_type = response.headers.get("Content-Type", "")
                if "image/" in content_type or "application/zip" in content_type:
                    raise ValueError("Refusing image/archive response before reading")
                if declared and int(declared) > limit:
                    raise ValueError("Metadata declared size exceeds bound")
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError("Metadata response exceeds bound; discarded")
                content_type = response.headers.get("Content-Type", "")
                if "image/" in content_type or data[:4] == b"PK\x03\x04":
                    raise ValueError("Refusing image/archive download")
                record.update({"final_url": response.url, "content_type": content_type})
            text = data.decode("utf-8")
            if name in EXPECTED_METADATA_MD5 and hashlib.md5(data).hexdigest() != EXPECTED_METADATA_MD5[name]:
                raise ValueError("Metadata MD5 does not match torchvision's recorded source; may be a login/quota page")
            target = output / (name + ".txt")
            target.write_bytes(data)
            total += len(data)
            record.update({"status": "downloaded", "path": str(target), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            record["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
            if name == "celebamask_hq_official_tree":
                tree = json.loads(text)
                for entry in tree.get("tree", []):
                    path = entry.get("path", "")
                    if entry.get("type") == "blob" and entry.get("size", MAX_FILE_BYTES + 1) <= MAX_FILE_BYTES and re.search(r"(mapping|partition|identity).*\.txt$", path, re.I):
                        queue.append(("official_" + re.sub(r"[^a-zA-Z0-9]", "_", path), "https://raw.githubusercontent.com/switchablenorms/CelebAMask-HQ/master/" + urllib.parse.quote(path)))
        except Exception as error:
            record.update({"status": "unavailable", "error": f"{type(error).__name__}: {error}"})
        records.append(record)
    write_json(output / "download_manifest.json", {"total_bytes": total, "limit_bytes": MAX_TOTAL_BYTES,
                                                    "downloaded_images": 0, "records": records})
    return records


def audit_official_partitions(metadata_by_name, output):
    partition = metadata_by_name.get("celeba_official_partition", {})
    identity = metadata_by_name.get("celeba_official_identity_ids", {})
    if partition.get("status") != "downloaded":
        partition = metadata_by_name.get("celeba_official_partition_public_download", partition)
    if identity.get("status") != "downloaded":
        identity = metadata_by_name.get("celeba_official_identity_ids_public_download", identity)
    result = {"status": "metadata_unavailable", "images_downloaded": 0,
              "note": "No claim of identity-disjoint official splits without the actual mapping files."}
    if partition.get("status") == "downloaded":
        rows = [line.split() for line in Path(partition["path"]).read_text().splitlines() if line.strip()]
        assignments = {image: int(split) for image, split in rows}
        if len(rows) != len(assignments) or any(split not in (0, 1, 2) for split in assignments.values()):
            raise ValueError("Malformed official partition")
        result.update({"status": "partition_verified", "n_images": len(assignments),
                       "partition_counts": {str(split): sum(value == split for value in assignments.values()) for split in (0, 1, 2)}})
        if identity.get("status") == "downloaded":
            identities = dict(line.split() for line in Path(identity["path"]).read_text().splitlines() if line.strip())
            grouped = {}
            for image, split in assignments.items():
                if image not in identities:
                    raise ValueError("Official image lacks identity metadata")
                grouped.setdefault(identities[image], set()).add(split)
            result.update({"status": "identity_partition_audited", "n_identities": len(grouped),
                           "identities_spanning_official_splits": sum(len(splits) > 1 for splits in grouped.values()),
                           "note": "Counts only, no raw identity IDs are disclosed in this report. This says nothing about third-party checkpoint pretraining."})
    write_json(output / "celeba_official_split_audit.json", result)
    return result


def local_cache_scan(roots=None):
    if roots is None:
        work_root = Path(os.environ.get("FACESCORELAB_WORK_DIR") or PROJECT / "training/work").expanduser()
        roots = [work_root / "datasets", PROJECT / "training/deep/downloads",
                 PROJECT / "training", Path.home() / ".cache/huggingface/hub"]
    reports = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        report = {"root": str(root), "exists": root.exists(), "max_depth": 0 if root.name == "Google" else 5,
                  "matching_paths": [], "image_count": 0, "visited_files": 0}
        if root.is_dir():
            for directory, subdirs, files in os.walk(root):
                depth = len(Path(directory).relative_to(root).parts)
                subdirs[:] = [name for name in subdirs if name not in {"site-packages", "Lib", "lib", ".git", "node_modules", ".venv", "venv", "__pycache__", "optimization"}] if depth < report["max_depth"] else []
                for name in files:
                    report["visited_files"] += 1
                    if report["visited_files"] > 100000:
                        raise ValueError("Bounded cache scan exceeded file cap")
                    path = Path(directory) / name
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        report["image_count"] += 1
                    if re.search(r"celeba|celebamask|list_eval_partition|identity_CelebA|list_attr_celeba", str(path), re.I):
                        if len(report["matching_paths"]) < 100:
                            report["matching_paths"].append({"path": str(path), "bytes": path.stat().st_size})
        reports.append(report)
    return {"bounded_roots": reports, "conclusion": "No whole-disk scan. A lack of hits means not found in these roots, not absent everywhere."}


def model_registry():
    assets = PROJECT / "app/src/main/assets"
    deep = json.loads((assets / "deep_models.json").read_text(encoding="utf-8-sig"))
    fpem = json.loads((assets / "fpem_model.json").read_text(encoding="utf-8-sig"))
    models = []
    definitions = [
        ("beauty_regression", "beauty_mobilenetv3_small.onnx", deep["cnn"].get("model_version"),
         deep["cnn"].get("initialization"), "SCUT-FBP5500 official fold 1; 3960 train, 440 validation, 1100 test",
         "RGB face crop; 224x224 bilinear resize; /255 then ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]; NCHW",
         ["Own official-fold split and checkpoint selection recorded", "Pretraining stated as ImageNet1K V1 only"],
         ["SCUT identity overlap cannot be excluded", "Exact ImageNet person-level overlap not established", "Independent real-phone accuracy unmeasured"]),
        ("visible_attributes", "face_attributes.onnx", deep["attributes"].get("model_version"), deep["attributes"].get("original_source"),
         "Training dataset and per-image partition not present in local model documentation",
         deep["attributes"].get("preprocessing") + " Mean/std normalization is baked into model graph; do not normalize twice.",
         ["Five independent binary outputs documented", "BSD-3-Clause code/model distribution declaration retained"],
         ["Training identities, image list, split and CelebA overlap unknown", "Independent attribute accuracy unmeasured"]),
        ("face_segmentation", "face_parsing_mask.onnx", "bisenet-resnet18-19-fp16-storage", deep["segmentation"].get("source"),
         "CelebAMask-HQ declared; exact checkpoint training/validation/test membership not established",
         "RGB face crop; 512x512 bilinear resize; /255 then ImageNet mean/std; NCHW; uint8 class-ID mask",
         ["CelebAMask-HQ provenance declared", "Code MIT license retained", "Numerical conversion agreement checked on examples, not segmentation ground truth"],
         ["Exact training image list and identity overlap unknown", "Independent annotated mask accuracy unmeasured", "Code license is not the image dataset license"]),
        ("visible_mouth_actions", "face_landmarker.task", deep["visibleMouthActions"].get("model_version"), deep["visibleMouthActions"].get("source"),
         "Google MediaPipe face landmark/blendshape model; exact training image and identity membership unavailable locally",
         "Full RGB bitmap supplied to MediaPipe; internal preprocessing; largest face chosen; smile=mean mouthSmileLeft/right; jawOpen separately",
         ["Actions are visible coefficients, not calibrated probabilities", "Cross-runtime coefficients checked on examples"],
         ["Training-image overlap unknown", "Label-level accuracy and threshold calibration unmeasured"]),
        ("fpem_reference", "fpem_mobile.onnx", "upstream-fpem-port", fpem.get("source"),
         "Upstream pretrained checkpoint; exact training identity/split membership not established locally",
         fpem.get("input_preprocessing"),
         ["Checkpoint hash and numerical port checks recorded", "Not a reproduced training/benchmark result"],
         ["Exact checkpoint training sample membership unknown", "Independent local accuracy unmeasured"]),
    ]
    for name, asset, version, source, training, preprocess, known, unknown in definitions:
        path = assets / asset
        expected = deep.get("assets", {}).get(asset, {}).get("sha256")
        if asset == "face_landmarker.task":
            expected = deep["visibleMouthActions"].get("asset_sha256")
        actual = sha256(path) if path.exists() else None
        models.append({"module": name, "asset": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None,
                       "sha256": actual, "recorded_sha256": expected, "hash_matches_record": actual == expected if expected else None,
                       "version": version, "source": source, "training_data_statement": training,
                       "preprocessing": preprocess, "known": known, "unknown": unknown,
                       "pretraining_overlap_status": "unknown_not_proven_disjoint"})
    return {"audit_date": "2026-11-13", "read_only_sources": [str(assets / "deep_models.json"), str(assets / "fpem_model.json"),
            str(PROJECT / "third_party/FaceAttribNet/README.md"), str(PROJECT / "app/src/main/java/com/example/facescorelab/OnnxLocalEngine.java")],
            "models": models, "claims_policy": "This registry distinguishes documented provenance from independent verification; conversion parity is not label accuracy."}


def templates(output):
    output.mkdir(parents=True, exist_ok=True)
    def empty_csv(name, fields):
        path = output / name
        if not path.exists():
            write_csv(path, [], fields)
    annotation_fields = ["image_id", "group_id", "split", "dataset_role", "attribute", "reviewer_1_target", "reviewer_2_target", "target", "unknown_reason", "label_source", "reviewer_1", "reviewer_2", "review_status", "adjudicator", "label_version", "source_reference"]
    empty_csv("attribute_annotations.csv", annotation_fields)
    empty_csv("attribute_predictions.csv", ["image_id", "attribute", "score", "prediction_status", "failure_reason", "model_version"])
    empty_csv("segmentation_annotations.csv", ["image_id", "group_id", "split", "dataset_role", "mask_path", "mask_schema", "label_source", "reviewer_1", "reviewer_2", "review_status", "adjudicator", "label_version", "source_reference"])
    empty_csv("segmentation_predictions.csv", ["image_id", "mask_path", "mask_schema", "model_version"])
    empty_csv("identity_import_private.csv", ["image_id", "source_identity_id"])
    empty_csv("review_disagreements.csv", ["image_id", "attribute_or_region", "reviewer_1", "reviewer_1_decision", "reviewer_2", "reviewer_2_decision", "reason", "adjudicator", "resolution", "label_version"])
    write_json(output / "fixed_demo_thresholds.json", {"version": "existing_app_demo_v3_1_not_calibrated", "selection_split": "fixed_uncalibrated_demo",
                "thresholds": {name: 0.25 if name == "mouth_open" else 0.5 for name in ATTRIBUTES},
                "note": "Existing presentation thresholds, not fitted optimal values or calibrated probabilities."})
    source_regions = [None, "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat"]
    write_json(output / "segmentation_classes.json", {"schema": MASK_SCHEMA, "ignore_id": 255,
                "classes": [{"id": i, "name": name, "celebamask_region_key": source_regions[i]} for i, name in enumerate(APP_CLASSES)],
                "warning": "Upstream merged-mask numeric IDs may differ. Map by semantic region name, not by assumed numeric order. Do not auto-resize labels."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append",
                        help="Bounded local scan root; repeat for multiple roots. Defaults to project training/work datasets, training, model downloads and the user Hugging Face cache")
    parser.add_argument("--fetch-public-metadata", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "model_provenance_registry.json", model_registry())
    write_json(args.output / "local_cache_audit.json", local_cache_scan(args.source_root))
    templates(args.output / "annotation_templates")
    metadata_manifest = args.output / "official_metadata/download_manifest.json"
    downloaded = fetch_metadata(args.output / "official_metadata") if args.fetch_public_metadata else (json.loads(metadata_manifest.read_text(encoding="utf-8"))["records"] if metadata_manifest.exists() else [])
    metadata_by_name = {row["name"]: row for row in downloaded}
    split_audit = audit_official_partitions(metadata_by_name, args.output)
    source_links = {}
    for name in ("celeba_official_page", "celebamask_hq_official_readme"):
        row = metadata_by_name.get(name)
        if row and row.get("status") == "downloaded":
            content = Path(row["path"]).read_text(encoding="utf-8")
            source_links[name] = sorted(set(re.findall(r'https?://[^\s\"<>\)]+', content)))
    registry = {"audit_date": "2026-11-13", "full_images_downloaded_by_this_audit": 0,
                "dataset_state": "candidate_metadata_only_not_training_ready",
                "candidates": [
                    {"name": "CelebA", "official_page": METADATA["celeba_official_page"], "intended_labels": ["Eyeglasses", "Smiling", "Mouth_Slightly_Open"],
                     "unsupported_direct_targets": ["mask", "left_eye_open", "right_eye_open", "ordinary_versus_sunglasses"],
                     "license_status": "Official Agreement verified in captured page: non-commercial research only; no redistribution outside internal use at one organization; identities on request",
                     "license_evidence": metadata_by_name.get("celeba_official_page", {}).get("path"),
                     "complete_local_dataset_verified": False, "full_dataset_bytes_verified": None,
                     "source_download_links": [url for url in source_links.get("celeba_official_page", []) if "drive.google.com" in url or "pan.baidu.com" in url],
                     "image_archive_reference": {"filename": "img_align_celeba.zip", "google_drive_id": "0B7EVK8r0v71pZjFTYXZWM3FlRnM", "md5_from_torchvision": "00d2c5bc6d35e252742224ab0c1e8fcb", "size_bytes_verified": None, "direct_download_verified": False},
                     "overlap": "May overlap existing pretraining; exact lists required. CelebAMask-HQ derives from CelebA."},
                    {"name": "CelebAMask-HQ", "official_page": "https://github.com/switchablenorms/CelebAMask-HQ",
                     "intended_task": "19-class face-component masks", "license_status": "Official Dataset Agreement verified: non-commercial research only; no redistribution outside internal use at one organization",
                     "license_evidence": metadata_by_name.get("celebamask_hq_official_readme", {}).get("path"),
                     "complete_local_dataset_verified": False, "full_dataset_bytes_verified": None,
                     "source_download_links": [url for url in source_links.get("celebamask_hq_official_readme", []) if "drive.google.com" in url or "pan.baidu.com" in url],
                     "image_archive_reference": {"google_drive_id": "1badu11NqxGf6qM3PTTooQDJvQbejgbTv", "size_bytes_verified": None, "direct_download_verified": False},
                     "overlap": "Existing BiSeNet trained on this dataset; exact split unknown. Public samples cannot automatically be called unseen."}],
                "metadata": downloaded,
                "official_split_audit": split_audit,
                "next_gate": "Select authorized image source, verify terms and sizes, establish immutable groups/splits, and obtain independent labels before accuracy evaluation."}
    for candidate in registry["candidates"]:
        if not candidate.get("license_evidence"):
            candidate["license_status"] = "not_verified_no_official_metadata_captured"
    write_json(args.output / "dataset_registry.json", registry)
    print(json.dumps({"output": str(args.output), "models_audited": 5, "metadata_downloaded": sum(row.get("status") == "downloaded" for row in downloaded),
                      "images_downloaded": 0}))


if __name__ == "__main__":
    main()
