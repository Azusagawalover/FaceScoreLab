"""Extract the preregistered official CelebA test subset AFTER candidate freeze.

Offline only. Never call this command before the visible-attribute candidate lock
exists. It does not train, tune thresholds, crop faces, or compute accuracy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import zlib

import numpy as np

from prepare_celeba_subset import EXPECTED, SAFE_ATTRIBUTES, hashes

HELDOUT_COUNT = 2000
HELDOUT_SEED = 20261113
OFFICIAL_COUNTS = {0: 162770, 1: 19867, 2: 19962}
LOCK_FORMAT = "visible-attributes-mnv3-v1"
FIELDS = ["name", "split", "image_path", *SAFE_ATTRIBUTES]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256(path):
    return hashes(path)["sha256"]


def validate_lock(path):
    # This must finish before dataset metadata or test labels are opened.
    lock = read_json(path)
    payload = {key: value for key, value in lock.items() if key != "lock_id"}
    if digest(payload) != lock.get("lock_id") or lock.get("format") != LOCK_FORMAT:
        raise ValueError("Invalid or modified visible-attribute candidate lock")
    selection = lock.get("heldout_selection", {})
    if selection != {"pool": "official CelebA test only", "sample_count": HELDOUT_COUNT,
                      "seed": HELDOUT_SEED, "identity_independent": False}:
        raise ValueError("Lock must preregister official test / 2000 images / seed 20261113 / image-level only")
    directory = Path(lock["run"]).resolve(strict=True)
    for filename, key in (("best.pt", "checkpoint_sha256"), ("visible_attributes.onnx", "onnx_sha256"),
                          ("android_metadata.json", "android_metadata_sha256")):
        target = (directory / filename).resolve(strict=True)
        if not target.is_relative_to(directory) or sha256(target) != lock[key]:
            raise ValueError(f"Frozen candidate artifact changed: {filename}")
    metadata = read_json(directory / "android_metadata.json")
    if metadata["thresholds"] != lock["thresholds"] or metadata["preprocessing"] != lock["preprocessing"]:
        raise ValueError("Lock thresholds/preprocessing differ from the frozen Android artifact")
    if metadata.get("model_sha256") != lock["onnx_sha256"]:
        raise ValueError("Frozen Android model identity differs from the ONNX artifact")
    return lock


def safe_path(root, *parts):
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path escapes its resolved root")
    return path


def official_partitions(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError("Malformed original split row")
            name, split = fields
            if not re.fullmatch(r"[0-9]{6}\.jpg", name) or name in result or split not in ("0", "1", "2"):
                raise ValueError("Invalid or duplicate original split filename")
            result[name] = int(split)
    if {split: sum(value == split for value in result.values()) for split in OFFICIAL_COUNTS} != OFFICIAL_COUNTS:
        raise ValueError("Original CelebA partition counts differ")
    return result


def selected_labels(path, selected, partitions):
    labels, seen = {}, set()
    with path.open(encoding="utf-8") as stream:
        declared = int(stream.readline())
        columns = stream.readline().split()
        if len(columns) != 40 or len(set(columns)) != 40:
            raise ValueError("Unexpected original attribute schema")
        offsets = [columns.index(name) + 1 for name in SAFE_ATTRIBUTES]
        for line in stream:
            values = line.split()
            if len(values) != 41 or values[0] not in partitions or values[0] in seen:
                raise ValueError("Original attribute filename/partition mismatch")
            name = values[0]
            seen.add(name)
            if name in selected:
                targets = [int(values[index]) for index in offsets]
                if any(value not in (-1, 1) for value in targets):
                    raise ValueError("Expected original -1/+1 selected labels")
                labels[name] = {label: int(value == 1) for label, value in zip(SAFE_ATTRIBUTES, targets)}
    if declared != len(partitions) or seen != set(partitions) or set(labels) != selected:
        raise ValueError("Incomplete original attribute/split mapping")
    return labels


def manifest_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def prepare(lock_path, dataset_root):
    lock = validate_lock(lock_path)
    root = Path(dataset_root).resolve(strict=True)
    checks = {}
    for name, expected in EXPECTED.items():
        actual = hashes(safe_path(root, name))
        if actual["md5"] != expected["md5"] or actual["bytes"] != expected["bytes"]:
            raise ValueError(f"Original file MD5/size mismatch: {name}")
        checks[name] = actual
    partitions = official_partitions(safe_path(root, "list_eval_partition.txt"))
    pool = sorted(name for name, split in partitions.items() if split == 2)
    names = sorted(np.random.default_rng(HELDOUT_SEED).choice(pool, HELDOUT_COUNT, replace=False).tolist())
    selected = set(names)
    if len(selected) != HELDOUT_COUNT:
        raise ValueError("Incorrect selected sample count")
    development_path = safe_path(root, "dev.csv")
    with development_path.open(encoding="utf-8-sig", newline="") as stream:
        development = list(csv.DictReader(stream))
    development_names = {row["name"] for row in development}
    if len(development_names) != len(development) or selected & development_names:
        raise ValueError("Development duplicates or held-out/development overlap")
    for row in development:
        if row["split"] not in ("train", "validation") or partitions.get(row["name"]) != {"train": 0, "validation": 1}[row["split"]]:
            raise ValueError("Development manifest does not follow official splits")
    manifest_path = safe_path(root, "heldout.csv")
    protocol_path = safe_path(root, "heldout_protocol.json")
    directory = safe_path(root, "images", "test")
    invariant = {"version": "celeba-fixed-heldout-v1", "seed": HELDOUT_SEED,
                 "counts": {"test": HELDOUT_COUNT}, "selected_names": {"test": names},
                 "lock_id": lock["lock_id"], "candidate_onnx_sha256": lock["onnx_sha256"],
                 "official_source_checks": checks, "development_manifest_sha256": sha256(development_path)}
    existing = None
    if protocol_path.exists() or manifest_path.exists():
        if not protocol_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Partial published held-out outputs exist; refusing to overwrite")
        existing = read_json(protocol_path)
        if any(existing.get(key) != value for key, value in invariant.items()):
            raise FileExistsError("Existing held-out output belongs to another lock/protocol")
    if directory.exists():
        files = list(directory.iterdir())
        if any(not path.is_file() or path.name not in selected for path in files):
            raise FileExistsError("Unexpected files in the fixed held-out image directory")
    # Selection is determined exclusively by the original split and frozen seed.
    labels = selected_labels(safe_path(root, "list_attr_celeba.txt"), selected, partitions)
    rows, image_hashes = [], {}
    extracted_bytes = 0
    with zipfile.ZipFile(safe_path(root, "img_align_celeba.zip")) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("Duplicate member names in original archive")
        for name in names:
            member = "img_align_celeba/" + name
            info = archive.getinfo(member)
            if info.is_dir() or info.file_size <= 0:
                raise ValueError("Selected archive member is not a nonempty image")
            path = safe_path(root, "images", "test", name)
            if path.exists():
                content = path.read_bytes()
                if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                    raise ValueError(f"Existing held-out image differs from original: {name}")
            elif existing:
                raise FileNotFoundError("Published held-out image is missing; refusing to silently replace it")
            else:
                content = archive.read(member)  # zipfile verifies the member CRC.
                directory.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(content)
            image_hashes[name] = hashlib.sha256(content).hexdigest()
            extracted_bytes += len(content)
            rows.append({"name": name, "split": "test", "image_path": str(path), **labels[name]})
    raw = manifest_bytes(rows)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    if existing:
        if manifest_path.read_bytes() != raw or existing.get("manifest_sha256") != manifest_hash or existing.get("image_sha256") != image_hashes:
            raise ValueError("Published held-out manifest/images changed")
    else:
        protocol = {**invariant, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "rng": "numpy.random.default_rng PCG64; sample sorted official test pool without replacement",
                    "sampling_note": "No class/error stratification or model predictions used to choose images",
                    "name_list_sha256": hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest(),
                    "manifest_sha256": manifest_hash, "image_sha256": image_hashes,
                    "extracted_image_bytes": extracted_bytes, "attributes": SAFE_ATTRIBUTES,
                    "label_encoding": "0=known negative; 1=known positive; original -1 mapped to 0",
                    "identity_independent": False,
                    "split_level": "Official CelebA image-level test pool; identity labels not acquired",
                    "limitations": "Public benchmark only; pretraining overlap of upstream face detector remains unknown",
                    "license": "Original CelebA noncommercial research and same-organization terms; no redistribution",
                    "network_access_performed": False, "model_training_performed": False,
                    "threshold_tuning_performed": False, "image_preprocessing_performed": False}
        # Exclusive creation preserves existing artifacts. A rare interrupted pair is
        # deliberately rejected on rerun rather than overwriting published evidence.
        with manifest_path.open("xb") as stream:
            stream.write(raw)
        with protocol_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return {"status": "reused_verified_same_protocol" if existing else "fixed_heldout_prepared_after_lock",
            "count": len(rows), "seed": HELDOUT_SEED, "lock_id": lock["lock_id"],
            "manifest": str(manifest_path), "protocol": str(protocol_path),
            "manifest_sha256": manifest_hash, "identity_independent": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.lock, args.dataset_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
