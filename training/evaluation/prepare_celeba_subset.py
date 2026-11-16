"""Acquire verified CelebA image/visible-label files for a bounded local course experiment.

No identity file is fetched. No model training, crop, inference, or test-image extraction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
import zlib

import numpy as np

from evaluation_common import sha256, write_csv, write_json

PROJECT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(os.environ.get("FACESCORELAB_WORK_DIR") or PROJECT / "training/work").expanduser()
EXPECTED = {
    "list_eval_partition.txt": {"bytes": 2836386, "md5": "d32c9cbf5e040fd4025c592c306e6668"},
    "list_attr_celeba.txt": {"bytes": 26721026, "md5": "75e246fa4810816ffd6ee81facbd244c"},
    "img_align_celeba.zip": {"bytes": 1443490838, "md5": "00d2c5bc6d35e252742224ab0c1e8fcb"},
}
TOTAL_BUDGET = 2 * 1024**3
SAFE_ATTRIBUTES = ["Smiling", "Eyeglasses", "Mouth_Slightly_Open"]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "bytes": Path(path).stat().st_size}


def quarantine(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Refusing quarantine outside the dataset directory")
    target = path.with_name(path.name + ".rejected-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if not target.is_relative_to(root):
        raise ValueError("Quarantine destination escaped dataset directory")
    path.replace(target)
    return str(target)


def download_verified(name, root, revision, expected_sha, budget, workers=1):
    expected = EXPECTED[name]
    destination = root / name
    part = destination.with_name(name + ".part")
    state_path = destination.with_name(name + ".transfer.json")
    url = f"https://huggingface.co/datasets/Yuehao/celeba/resolve/{revision}/{name}"
    if destination.exists():
        actual = hashes(destination)
        if actual["bytes"] == expected["bytes"] and actual["md5"] == expected["md5"] and (not expected_sha or actual["sha256"] == expected_sha):
            return {"file": str(destination), "source": url, "reused_verified_file": True, **actual}
        rejected = quarantine(destination, root)
        raise ValueError(f"Existing file hash mismatch; quarantined at {rejected}")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"source": url, "expected": expected, "etag": None}
    if state.get("source") != url or state.get("expected") != expected:
        raise ValueError("Transfer provenance changed; refusing resume")
    if part.exists() and part.stat().st_size and not state.get("etag"):
        raise ValueError("Partial file lacks a stable ETag record; refusing unsafe resume")
    if workers > 1 and part.exists() and part.stat().st_size < expected["bytes"]:
        from parallel_public_download import transfer_ranges
        transfer_ranges(url, part, root, state, state_path, expected["bytes"], budget, TOTAL_BUDGET, atomic_json, workers)
    last_progress = 0.
    while not part.exists() or part.stat().st_size < expected["bytes"]:
        start = part.stat().st_size if part.exists() else 0
        if start > expected["bytes"]:
            raise ValueError("Partial file exceeds expected size")
        headers = {"User-Agent": "FaceScoreLab-bounded-research-dataset-preparation/1.0", "Accept-Encoding": "identity"}
        if start:
            headers.update({"Range": f"bytes={start}-", "If-Range": state["etag"]})
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as response:
                response_etag = response.headers.get("ETag")
                if start:
                    expected_range = f"bytes {start}-{expected['bytes'] - 1}/{expected['bytes']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Resume Range ignored or changed; no response body accepted")
                    if response_etag != state["etag"]:
                        raise ValueError("ETag changed; refusing to concatenate different objects")
                elif response.status != 200:
                    raise ValueError(f"Expected complete public 200 response, got {response.status}")
                declared = response.headers.get("Content-Length")
                if not declared or int(declared) != expected["bytes"] - start:
                    raise ValueError("Content-Length does not match the verified repository file size")
                if not response_etag or response_etag.startswith("W/"):
                    raise ValueError("A strong ETag is required for safe resume")
                state.update({"etag": response_etag, "http_status": response.status, "status": "downloading"})
                atomic_json(state_path, state)
                with part.open("ab" if start else "wb") as handle:
                    while True:
                        remaining = TOTAL_BUDGET - budget["accounted_bytes"]
                        if remaining <= 0:
                            raise ValueError("The 2 GiB download budget has been exhausted")
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        budget["accounted_bytes"] += len(chunk)
                        atomic_json(root / "download_budget.json", budget)
                        if handle.tell() + len(chunk) > expected["bytes"]:
                            raise ValueError("Response exceeds declared size")
                        handle.write(chunk)
                        current = time.monotonic()
                        if current - last_progress >= 15:
                            handle.flush()
                            state["bytes_on_disk"] = handle.tell()
                            atomic_json(state_path, state)
                            print(json.dumps({"file": name, "bytes": handle.tell(), "total": expected["bytes"], "percent": round(100 * handle.tell() / expected["bytes"], 1)}), flush=True)
                            last_progress = current
                if part.stat().st_size != expected["bytes"]:
                    raise ConnectionError("Incomplete public response; saved bytes can resume with the same ETag")
        except (TimeoutError, ConnectionError, urllib.error.URLError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                raise RuntimeError("Authentication/access restriction encountered; no bypass attempted") from None
            failures = int(state.get("network_failures", 0)) + 1
            state.update({"network_failures": failures, "status": "partial_network_failure", "error_type": type(error).__name__})
            atomic_json(state_path, state)
            if failures >= 3:
                raise RuntimeError("Three network failures; partial file preserved for an explicit later resume") from None
            time.sleep(2)
    actual = hashes(part)
    if actual["bytes"] != expected["bytes"] or actual["md5"] != expected["md5"] or (expected_sha and actual["sha256"] != expected_sha):
        rejected = quarantine(part, root)
        state.update({"status": "hash_mismatch_rejected", "actual": actual, "quarantine": rejected})
        atomic_json(state_path, state)
        raise ValueError(f"Downloaded {name} failed original checksum; quarantined and not used")
    part.replace(destination)
    state.update({"status": "verified", "actual": actual})
    atomic_json(state_path, state)
    print(json.dumps({"verified": name, **actual}), flush=True)
    return {"file": str(destination), "source": url, "reused_verified_file": False, **actual}


def create_subset(root, seed=42):
    partitions = {}
    with (root / "list_eval_partition.txt").open() as handle:
        for line in handle:
            name, split = line.split()
            if not re.fullmatch(r"\d{6}\.jpg", name) or name in partitions or split not in ("0", "1", "2"):
                raise ValueError("Unexpected official partition format")
            partitions[name] = int(split)
    pools = {value: sorted(name for name, split in partitions.items() if split == value) for value in (0, 1, 2)}
    if [len(pools[value]) for value in (0, 1, 2)] != [162770, 19867, 19962]:
        raise ValueError("Official split counts do not match")
    rng = np.random.default_rng(seed)
    train = sorted(rng.choice(pools[0], size=20000, replace=False).tolist())
    validation = sorted(rng.choice(pools[1], size=2000, replace=False).tolist())
    selected = set(train) | set(validation)
    if len(selected) != 22000 or selected.intersection(pools[2]):
        raise ValueError("Subset overlap")
    labels, test_metadata = {}, []
    with (root / "list_attr_celeba.txt").open() as handle:
        declared = int(handle.readline())
        columns = handle.readline().split()
        offsets = [columns.index(name) for name in SAFE_ATTRIBUTES]
        rows = 0
        seen = set()
        for line in handle:
            values = line.split()
            name = values[0]
            if len(values) != 41 or name not in partitions or name in seen:
                raise ValueError("Attribute row/name mismatch")
            seen.add(name)
            targets = [int(values[index + 1]) for index in offsets]
            if any(value not in (-1, 1) for value in targets):
                raise ValueError("Original attribute values must be -1/+1")
            safe = {attribute: int(value == 1) for attribute, value in zip(SAFE_ATTRIBUTES, targets)}
            if name in selected:
                labels[name] = safe
            elif partitions[name] == 2:
                test_metadata.append({"image_id": name, "official_split": "test", "image_extracted": False, **safe})
            rows += 1
    if declared != 202599 or rows != declared or seen != set(partitions):
        raise ValueError("Full original image/attribute/split name mapping does not match")
    manifest = []
    output_bytes = 0
    with zipfile.ZipFile(root / "img_align_celeba.zip") as archive:
        for split, names in (("train", train), ("validation", validation)):
            directory = root / "images" / split
            directory.mkdir(parents=True, exist_ok=True)
            for name in names:
                member = "img_align_celeba/" + name
                info = archive.getinfo(member)
                destination = directory / name
                if not destination.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Image extraction destination escaped dataset root")
                if destination.exists():
                    content = destination.read_bytes()
                    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                        raise ValueError(f"Existing extracted file differs: {destination}")
                else:
                    content = archive.read(member)  # zipfile verifies the member CRC.
                    destination.write_bytes(content)
                output_bytes += len(content)
                manifest.append({"image_id": name, "path": str(destination), "split": split, "official_split": split,
                                 "group_id": "image_" + name, "group_id_origin": "image_level_only",
                                 "source": "CelebA_original_files_verified_MD5_via_Yuehao_mirror",
                                 "dataset_role": "official_annotation_benchmark_subset", "sha256": hashlib.sha256(content).hexdigest(),
                                 "source_archive_member": member, **labels[name]})
    write_csv(root / "subset_seed42.csv", manifest)
    write_csv(root / "dev.csv", [{"name": row["image_id"], "split": row["split"], "image_path": row["path"],
                                 **{attribute: row[attribute] for attribute in SAFE_ATTRIBUTES}} for row in manifest])
    write_csv(root / "official_test_metadata_only.csv", test_metadata)
    write_json(root / "subset_protocol.json", {"seed": seed, "rng": "numpy.random.default_rng PCG64; sample sorted official train pool then sorted official validation pool",
               "counts": {"train": len(train), "validation": len(validation), "test_metadata": len(test_metadata), "test_images_extracted": 0},
               "selected_names": {"train": train, "validation": validation}, "extracted_image_bytes": output_bytes,
               "identity_independent": False, "split_level": "official image-level splits; no identity file downloaded or identity-disjoint claim",
               "pretraining_overlap": "unknown for existing FaceAttribNet/BiSeNet/MediaPipe; this preparation does not establish an unseen model benchmark",
               "original_full_label_md5_verified": True, "full_image_label_partition_mapping_verified": True,
               "sampling_note": "Uniform random image sampling, not label/identity stratification. Threshold selection uses validation only."})
    write_json(root / "label_schema.json", {"attributes": SAFE_ATTRIBUTES, "source_values": [-1, 1], "prepared_values": [0, 1],
               "mapping": "-1 -> 0; +1 -> 1", "target_kind": "original CelebA visible attribute labels",
               "notes": {"Eyeglasses": "Original benchmark attribute; do not automatically interpret as ordinary-vs-sunglasses subclassification.",
                         "Smiling": "Visible smiling label, not emotional state; separate from continuous MediaPipe coefficient.",
                         "Mouth_Slightly_Open": "Original visible mouth-opening label; not the continuous jawOpen coefficient."},
               "excluded_from_processing": "All other CelebA attribute targets and identity labels", "image_preprocessing_performed": False})
    return {"train_images": len(train), "validation_images": len(validation), "test_images_extracted": 0,
            "extracted_image_bytes": output_bytes, "manifest": str(root / "subset_seed42.csv"), "development_csv": str(root / "dev.csv")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT,
                        help="Work directory containing data-audit; defaults to FACESCORELAB_WORK_DIR or project training/work")
    parser.add_argument("--output", type=Path,
                        help="Dataset directory inside <run-root>/datasets; defaults to <run-root>/datasets/celeba")
    parser.add_argument("--stage", choices=["labels", "all"], default="labels")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    root = (args.output.expanduser() if args.output is not None else run_root / "datasets/celeba").resolve()
    if not root.is_relative_to((run_root / "datasets").resolve()):
        raise ValueError("Output must stay inside the authorized run datasets directory")
    root.mkdir(parents=True, exist_ok=True)
    audit = run_root / "data-audit"
    metadata = audit / "availability_metadata"
    info = json.loads((metadata / "Yuehao__celeba__repository_info.json").read_text())
    if info.get("gated") or info.get("private"):
        raise ValueError("Repository is not publicly accessible; no authentication attempted")
    revision = info["sha"]
    tree = json.loads((metadata / "Yuehao__celeba__root_tree.txt").read_text())
    files = {row["path"]: row for row in tree if row["type"] == "file"}
    for name, expected in EXPECTED.items():
        if files[name]["size"] != expected["bytes"]:
            raise ValueError("Repository file size differs from audited source")
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    copies = {
        audit / "official_metadata/celeba_official_page.txt": provenance / "CelebA_official_page_and_agreement.html",
        metadata / "flwrlabs__celeba__LICENSE.txt": provenance / "CelebA_original_release_agreement.txt",
        metadata / "Yuehao__celeba__readme.txt": provenance / "mirror_README.txt",
        metadata / "Yuehao__celeba__repository_info.json": provenance / "mirror_revision.json",
        metadata / "Yuehao__celeba__root_tree.txt": provenance / "mirror_file_metadata.json",
        audit / "official_metadata/torchvision_celeba_loader.txt": provenance / "torchvision_original_expected_checksums.py.txt",
    }
    for source, destination in copies.items():
        if not destination.exists():
            shutil.copyfile(source, destination)
    license_text = (provenance / "CelebA_original_release_agreement.txt").read_text()
    if "non-commercial research purposes only" not in license_text or "same organization" not in license_text:
        raise ValueError("Expected original license conditions unavailable; stopping before data download")
    budget_path = root / "download_budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {
        "limit_bytes": TOTAL_BUDGET, "accounted_bytes": 5 * 1024 * 1024,
        "prior_metadata_conservative_allowance_bytes": 5 * 1024 * 1024,
        "note": "Includes a conservative 5 MiB allowance for earlier metadata traffic. No paid or authenticated sources."}
    partition_source = metadata / "Yuehao__celeba__list_eval_partition.txt"
    if hashes(partition_source)["md5"] != EXPECTED["list_eval_partition.txt"]["md5"]:
        raise ValueError("Official partition MD5 mismatch")
    if not (root / "list_eval_partition.txt").exists():
        shutil.copyfile(partition_source, root / "list_eval_partition.txt")
    checks = {"partition": hashes(root / "list_eval_partition.txt")}
    if checks["partition"]["md5"] != EXPECTED["list_eval_partition.txt"]["md5"]:
        raise ValueError("Existing dataset-directory partition differs from the original verified file")
    checks["attributes"] = download_verified("list_attr_celeba.txt", root, revision, files["list_attr_celeba.txt"].get("lfs", {}).get("oid"), budget)
    startup = {"license": "Original CelebA non-commercial research/internal same-organization terms retained; no redistribution of images",
               "identity_data_requested_or_downloaded": False, "source_public_no_login": True,
               "repository_revision": revision, "original_partition_md5_passed": True,
               "original_attributes_md5_passed": checks["attributes"]["md5"] == EXPECTED["list_attr_celeba.txt"]["md5"],
               "zip_download_start_conditions_met": True, "checks": checks}
    write_json(root / "startup_conditions.json", startup)
    print(json.dumps({"startup_conditions_met": True, "stage": args.stage, "attributes_md5": checks["attributes"]["md5"]}), flush=True)
    if args.stage == "all":
        checks["images_archive"] = download_verified("img_align_celeba.zip", root, revision, files["img_align_celeba.zip"].get("lfs", {}).get("oid"), budget, args.workers)
        subset = create_subset(root)
        write_json(root / "preparation_result.json", {"status": "verified_original_files_and_fixed_subset_prepared", "checks": checks,
                   "subset": subset, "download_accounting": budget, "model_training_performed": False,
                   "image_preprocessing_performed": False, "independent_accuracy_claimed": False,
                   "identity_data_downloaded": False})
        print(json.dumps({"complete": True, "output": str(root), **subset}), flush=True)


if __name__ == "__main__":
    main()
