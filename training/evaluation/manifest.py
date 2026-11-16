"""Bounded image inventory and deterministic identity/duplicate-safe splits."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets

from PIL import Image, ImageOps

from evaluation_common import read_records, sha256, write_csv, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXCLUDED_DIRECTORIES = {".git", ".gradle", "build", "node_modules", "site-packages", ".venv", "venv", "__pycache__"}
SPLITS = ("train", "validation", "test")
KNOWN_GROUP_ORIGINS = {"official_identity_id", "participant_code"}


def inventory(root, source, role="unlabeled_candidate", max_files=100000, max_depth=8):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")
    records, failures = [], []
    for directory, subdirs, files in os.walk(root):
        depth = len(Path(directory).relative_to(root).parts)
        subdirs[:] = sorted(name for name in subdirs if name not in EXCLUDED_DIRECTORIES and not name.startswith(".")) if depth < max_depth else []
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if len(records) + len(failures) >= max_files:
                raise ValueError("Inventory file cap reached; choose a narrower root or explicit larger bound")
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as image:
                    orientation = int(image.getexif().get(274, 1))
                    corrected = ImageOps.exif_transpose(image).convert("RGB")
                    pixel_hash = hashlib.sha256(f"RGB:{corrected.width}x{corrected.height}:".encode() + corrected.tobytes()).hexdigest()
                    width, height = corrected.size
                records.append({"image_id": hashlib.sha256(f"{source}:{relative}".encode()).hexdigest()[:24],
                                "path": str(path), "source": source, "source_relative_path": relative,
                                "dataset_role": role, "group_id": "", "group_id_origin": "unknown",
                                "sha256": sha256(path), "decoded_pixel_sha256": pixel_hash,
                                "width": width, "height": height, "exif_orientation": orientation,
                                "duplicate_group_id": "", "split": "", "license_status": "not_verified"})
            except (OSError, ValueError) as error:
                failures.append({"path": str(path), "error": str(error)})
    return records, failures


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[second] = first


def _components(records):
    union = DisjointSet(len(records))
    seen = {}
    for index, row in enumerate(records):
        for key in ("group_id", "sha256", "decoded_pixel_sha256", "duplicate_group_id"):
            value = str(row.get(key, "")).strip()
            if not value:
                continue
            token = (key, value)
            if token in seen:
                union.union(index, seen[token])
            else:
                seen[token] = index
    result = {}
    for index in range(len(records)):
        result.setdefault(union.find(index), []).append(index)
    return list(result.values())


def validate_manifest(records, require_identity=True):
    if not records:
        raise ValueError("Manifest is empty")
    ids = [row.get("image_id") for row in records]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("image_id must be nonempty and unique")
    unknown = [row["image_id"] for row in records if not row.get("group_id") or row.get("group_id_origin") not in KNOWN_GROUP_ORIGINS]
    if require_identity and unknown:
        raise ValueError(f"{len(unknown)} image(s) lack a declared identity group; image-only isolation is a separate weaker protocol")
    bad_splits = [row["image_id"] for row in records if row.get("split") not in SPLITS]
    if bad_splits:
        raise ValueError(f"Unassigned/invalid splits: {bad_splits[:5]}")
    components = _components(records)
    for members in components:
        present = {records[index]["split"] for index in members}
        if len(present) > 1:
            raise ValueError(f"Identity or exact/declared derivative duplicate spans splits: {[records[i]['image_id'] for i in members[:5]]}")
    return {"n_images": len(records), "n_connected_groups": len(components), "unknown_identity_images": len(unknown),
            "identity_isolation_declared": not unknown, "identity_isolation_independently_verified": False,
            "counts": {split: sum(row["split"] == split for row in records) for split in SPLITS},
            "checks": "Identity declarations, exact-file hashes, EXIF-corrected exact-pixel hashes, declared derivative groups",
            "limitation": "No biometric identity matching; undeclared identities/near-duplicates and pretraining overlap remain unknown."}


def build_splits(records, seed="20261113", ratios=(0.7, 0.15, 0.15), previous=None,
                 allow_image_level=False, additions_train_only=False):
    records = [dict(row) for row in records]
    if not records:
        raise ValueError("Manifest is empty")
    ids = [row.get("image_id") for row in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("image_id must be unique and nonempty")
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three nonnegative ratios summing to 1 are required")
    previous = previous or []
    if additions_train_only and not previous:
        raise ValueError("Train-only additions require a frozen previous manifest")
    previous_by_id = {row["image_id"]: row for row in previous}
    if len(previous_by_id) != len(previous):
        raise ValueError("Duplicate IDs in previous manifest")
    if not set(previous_by_id).issubset(ids):
        raise ValueError("Previous images cannot silently disappear from a frozen manifest")
    for row in records:
        previous_row = previous_by_id.get(row["image_id"])
        if previous_row:
            for key in ("group_id", "group_id_origin", "sha256", "decoded_pixel_sha256", "duplicate_group_id"):
                if row.get(key, "") != previous_row.get(key, ""):
                    raise ValueError(f"Frozen identity/hash declaration changed for {row['image_id']}: {key}")
            row["split"] = previous_row["split"]
        elif row.get("split") not in (None, ""):
            raise ValueError("Input assignments must be blank unless present in --previous")
        if not row.get("group_id"):
            if not allow_image_level:
                raise ValueError(f"Missing identity group: {row['image_id']}")
            row["group_id"] = "image_" + row["image_id"]
            row["group_id_origin"] = "image_level_only"
        elif row.get("group_id_origin") not in KNOWN_GROUP_ORIGINS:
            if not allow_image_level:
                raise ValueError("group_id_origin must record official_identity_id or participant_code, not unknown")
    for members in _components(records):
        frozen = {records[index]["split"] for index in members if records[index]["image_id"] in previous_by_id}
        if len(frozen) > 1:
            raise ValueError("New duplicate/identity links connect previously separated splits; freeze and investigate")
        has_additions = any(records[index]["image_id"] not in previous_by_id for index in members)
        if frozen:
            assignment = next(iter(frozen))
            if assignment not in SPLITS:
                raise ValueError("Invalid split in previous manifest")
            if additions_train_only and has_additions and assignment != "train":
                raise ValueError("Error-mined additions overlap a locked validation/test identity or duplicate")
        elif additions_train_only:
            assignment = "train"
        else:
            identity = min(records[index]["group_id"] for index in members)
            number = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest(), 16) / (1 << 256)
            assignment = "train" if number < ratios[0] else "validation" if number < sum(ratios[:2]) else "test"
        for index in members:
            records[index]["split"] = assignment
    audit = validate_manifest(records, require_identity=not allow_image_level)
    audit.update({"seed": seed, "requested_ratios": ratios, "assignment": "hash-based group assignment; counts are approximate, no label stratification",
                  "frozen_previous_images": len(previous), "new_images_train_only": additions_train_only})
    return records, audit


def import_identities(records, mapping, key, namespace, origin="official_identity_id"):
    mapping_by_id = {}
    for row in mapping:
        if row["image_id"] in mapping_by_id:
            raise ValueError("Duplicate identity-map image ID")
        if not row.get("source_identity_id"):
            raise ValueError("Blank source_identity_id")
        mapping_by_id[row["image_id"]] = str(row["source_identity_id"])
    if not set(mapping_by_id).issubset({row["image_id"] for row in records}):
        raise ValueError("Identity map contains unknown image IDs")
    result = []
    for original in records:
        row = dict(original)
        if row["image_id"] in mapping_by_id:
            value = (namespace + ":" + mapping_by_id[row["image_id"]]).encode()
            row["group_id"] = "g_" + hmac.new(key, value, hashlib.sha256).hexdigest()[:24]
            row["group_id_origin"] = origin
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("inventory")
    scan.add_argument("--root", type=Path, required=True)
    scan.add_argument("--source", required=True)
    scan.add_argument("--role", choices=["unlabeled_candidate", "functional_fixture", "accuracy_set"], default="unlabeled_candidate")
    scan.add_argument("--max-files", type=int, default=100000)
    scan.add_argument("--max-depth", type=int, default=8)
    scan.add_argument("--output", type=Path, required=True)
    identity = sub.add_parser("import-identities")
    identity.add_argument("--input", required=True, type=Path)
    identity.add_argument("--identity-map", required=True, type=Path)
    identity.add_argument("--key-file", required=True, type=Path)
    identity.add_argument("--namespace", required=True)
    identity.add_argument("--origin", choices=["official_identity_id", "participant_code"], default="official_identity_id")
    identity.add_argument("--output", required=True, type=Path)
    split = sub.add_parser("split")
    split.add_argument("--input", required=True, type=Path)
    split.add_argument("--previous", type=Path)
    split.add_argument("--seed", default="20261113")
    split.add_argument("--ratios", nargs=3, type=float, default=[.7, .15, .15])
    split.add_argument("--allow-image-level", action="store_true")
    split.add_argument("--additions-train-only", action="store_true")
    split.add_argument("--output", required=True, type=Path)
    audit = sub.add_parser("validate")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--allow-image-level", action="store_true")
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        rows, failures = inventory(args.root, args.source, args.role, args.max_files, args.max_depth)
        write_csv(args.output, rows)
        write_json(args.output.with_suffix(".audit.json"), {"root": str(args.root), "n_images": len(rows), "failures": failures, "scope": "explicit root only", "role": args.role})
    elif args.command == "import-identities":
        if not args.key_file.exists():
            args.key_file.parent.mkdir(parents=True, exist_ok=True)
            args.key_file.write_bytes(secrets.token_bytes(32))
        key = args.key_file.read_bytes()
        if len(key) < 32:
            raise ValueError("Pseudonymization key must contain at least 32 bytes; keep it private and stable")
        rows = import_identities(read_records(args.input), read_records(args.identity_map), key, args.namespace, args.origin)
        write_csv(args.output, rows)
    elif args.command == "split":
        rows, report = build_splits(read_records(args.input), args.seed, args.ratios,
                                   read_records(args.previous) if args.previous else None,
                                   args.allow_image_level, args.additions_train_only)
        write_csv(args.output, rows)
        write_json(args.output.with_suffix(".audit.json"), report)
    else:
        write_json(args.output, validate_manifest(read_records(args.input), not args.allow_image_level))
    print(json.dumps({"command": args.command, "output": str(args.output)}))


if __name__ == "__main__":
    main()
