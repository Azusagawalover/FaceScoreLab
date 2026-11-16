"""Small, explicit file and bootstrap helpers; no model inference or ground-truth synthesis."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Input must be a CSV, JSON array, or JSON object containing records[]")
    return records


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def finite_mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def interval(values, confidence=0.95):
    usable = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(usable):
        return {"low": None, "high": None, "valid_replicates": 0, "total_replicates": len(values)}
    tail = (1 - confidence) / 2
    low, high = np.quantile(usable, [tail, 1 - tail])
    return {"low": float(low), "high": float(high), "valid_replicates": len(usable), "total_replicates": len(values)}


def paired_group_bootstrap(group_a, metric_function, group_b=None, repeats=2000, seed=20261113):
    """Resample groups, not rows/pixels; both models use the exact same draws."""
    if repeats < 0:
        raise ValueError("repeats must be nonnegative")
    group_a = np.asarray(group_a)
    if group_a.ndim < 2 or not len(group_a):
        raise ValueError("At least one group is required")
    if group_b is not None and np.shape(group_b) != np.shape(group_a):
        raise ValueError("Paired models must have identical ordered groups and tensor shape")
    group_b = np.asarray(group_b) if group_b is not None else None
    rng = np.random.default_rng(seed)
    collected = {"a": {}, "b": {}, "delta_b_minus_a": {}}
    for _ in range(repeats):
        indices = rng.integers(0, len(group_a), size=len(group_a))
        metrics_a = metric_function(group_a[indices].sum(axis=0))
        metrics_b = metric_function(group_b[indices].sum(axis=0)) if group_b is not None else None
        for name, value in metrics_a.items():
            collected["a"].setdefault(name, []).append(value)
            if metrics_b is not None:
                other = metrics_b[name]
                collected["b"].setdefault(name, []).append(other)
                delta = other - value if other is not None and value is not None else None
                collected["delta_b_minus_a"].setdefault(name, []).append(delta)
    return {
        "method": "paired percentile bootstrap of groups, 95% interval",
        "n_groups": len(group_a), "repeats": repeats, "seed": seed,
        "warning": "One group cannot estimate between-group uncertainty; few groups or rare positives give unstable intervals." if len(group_a) < 30 else None,
        "intervals": {model: {name: interval(values) for name, values in metrics.items()} for model, metrics in collected.items() if metrics},
    }


APP_CLASSES = ["background", "skin", "left_brow", "right_brow", "left_eye", "right_eye", "eyeglasses", "left_ear", "right_ear", "ear_accessory", "nose", "mouth_interior", "upper_lip", "lower_lip", "neck", "necklace", "clothing", "hair", "hat"]
ATTRIBUTES = ["left_eye_open", "right_eye_open", "eyeglasses", "mask", "sunglasses", "smiling", "mouth_open"]
MASK_SCHEMA = "facescorelab_app_19_v1"
ALLOWED_LABEL_SOURCES = {"official_annotation", "human_double_reviewed"}


def check_annotation(row, allow_fixtures=False):
    for key in ("image_id", "group_id", "split", "dataset_role", "label_source"):
        if not str(row.get(key, "")).strip():
            raise ValueError(f"Missing {key} in annotation: {row.get('image_id')}")
    if row["dataset_role"] == "functional_fixture":
        if not allow_fixtures:
            raise ValueError("functional_fixture is not an accuracy set; use explicit fixture mode for tool testing only")
        if row["label_source"] != "synthetic_test_fixture":
            raise ValueError("Fixture mode accepts synthetic_test_fixture truth only; do not turn model outputs into labels")
        return
    if row["dataset_role"] != "accuracy_set" or row["label_source"] not in ALLOWED_LABEL_SOURCES:
        raise ValueError("Accuracy labels must be official_annotation or human_double_reviewed, never model predictions")
    if not row.get("source_reference"):
        raise ValueError("Accuracy labels need a source_reference to the annotation source or review ledger")
    if row["label_source"] == "human_double_reviewed":
        reviewers = [str(row.get(key, "")).strip() for key in ("reviewer_1", "reviewer_2")]
        if not all(reviewers) or reviewers[0] == reviewers[1]:
            raise ValueError("Two distinct reviewer IDs are required")
        if row.get("review_status") not in ("agreed", "adjudicated"):
            raise ValueError("Unresolved annotation cannot be evaluated")
        if row.get("review_status") == "adjudicated" and not row.get("adjudicator"):
            raise ValueError("Adjudicated annotations require an adjudicator ID")
        if "attribute" in row and row.get("review_status") == "agreed":
            def normalized(value):
                text = str(value).strip().lower() if value is not None else "unknown"
                return "unknown" if text in ("", "na", "null", "ungradable", "unknown") else text
            if "reviewer_1_target" not in row or "reviewer_2_target" not in row:
                raise ValueError("Agreed attribute labels require both reviewers' independent target fields")
            values = [normalized(row.get(key)) for key in ("reviewer_1_target", "reviewer_2_target", "target")]
            if len(set(values)) != 1:
                raise ValueError("Agreed reviewer targets and final target do not match; adjudicate or mark unknown")
