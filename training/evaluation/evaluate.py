"""Evaluate independently supplied labels and predictions, with grouped paired intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation_common import (APP_CLASSES, ATTRIBUTES, MASK_SCHEMA, check_annotation, finite_mean,
                               paired_group_bootstrap, ratio, read_records, sha256, write_csv, write_json)


def binary_metrics(counts):
    tp, fp, fn, tn = (int(value) for value in counts)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "positive_support": tp + fn, "negative_support": tn + fp}


def attribute_flat_metrics(counts, names):
    result = {}
    for name, cells in zip(names, counts):
        metrics = binary_metrics(cells)
        for metric in ("precision", "recall", "f1"):
            result[f"{name}.{metric}"] = metrics[metric]
    result["macro_f1"] = finite_mean(result[f"{name}.f1"] for name in names)
    return result


def _unique(rows, key_function):
    result = {}
    for row in rows:
        key = key_function(row)
        if key in result:
            raise ValueError(f"Duplicate record key: {key}")
        result[key] = row
    return result


def _target(value):
    if value is None or str(value).strip().lower() in ("", "unknown", "ungradable", "na", "null"):
        return None
    if str(value) not in ("0", "1"):
        raise ValueError(f"Binary label must be 0, 1 or unknown, got {value!r}; convert CelebA -1 explicitly")
    return int(value)


def _score(row):
    status = row.get("prediction_status") or "available"
    if status not in ("available", "unavailable"):
        raise ValueError("prediction_status must be available or unavailable")
    if status == "unavailable":
        if str(row.get("score", "")).strip() not in ("", "None", "null"):
            raise ValueError("Unavailable prediction must not also have a score")
        return None
    value = float(row["score"])
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Attribute score must be finite in [0,1]")
    return value


def evaluate_attributes(labels, predictions_a, thresholds, predictions_b=None, split="validation", repeats=2000,
                        seed=20261113, allow_fixtures=False, thresholds_b=None):
    selected = [row for row in labels if row.get("split") == split]
    if not selected:
        raise ValueError(f"No labels for split {split}")
    for row in selected:
        check_annotation(row, allow_fixtures)
        if row.get("attribute") not in ATTRIBUTES:
            raise ValueError(f"Unsupported attribute: {row.get('attribute')}")
    truth = _unique(selected, lambda row: (row["image_id"], row["attribute"]))
    pa = _unique(predictions_a, lambda row: (row["image_id"], row["attribute"]))
    pb = _unique(predictions_b, lambda row: (row["image_id"], row["attribute"])) if predictions_b is not None else None
    for prediction in (pa, pb):
        if prediction is not None and not set(truth).issubset(prediction):
            missing = list(set(truth) - set(prediction))[:5]
            raise ValueError(f"Missing prediction rows: {missing}; record explicit unavailable rows for inference failures")
    names = sorted({row["attribute"] for row in selected})
    thresholds_b = thresholds if thresholds_b is None else thresholds_b
    for name in names:
        if name not in thresholds or not 0 <= float(thresholds[name]) <= 1:
            raise ValueError(f"Explicit threshold in [0,1] required for {name}")
        if pb is not None and (name not in thresholds_b or not 0 <= float(thresholds_b[name]) <= 1):
            raise ValueError(f"Explicit model B threshold in [0,1] required for {name}")
    image_groups = {}
    for row in selected:
        if row["image_id"] in image_groups and image_groups[row["image_id"]] != row["group_id"]:
            raise ValueError("One image cannot belong to different groups")
        image_groups[row["image_id"]] = row["group_id"]
    groups = sorted(set(image_groups.values()))
    group_index = {group: index for index, group in enumerate(groups)}
    counts_a = np.zeros((len(groups), len(names), 4), dtype=np.int64)
    counts_b = np.zeros_like(counts_a) if pb is not None else None
    coverage = {name: {"label_rows": 0, "unknown_truth": 0, "known_truth": 0,
                       "a_unavailable": 0, "b_unavailable": 0, "common_evaluated": 0} for name in names}
    for key, row in truth.items():
        name = row["attribute"]
        cov = coverage[name]
        cov["label_rows"] += 1
        target = _target(row.get("target"))
        if target is None:
            cov["unknown_truth"] += 1
            continue
        cov["known_truth"] += 1
        score_a = _score(pa[key])
        score_b = _score(pb[key]) if pb is not None else score_a
        cov["a_unavailable"] += int(score_a is None)
        cov["b_unavailable"] += int(pb is not None and score_b is None)
        if score_a is None or score_b is None:
            continue
        cov["common_evaluated"] += 1
        g, a = group_index[row["group_id"]], names.index(name)
        for score, tensor, cutoff in ((score_a, counts_a, thresholds), (score_b, counts_b, thresholds_b)):
            if tensor is None:
                continue
            predicted = score >= float(cutoff[name])
            cell = 0 if predicted and target else 1 if predicted else 2 if target else 3
            tensor[g, a, cell] += 1
    for cov in coverage.values():
        cov["a_coverage"] = ratio(cov["known_truth"] - cov["a_unavailable"], cov["known_truth"])
        cov["b_coverage"] = ratio(cov["known_truth"] - cov["b_unavailable"], cov["known_truth"]) if pb is not None else None
        cov["evaluation_coverage"] = ratio(cov["common_evaluated"], cov["known_truth"])
    if not counts_a.sum():
        raise ValueError("No known, available paired labels to evaluate")
    result = {"task": "visible_attributes", "split": split,
              "evidence_kind": "functional_fixture_not_accuracy" if allow_fixtures else "independently_supplied_annotations",
              "independence_verified": False,
              "annotation_provenance_note": "Label-source fields are required declarations; inspect registry and audits before claiming independent accuracy.",
              "thresholds": thresholds, "n_images": len(image_groups), "n_groups": len(groups),
              "cohort": "common available predictions; missing inference is reported as coverage, never silently a correct negative",
              "coverage": coverage,
              "model_a": {name: binary_metrics(cells) for name, cells in zip(names, counts_a.sum(axis=0))},
              "macro_a": attribute_flat_metrics(counts_a.sum(axis=0), names)["macro_f1"]}
    if counts_b is not None:
        result["thresholds_b"] = thresholds_b
        result["model_b"] = {name: binary_metrics(cells) for name, cells in zip(names, counts_b.sum(axis=0))}
        result["macro_b"] = attribute_flat_metrics(counts_b.sum(axis=0), names)["macro_f1"]
    result["bootstrap"] = paired_group_bootstrap(counts_a, lambda x: attribute_flat_metrics(x, names), counts_b, repeats, seed)
    return result


def confusion_matrix(target, prediction, classes=19, ignore=255):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Masks must be 2D with identical size; no implicit resizing is permitted")
    if target.dtype.kind not in "ui" or prediction.dtype.kind not in "ui":
        raise ValueError("Masks must contain integer class IDs, not colors or probabilities")
    valid = target != ignore
    y, p = target[valid].astype(np.int64), prediction[valid].astype(np.int64)
    if np.any((y < 0) | (y >= classes)) or np.any((p < 0) | (p >= classes)):
        raise ValueError("Out-of-range class at a non-ignored pixel")
    return np.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)


def segmentation_metrics(matrix):
    matrix = np.asarray(matrix)
    result = {}
    for index, name in enumerate(APP_CLASSES):
        tp = int(matrix[index, index])
        actual, predicted = int(matrix[index, :].sum()), int(matrix[:, index].sum())
        union = actual + predicted - tp
        result[name] = {"class_id": index, "true_pixels": actual, "predicted_pixels": predicted,
                        "intersection": tp, "union": union, "iou": ratio(tp, union),
                        "dice": ratio(2 * tp, actual + predicted), "absent_from_truth_and_prediction": union == 0}
    return result


def segmentation_flat(matrix):
    classes = segmentation_metrics(matrix)
    result = {}
    for name, metrics in classes.items():
        result[f"{name}.iou"] = metrics["iou"]
        result[f"{name}.dice"] = metrics["dice"]
    result["miou_including_background"] = finite_mean(row["iou"] for row in classes.values())
    result["miou_excluding_background"] = finite_mean(row["iou"] for name, row in classes.items() if name != "background")
    result["mdice_excluding_background"] = finite_mean(row["dice"] for name, row in classes.items() if name != "background")
    result["pixel_accuracy"] = ratio(np.trace(matrix), matrix.sum())
    return result


def read_mask(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image).copy()


def evaluate_segmentation(labels, predictions_a, predictions_b=None, split="validation", repeats=2000,
                          seed=20261113, allow_fixtures=False, base_labels=Path("."), base_a=Path("."), base_b=Path(".")):
    selected = [row for row in labels if row.get("split") == split]
    if not selected:
        raise ValueError(f"No segmentation labels for {split}")
    truth = _unique(selected, lambda row: row["image_id"])
    pa = _unique(predictions_a, lambda row: row["image_id"])
    pb = _unique(predictions_b, lambda row: row["image_id"]) if predictions_b is not None else None
    for prediction in (pa, pb):
        if prediction is not None and not set(truth).issubset(prediction):
            raise ValueError("Every selected image needs a prediction mask; inference failures must be resolved/reported separately")
    for row in selected:
        check_annotation(row, allow_fixtures)
    groups = sorted({row["group_id"] for row in selected})
    groups_index = {value: index for index, value in enumerate(groups)}
    counts_a = np.zeros((len(groups), 19, 19), dtype=np.int64)
    counts_b = np.zeros_like(counts_a) if pb is not None else None
    support_a = np.zeros(19, dtype=np.int64)
    support_b = np.zeros(19, dtype=np.int64)
    actual_support = np.zeros(19, dtype=np.int64)
    ignored_pixels = 0
    per_image = []
    for image_id, row in truth.items():
        records = [row, pa[image_id]] + ([pb[image_id]] if pb is not None else [])
        if any(record.get("mask_schema") != MASK_SCHEMA for record in records):
            raise ValueError(f"Every mask must explicitly declare {MASK_SCHEMA}; remap upstream labels before evaluation")
        target = read_mask(Path(base_labels) / row["mask_path"])
        ignored_pixels += int((target == 255).sum())
        prediction_a = read_mask(Path(base_a) / pa[image_id]["mask_path"])
        ca = confusion_matrix(target, prediction_a)
        if not ca.sum():
            raise ValueError(f"All pixels ignored in {image_id}; exclude explicitly before freezing the cohort")
        counts_a[groups_index[row["group_id"]]] += ca
        support_a += (ca.sum(axis=0) > 0)
        actual_support += (ca.sum(axis=1) > 0)
        record = {"image_id": image_id, "group_id": row["group_id"], "valid_pixels": int(ca.sum()),
                  "a": segmentation_flat(ca)}
        if pb is not None:
            prediction_b = read_mask(Path(base_b) / pb[image_id]["mask_path"])
            cb = confusion_matrix(target, prediction_b)
            counts_b[groups_index[row["group_id"]]] += cb
            support_b += (cb.sum(axis=0) > 0)
            record["b"] = segmentation_flat(cb)
        per_image.append(record)
    def summary(counts, support):
        matrix = counts.sum(axis=0)
        classes = segmentation_metrics(matrix)
        for index, name in enumerate(APP_CLASSES):
            classes[name]["images_with_ground_truth"] = int(actual_support[index])
            classes[name]["images_with_predictions"] = int(support[index])
        return {"classes": classes, "aggregate": segmentation_flat(matrix), "confusion_matrix": matrix.tolist()}
    result = {"task": "face_segmentation", "split": split, "mask_schema": MASK_SCHEMA, "ignore_id": 255,
              "evidence_kind": "functional_fixture_not_accuracy" if allow_fixtures else "independently_supplied_annotations",
              "independence_verified": False, "n_images": len(selected), "n_groups": len(groups),
              "ignored_pixels": ignored_pixels,
              "aggregation": "Sum per-image confusion matrices, then calculate per-class IoU/Dice; row=true, column=predicted",
              "absent_rule": "Absent in truth AND predictions: null, excluded from macro. False-positive-only class: 0, included.",
              "model_a": summary(counts_a, support_a), "per_image": per_image}
    if counts_b is not None:
        result["model_b"] = summary(counts_b, support_b)
    result["bootstrap"] = paired_group_bootstrap(counts_a, segmentation_flat, counts_b, repeats, seed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["attributes", "segmentation"])
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--compare-thresholds", type=Path, help="Model B thresholds fixed on validation; defaults to model A thresholds")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--allow-functional-fixtures", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    labels, a = read_records(args.labels), read_records(args.predictions)
    b = read_records(args.compare) if args.compare else None
    common = dict(predictions_b=b, split=args.split, repeats=args.bootstrap, seed=args.seed,
                  allow_fixtures=args.allow_functional_fixtures)
    if args.task == "attributes":
        if args.thresholds is None:
            parser.error("--thresholds is required; thresholds must be fixed before test evaluation")
        threshold_config = json.loads(args.thresholds.read_text(encoding="utf-8-sig"))
        if threshold_config.get("selection_split") not in ("validation", "fixed_uncalibrated_demo"):
            raise ValueError("Threshold provenance must declare validation or fixed_uncalibrated_demo; never test")
        threshold_b = None
        if args.compare_thresholds:
            if not args.compare:
                parser.error("--compare-thresholds requires --compare")
            threshold_b = json.loads(args.compare_thresholds.read_text(encoding="utf-8-sig"))
            if threshold_b.get("selection_split") not in ("validation", "fixed_uncalibrated_demo"):
                raise ValueError("Model B thresholds must not be selected on test")
        result = evaluate_attributes(labels, a, threshold_config["thresholds"], thresholds_b=threshold_b["thresholds"] if threshold_b else None, **common)
        result["threshold_provenance"] = threshold_config
        if threshold_b:
            result["threshold_provenance_b"] = threshold_b
    else:
        result = evaluate_segmentation(labels, a, base_labels=args.labels.parent, base_a=args.predictions.parent,
                                       base_b=args.compare.parent if args.compare else Path("."), **common)
    result["input_sha256"] = {"labels": sha256(args.labels), "predictions_a": sha256(args.predictions),
                               "predictions_b": sha256(args.compare) if args.compare else None}
    result["locked_test_accessed"] = args.split == "test"
    write_json(args.output, result)
    table = []
    for model in ("model_a", "model_b"):
        if model not in result:
            continue
        values = result[model] if args.task == "attributes" else result[model]["classes"]
        for name, metrics in values.items():
            table.append({"model": model, "label": name, **metrics})
    write_csv(args.output.with_suffix(".csv"), table)
    print(json.dumps({"output": str(args.output), "task": args.task, "n_images": result["n_images"],
                      "n_groups": result["n_groups"], "evidence_kind": result["evidence_kind"]}))


if __name__ == "__main__":
    main()
