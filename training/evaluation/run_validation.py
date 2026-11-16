"""Exercise CSV/JSON/PNG command-line paths using explicit synthetic fixtures only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from evaluation_common import MASK_SCHEMA, sha256, write_csv, write_json

HERE = Path(__file__).resolve().parent


def run(command, output, label):
    result = subprocess.run(command, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / (label + ".log.txt")).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see saved log")
    return {"check": label, "exit_code": result.returncode, "log": str(output / (label + ".log.txt"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = [run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_evaluation.py", "-v"], output, "unit_tests")]
    fixtures = output / "synthetic_fixtures_not_model_accuracy"
    fixtures.mkdir(exist_ok=True)
    targets = [1, 1, 0, 0, 1, 0]
    a_scores = [.9, .2, .6, .2, .8, .1]
    b_scores = [.9, .8, .1, .2, .8, .1]
    labels, a, b = [], [], []
    for index, (target, first, second) in enumerate(zip(targets, a_scores, b_scores)):
        image = f"synthetic_{index}"
        labels.append({"image_id": image, "group_id": f"synthetic_group_{index // 2}", "split": "validation",
                       "dataset_role": "functional_fixture", "label_source": "synthetic_test_fixture",
                       "attribute": "smiling", "target": target})
        a.append({"image_id": image, "attribute": "smiling", "score": first})
        b.append({"image_id": image, "attribute": "smiling", "score": second})
    write_csv(fixtures / "attributes_truth.csv", labels)
    write_csv(fixtures / "attributes_a.csv", a)
    write_json(fixtures / "attributes_b.json", {"records": b})
    write_json(fixtures / "thresholds.json", {"version": "synthetic_test_only", "selection_split": "fixed_uncalibrated_demo", "thresholds": {"smiling": .5}})
    checks.append(run([sys.executable, str(HERE / "evaluate.py"), "attributes", "--labels", str(fixtures / "attributes_truth.csv"),
                       "--predictions", str(fixtures / "attributes_a.csv"), "--compare", str(fixtures / "attributes_b.json"),
                       "--thresholds", str(fixtures / "thresholds.json"), "--allow-functional-fixtures", "--bootstrap", "200",
                       "--output", str(output / "synthetic_attribute_tool_check.json")], output, "attributes_csv_json_cli"))
    attr = json.loads((output / "synthetic_attribute_tool_check.json").read_text(encoding="utf-8"))
    assert abs(attr["model_a"]["smiling"]["f1"] - 2 / 3) < 1e-12
    assert attr["model_b"]["smiling"]["f1"] == 1
    assert attr["n_groups"] == 3 and attr["evidence_kind"] == "functional_fixture_not_accuracy"
    labels, a, b = [], [], []
    for index in range(3):
        target = np.array([[0, 1], [4, 255]], np.uint8)
        pred_a = np.array([[0, 1], [1, 99]], np.uint8)
        pred_b = np.array([[0, 1], [4, 99]], np.uint8)
        image_id = f"synthetic_mask_{index}"
        for name, pixels in (("truth", target), ("a", pred_a), ("b", pred_b)):
            Image.fromarray(pixels).save(fixtures / f"{image_id}_{name}.png")
        labels.append({"image_id": image_id, "group_id": f"synthetic_mask_group_{index}", "split": "validation",
                       "dataset_role": "functional_fixture", "label_source": "synthetic_test_fixture",
                       "mask_path": f"{image_id}_truth.png", "mask_schema": MASK_SCHEMA})
        a.append({"image_id": image_id, "mask_path": f"{image_id}_a.png", "mask_schema": MASK_SCHEMA})
        b.append({"image_id": image_id, "mask_path": f"{image_id}_b.png", "mask_schema": MASK_SCHEMA})
    write_json(fixtures / "masks_truth.json", labels)
    write_csv(fixtures / "masks_a.csv", a)
    write_csv(fixtures / "masks_b.csv", b)
    checks.append(run([sys.executable, str(HERE / "evaluate.py"), "segmentation", "--labels", str(fixtures / "masks_truth.json"),
                       "--predictions", str(fixtures / "masks_a.csv"), "--compare", str(fixtures / "masks_b.csv"),
                       "--allow-functional-fixtures", "--bootstrap", "200", "--output", str(output / "synthetic_segmentation_tool_check.json")],
                       output, "segmentation_json_csv_png_cli"))
    seg = json.loads((output / "synthetic_segmentation_tool_check.json").read_text(encoding="utf-8"))
    assert seg["ignored_pixels"] == 3
    assert seg["model_b"]["aggregate"]["miou_excluding_background"] == 1
    assert seg["model_a"]["classes"]["left_eye"]["iou"] == 0
    assert seg["model_b"]["classes"]["hat"]["iou"] is None
    inventory_path = output / "synthetic_inventory.csv"
    checks.append(run([sys.executable, str(HERE / "manifest.py"), "inventory", "--root", str(fixtures), "--source", "synthetic_tool_masks",
                       "--role", "functional_fixture", "--output", str(inventory_path)], output, "inventory_cli"))
    write_json(output / "validation_summary.json", {"scope": "Tool correctness with synthetic fixtures, not any model accuracy",
                "all_checks_passed": True, "checks": checks, "fixture_ground_truth": "Hand-specified synthetic integer arrays and binary numbers",
                "real_people_labeled": 0, "model_inference_performed": False,
                "source_sha256": {path.name: sha256(path) for path in sorted(HERE.glob("*.py"))}})
    print(json.dumps({"all_checks_passed": True, "checks": len(checks), "output": str(output / "validation_summary.json")}))


if __name__ == "__main__":
    main()
