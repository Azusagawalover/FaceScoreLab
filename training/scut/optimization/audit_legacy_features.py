"""Compare only development samples against the old anonymous feature cache."""
import argparse
import json
from pathlib import Path
import numpy as np
from runner import atomic_json, load_development_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dev-manifest", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--old-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = load_development_rows(args.dev_manifest)
    names = [r["name"] for r in rows]
    legacy_names = json.loads((args.data_dir/"frozen_early_names.json").read_text())
    old_lookup = {name: i for i, name in enumerate(legacy_names)}
    current_metrics = json.loads((args.run/"metrics.json").read_text())
    cache_path = Path(current_metrics["cache"]["path"])
    metadata = json.loads((cache_path.parent/"cache_manifest.json").read_text())
    assert [r["name"] for r in metadata["spec"]["inputs"]] == names
    old = np.load(args.data_dir/"frozen_early_maps.npy", mmap_mode="r", allow_pickle=False)
    new = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    assert new.shape[0] == 1 and new.shape[1] == len(rows)
    total_sum = total_elements = differing_elements = 0
    max_error = 0.
    samples = []
    for start in range(0, len(rows), 32):
        selected = rows[start:start+32]
        old_values = np.asarray(old[[old_lookup[r["name"]] for r in selected]])
        new_values = np.asarray(new[0, start:start+len(selected)])
        absolute = np.abs(old_values.astype(np.float64)-new_values.astype(np.float64))
        total_sum += float(absolute.sum())
        total_elements += absolute.size
        differing_elements += int(np.count_nonzero(absolute))
        max_error = max(max_error, float(absolute.max()))
        flat = absolute.reshape(len(selected), -1)
        for row, maximum, average in zip(selected, flat.max(1), flat.mean(1)):
            samples.append({"name": row["name"], "split": row["split"],
                            "max_abs_difference": float(maximum), "mean_abs_difference": float(average)})
    old._mmap.close()
    new._mmap.close()
    old_logs = json.loads(args.old_log.read_text())
    current_logs = json.loads((args.run/"training_log.json").read_text())
    historical = {(r["stage"], r["epoch"]): r for r in old_logs}
    differences = []
    for row in current_logs:
        old_row = historical.get((row["stage"], row["epoch"]))
        if old_row:
            differences.append({"stage": row["stage"], "epoch": row["epoch"],
                                "old_validation_mae": old_row["validation"]["mae"],
                                "new_validation_mae": row["validation"]["mae"],
                                "validation_mae_difference": row["validation"]["mae"]-old_row["validation"]["mae"],
                                "training_loss_difference": row["train_loss"]-old_row["train_mse"]})
    result = {"development_count": len(rows), "test_targets_opened": False,
              "test_feature_rows_compared": 0, "elements_compared": total_elements,
              "differing_elements": differing_elements, "max_abs_difference": max_error,
              "mean_abs_difference": total_sum/total_elements,
              "images_with_any_difference": sum(r["max_abs_difference"] > 0 for r in samples),
              "largest_differences": sorted(samples, key=lambda r: r["max_abs_difference"], reverse=True)[:20],
              "epoch_comparisons": differences,
              "note": "Only indexed train/validation rows of legacy cache were compared; no legacy test feature rows or labels used."}
    atomic_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("largest_differences", "epoch_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
