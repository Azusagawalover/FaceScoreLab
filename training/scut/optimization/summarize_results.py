"""Read-only validation selection and locked, already-evaluated SCUT reports.

No training or inference imports are used. The final command will never run an
evaluation; it requires completed evaluation.json / test_predictions.csv files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = "facescore-results-v1"
RECIPES = ("R0", "R1", "R2", "R3", "R4", "R5")
SEEDS = (17, 42, 2026)
TARGET_TOLERANCE = 1e-5
METRIC_TOLERANCE = 2e-6
HISTORICAL_STATUS = "previously_observed_official_fold1_historical_benchmark"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def atomic_write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def write_text(path, text):
    atomic_write(path, text.encode("utf-8"))


def write_csv(path, rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue().encode("utf-8-sig"))


def now():
    return datetime.now(timezone.utc).isoformat()


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    bounds = np.concatenate(([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1, [len(values)]))
    ranks = np.empty(len(values), np.float64)
    for start, end in zip(bounds[:-1], bounds[1:]):
        ranks[order[start:end]] = (start+end-1)/2+1
    return ranks


def regression_metrics(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if not len(y) or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid regression arrays")
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-10 and b.std() > 1e-10 else None
    return {"n": len(y), "mae": float(np.abs(y-p).mean()), "rmse": float(np.sqrt(np.mean((y-p)**2))),
            "pearson": corr(y, p), "spearman": corr(average_ranks(y), average_ranks(p))}


def read_predictions(path, expected_split, historical=False):
    """Check the split before reading target/prediction values."""
    rows = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"name", "target_1_5", "prediction_raw"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected prediction CSV header: {path}")
        if not historical and "split" not in reader.fieldnames:
            raise ValueError("Runner prediction CSV must contain split")
        for row in reader:
            if not historical and row["split"] != expected_split:
                raise ValueError(f"Prohibited split in {path}: {row['split']}")
            name = row["name"]
            if name in rows:
                raise ValueError(f"Duplicate prediction name: {name}")
            y, prediction = float(row["target_1_5"]), float(row["prediction_raw"])
            if not math.isfinite(y) or not 1 <= y <= 5 or not math.isfinite(prediction):
                raise ValueError("Invalid target or prediction")
            rows[name] = (y, prediction)
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    return rows


def align(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("Paired comparison requires exactly the same image names")
    names = sorted(reference)
    y = np.array([reference[name][0] for name in names], np.float64)
    other_y = np.array([candidate[name][0] for name in names], np.float64)
    target_difference = float(np.max(np.abs(y-other_y)))
    if target_difference > TARGET_TOLERANCE:
        raise ValueError("Paired comparison targets differ beyond recorded precision tolerance")
    return names, y, np.array([reference[name][1] for name in names]), np.array([candidate[name][1] for name in names]), target_difference


def check_metric(predictions, recorded):
    names = sorted(predictions)
    computed = regression_metrics([predictions[n][0] for n in names], [predictions[n][1] for n in names])
    if int(recorded["n"]) != computed["n"]:
        raise ValueError("Prediction row count disagrees with metric metadata")
    for key in ("mae", "rmse", "pearson", "spearman"):
        if key in recorded and recorded[key] is not None:
            if computed[key] is None or abs(float(recorded[key])-computed[key]) > METRIC_TOLERANCE:
                raise ValueError(f"Saved {key} does not agree with prediction CSV")
    return computed


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": .22,
                         "figure.dpi": 130, "savefig.dpi": 180, "font.size": 10})
    return plt


def base_report(kind):
    return {"format": VERSION, "kind": kind, "created_utc": now(), "target_scale": [1, 5],
            "selection_metric": "validation_raw_mae_1_to_5", "test_evaluation_triggered": False,
            "limitations": ["SCUT official fold 1 is a previously observed historical benchmark, not a new blind test.",
                            "These reports do not establish attribute F1, segmentation mIoU, or real-phone accuracy.",
                            "Pearson and Spearman are correlations, not accuracy percentages."]}


def load_development_run(root, recipe, seed, require_completed=False):
    directory = Path(root)/f"{recipe}-seed{seed}"
    metric_path = directory/"metrics.json"
    if not metric_path.exists():
        if require_completed:
            raise FileNotFoundError(f"Missing completed development run: {directory}")
        return {"recipe": recipe, "seed": seed, "run": str(directory.resolve()), "status": "missing"}, None
    metadata = read_json(metric_path)
    config = metadata["config"]
    if config["id"] != recipe or config["seed"] != seed or metadata.get("smoke_not_for_release", False):
        raise ValueError("Run identity mismatch or smoke result cannot enter comparison")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("Development record must explicitly state test_evaluated=false")
    if require_completed and metadata["status"] != "completed":
        raise ValueError(f"Confirmation requires completed runs: {directory}")
    row = {"recipe": recipe, "seed": seed, "run": str(directory.resolve()), "status": metadata["status"],
           "validation_mae": metadata.get("validation_mae"), "best_stage": metadata.get("best_stage"),
           "best_epoch": metadata.get("best_epoch"), "epochs_completed": metadata.get("epochs_completed"),
           "runner_wall_seconds": metadata.get("wall_seconds"), "fingerprint": metadata["fingerprint"],
           "best_checkpoint_sha256": metadata.get("best_checkpoint_sha256"),
           "metrics_sha256": sha256(metric_path), "train_n": metadata["counts"]["train"],
           "validation_n": metadata["counts"]["validation"]}
    prediction_path = directory/"validation_predictions.csv"
    predictions = None
    # In-progress checkpoint/CSV/metrics files are individually atomic but not a
    # multi-file transaction. Validate completed snapshots only.
    if row["validation_mae"] is not None and metadata["status"] == "completed":
        predictions = read_predictions(prediction_path, "validation")
        computed = check_metric(predictions, {"n": row["validation_n"], "mae": row["validation_mae"]})
        row.update(validation_rmse=computed["rmse"], validation_pearson=computed["pearson"],
                   validation_spearman=computed["spearman"], prediction_csv_mae=computed["mae"],
                   validation_predictions_sha256=sha256(prediction_path))
    return row, predictions


def screening(args):
    output = Path(args.output)
    rows, reference = [], None
    curves = []
    for recipe in RECIPES:
        row, prediction = load_development_run(args.runs_root, recipe, args.seed)
        if prediction:
            if reference is None:
                reference = prediction
            else:
                align(reference, prediction)
        rows.append(row)
        log_path = Path(row["run"])/"training_log.json"
        if log_path.exists():
            log = read_json(log_path)
            curves.append((recipe, [float(epoch["validation"]["mae"]) for epoch in log]))
    complete = all(r["status"] == "completed" and r.get("validation_mae") is not None for r in rows)
    finished = [r for r in rows if r["status"] == "completed" and r.get("validation_mae") is not None]
    best = min(finished, key=lambda r: (r["validation_mae"], r["recipe"])) if finished else None
    result = {**base_report("screen"), "screening_seed": args.seed, "all_six_completed": complete,
              "recommended_recipe": best["recipe"] if complete else None,
              "provisional_best_completed_recipe": best["recipe"] if best else None,
              "recommendation_basis": "Lowest saved validation raw MAE; no test data consulted",
              "requires_confirmation": bool(complete and best and best["recipe"] != "R0"), "runs": rows,
              "cost_note": "Summed runner time is not full process/phase wall time; startup, export, failed uncommitted work require outer accounting.",
              "sum_runner_wall_seconds": sum(r.get("runner_wall_seconds") or 0 for r in rows)}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"screening_comparison.csv", rows, ["recipe", "seed", "status", "validation_mae", "validation_rmse",
              "validation_pearson", "validation_spearman", "best_stage", "best_epoch", "epochs_completed", "runner_wall_seconds", "run"])
    plt = plots()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for recipe, values in curves:
        ax.plot(np.arange(1, len(values)+1), values, label=recipe, linewidth=1.4)
    ax.set(xlabel="Completed epochs (head + tail)", ylabel="Validation MAE (1-5)", title="Development validation only")
    if curves:
        ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output/"validation_mae_curves.png")
    plt.close(fig)
    write_json(output/"screening.json", result)
    text = ["六配置初筛结果（只使用开发集）", "", f"全部六组完成：{'是' if complete else '否；当前只是进度快照'}"]
    for row in rows:
        value = f"{row['validation_mae']:.6f}" if row.get("validation_mae") is not None else "暂无"
        text.append(f"{row['recipe']} / seed {row['seed']}：{row['status']}，验证MAE {value}")
    if complete:
        text.append(f"按验证MAE最低推荐：{result['recommended_recipe']}。这只是开发选择，尚不代表测试性能提高。")
        if result["recommended_recipe"] == "R0":
            text.append("本轮新配置未超过本批R0，不应另选测试分数好看的配置替代此决定。")
    text += ["", "本批R0是原训练方法在连续NCHW新流程中的复测；与历史旧流程基线分开报告。",
             "未读取测试结果；尚无本轮属性F1、分割mIoU或真机准确率结论。"]
    write_text(output/"结果说明.txt", "\n".join(text)+"\n")
    return result


def sample_summary(values):
    values = np.asarray(values, np.float64)
    return {"n": len(values), "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
            "minimum": float(values.min()), "maximum": float(values.max())}


def confirmation(args):
    if args.winner == "R0":
        raise ValueError("A separate winner recipe is required; do not compare R0 against itself")
    if args.minimum_improvement < 0:
        raise ValueError("Minimum improvement must be nonnegative")
    rows, reference = [], None
    for recipe in ("R0", args.winner):
        for seed in SEEDS:
            row, prediction = load_development_run(args.runs_root, recipe, seed, require_completed=True)
            if reference is None:
                reference = prediction
            else:
                align(reference, prediction)
            rows.append(row)
    summaries = {recipe: sample_summary([r["validation_mae"] for r in rows if r["recipe"] == recipe]) for recipe in ("R0", args.winner)}
    improvement = summaries["R0"]["mean"]-summaries[args.winner]["mean"]
    selected = min((r for r in rows if r["recipe"] == args.winner), key=lambda r: (r["validation_mae"], r["seed"]))
    paired = [{"seed": seed,
               "baseline_validation_mae": next(r["validation_mae"] for r in rows if r["recipe"] == "R0" and r["seed"] == seed),
               "candidate_validation_mae": next(r["validation_mae"] for r in rows if r["recipe"] == args.winner and r["seed"] == seed)} for seed in SEEDS]
    for row in paired:
        row["improvement"] = row["baseline_validation_mae"]-row["candidate_validation_mae"]
    result = {**base_report("confirm"), "baseline_recipe": "R0", "winner_recipe": args.winner, "seeds": list(SEEDS),
              "runs": rows, "validation_summary": summaries, "paired_seed_validation": paired,
              "mean_validation_mae_improvement": improvement, "minimum_mean_improvement": args.minimum_improvement,
              "validation_gate_passed": improvement >= args.minimum_improvement,
              "seeds_improved": sum(r["improvement"] > 0 for r in paired),
              "deployment_checkpoint": {k: selected[k] for k in ("recipe", "seed", "run", "fingerprint", "validation_mae", "best_checkpoint_sha256")},
              "deployment_selection_rule": "Lowest validation MAE within winner recipe; ties use ascending seed. Never reselect using test.",
              "note": "Selected checkpoint is a frozen evaluation candidate, not automatic deployment approval."}
    result["confirmation_id"] = digest(result)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output/"confirmation.json"
    if target.exists():
        raise FileExistsError("Confirmation is a frozen selection record; choose a new output directory")
    write_csv(output/"validation_seed_comparison.csv", rows, ["recipe", "seed", "validation_mae", "validation_rmse", "validation_pearson",
              "validation_spearman", "best_stage", "best_epoch", "best_checkpoint_sha256", "run"])
    draw_seed_plot(rows, "validation_mae", "Validation MAE (1-5)", ("R0", args.winner), output/"validation_three_seeds.png")
    write_json(target, result)
    text = ["三种子确认（只使用开发集）", ""]
    for recipe, stats in summaries.items():
        text.append(f"{recipe}：验证MAE {stats['mean']:.6f} ± {stats['sample_standard_deviation']:.6f}（3个seed，样本标准差）。")
    text += [f"平均验证MAE降低：{improvement:.6f}；预设门槛：{args.minimum_improvement:.6f}；{'通过' if result['validation_gate_passed'] else '未通过'}。",
             f"3个配对seed中，{result['seeds_improved']}个验证误差下降。",
             f"按验证集选择的候选：{selected['recipe']}-seed{selected['seed']}，验证MAE {selected['validation_mae']:.6f}。",
             "该选择已经冻结；最终测试后不能改选其中测试分数最好的seed。",
             "这是训练方法的开发比较，不能等同于旧App固定权重的部署对比。",
             "尚未读取测试结果，也没有属性F1、分割mIoU或真机准确率结论。"]
    write_text(output/"结果说明.txt", "\n".join(text)+"\n")
    return result


def draw_seed_plot(rows, key, ylabel, recipes, path):
    plt = plots()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for seed in SEEDS:
        chosen = [next(r[key] for r in rows if r["recipe"] == recipe and r["seed"] == seed) for recipe in recipes]
        ax.plot(range(len(recipes)), chosen, marker="o", linewidth=1, alpha=.65, label=f"seed {seed}")
    means = [np.mean([r[key] for r in rows if r["recipe"] == recipe]) for recipe in recipes]
    deviations = [np.std([r[key] for r in rows if r["recipe"] == recipe], ddof=1) for recipe in recipes]
    ax.errorbar(range(len(recipes)), means, yerr=deviations, fmt="ks", capsize=6, label="mean +/- sample SD", markersize=7)
    ax.set(xticks=range(len(recipes)), xticklabels=recipes, ylabel=ylabel, title="Three paired training seeds")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def verified_confirmation(path):
    result = read_json(path)
    recorded = result.pop("confirmation_id", None)
    if not recorded or digest(result) != recorded:
        raise ValueError("Confirmation record is missing or was modified")
    result["confirmation_id"] = recorded
    if result["kind"] != "confirm" or tuple(result["seeds"]) != SEEDS:
        raise ValueError("Unexpected confirmation protocol")
    return result


def verified_lock(path):
    lock = read_json(path)
    recorded = lock.pop("lock_id", None)
    if not recorded or digest(lock) != recorded:
        raise ValueError("Runner candidate lock is missing or was modified")
    lock["lock_id"] = recorded
    if lock["test_status"] != HISTORICAL_STATUS:
        raise ValueError("Unexpected test protocol in runner lock")
    return lock


def paired_bootstrap(baseline_errors, candidate_errors, repetitions=5000, seed=20261113):
    """Conditional image bootstrap; positive delta means lower candidate MAE."""
    baseline_errors = np.asarray(baseline_errors, np.float64)
    candidate_errors = np.asarray(candidate_errors, np.float64)
    if baseline_errors.ndim != 1 or baseline_errors.shape != candidate_errors.shape or not len(baseline_errors):
        raise ValueError("Bootstrap requires same-length one-dimensional paired errors")
    if not np.isfinite(baseline_errors).all() or not np.isfinite(candidate_errors).all() or repetitions < 100:
        raise ValueError("Invalid bootstrap values/repetitions")
    differences = baseline_errors-candidate_errors
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 128):
        size = min(128, repetitions-start)
        indices = rng.integers(0, len(differences), (size, len(differences)))
        means[start:start+size] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return {"n_images": len(differences), "repetitions": repetitions, "rng_seed": seed,
            "delta_definition": "baseline MAE minus candidate MAE; positive favors candidate",
            "mae_improvement": float(differences.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
            "interval_excludes_zero_in_improvement_direction": bool(lower > 0),
            "bootstrap_unit": "image", "method": "paired percentile bootstrap",
            "limitation": "Conditional on these fitted models; does not capture training-seed uncertainty or unresolved same-person image correlation."}


def load_locked_evaluations(evaluations_root, lock, confirmation):
    required = {(r["recipe"], r["seed"]): r for r in confirmation["runs"]}
    entries = {}
    for entry in lock["entries"]:
        key = (entry["config"]["id"], entry["config"]["seed"])
        if key in required:
            expected = required[key]
            if key in entries or entry["fingerprint"] != expected["fingerprint"] or entry["checkpoint_sha256"] != expected["best_checkpoint_sha256"]:
                raise ValueError("Frozen confirmation differs from runner lock")
            if abs(entry["validation_mae"]-expected["validation_mae"]) > 1e-12:
                raise ValueError("Locked validation selection changed")
            if entry["config"].get("smoke", False):
                raise ValueError("A smoke run cannot enter final comparison")
            entries[key] = entry
    if set(entries) != set(required):
        raise ValueError("All six baseline/winner seed checkpoints must be in the lock")
    results = {}
    paths = sorted(Path(evaluations_root).rglob("evaluation.json"))
    if not paths:
        raise FileNotFoundError("No completed evaluate-test outputs. This report tool never starts evaluation.")
    for path in paths:
        metadata = read_json(path)
        if metadata.get("lock_id") != lock["lock_id"]:
            continue
        matches = [key for key, entry in entries.items() if metadata.get("onnx_sha256") == entry["onnx_sha256"]
                   and metadata.get("checkpoint_sha256") == entry["checkpoint_sha256"]]
        if not matches:
            raise ValueError(f"Evaluation result does not match a confirmed locked checkpoint: {path}")
        if metadata.get("test_status") != HISTORICAL_STATUS or metadata.get("target_scale") != [1, 5]:
            raise ValueError("Evaluation scale/protocol does not match the fixed historical benchmark")
        prediction_path = path.parent/"test_predictions.csv"
        predictions = read_predictions(prediction_path, "test")
        computed = check_metric(predictions, metadata["test_raw"])
        if computed["n"] != 1100:
            raise ValueError("A completed official fold 1 evaluation must contain 1100 predictions")
        for key in matches:
            if key in results:
                if results[key]["predictions_sha256"] != sha256(prediction_path):
                    raise ValueError("Conflicting completed evaluations for the same locked checkpoint")
                continue
            results[key] = {"entry": entries[key], "metadata": metadata, "predictions": predictions,
                            "computed": computed, "path": str(path.resolve()), "evaluation_sha256": sha256(path),
                            "predictions_sha256": sha256(prediction_path)}
    missing = sorted(set(required)-set(results))
    if missing:
        raise FileNotFoundError(f"Missing completed locked evaluate-test outputs: {missing}; no evaluation was triggered")
    return results


def final_report(args):
    if bool(args.historical_evaluation) != bool(args.historical_predictions):
        raise ValueError("Supply both historical evaluation and predictions, or neither")
    confirmation = verified_confirmation(args.confirmation)
    lock = verified_lock(args.lock)
    evaluated = load_locked_evaluations(args.evaluations_root, lock, confirmation)
    baseline, winner = confirmation["baseline_recipe"], confirmation["winner_recipe"]
    reference = evaluated[(baseline, SEEDS[0])]["predictions"]
    names = sorted(reference)
    y = np.array([reference[n][0] for n in names])
    errors, rows, predictions_by_key = {}, [], {}
    max_target_difference = 0.
    for recipe in (baseline, winner):
        for seed in SEEDS:
            result = evaluated[(recipe, seed)]
            _, _, _, prediction, target_difference = align(reference, result["predictions"])
            max_target_difference = max(max_target_difference, target_difference)
            errors[(recipe, seed)] = np.abs(y-prediction)
            predictions_by_key[(recipe, seed)] = prediction
            rows.append({"recipe": recipe, "seed": seed, **result["computed"],
                         "validation_mae": result["entry"]["validation_mae"],
                         "evaluation_json": result["path"], "checkpoint_sha256": result["entry"]["checkpoint_sha256"],
                         "onnx_sha256": result["entry"]["onnx_sha256"]})
    method_summaries = {recipe: {metric: sample_summary([r[metric] for r in rows if r["recipe"] == recipe])
                               for metric in ("mae", "rmse", "pearson", "spearman")
                               if all(r[metric] is not None for r in rows if r["recipe"] == recipe)}
                        for recipe in (baseline, winner)}
    baseline_mean_errors = np.stack([errors[(baseline, seed)] for seed in SEEDS]).mean(axis=0)
    winner_mean_errors = np.stack([errors[(winner, seed)] for seed in SEEDS]).mean(axis=0)
    method_ci = paired_bootstrap(baseline_mean_errors, winner_mean_errors, args.bootstrap, args.bootstrap_seed)
    method_ci["seed_aggregation"] = "Mean of per-model absolute errors across the three fixed seeds; not an ensemble prediction"
    selected = confirmation["deployment_checkpoint"]
    chosen_key = (selected["recipe"], selected["seed"])
    chosen = evaluated[chosen_key]
    deployment = {"status": "unavailable_without_historical_prediction_pair", "candidate": selected,
                  "candidate_test_raw": chosen["computed"],
                  "note": "The current-method R0 three-seed reference is distinct from the historical fixed deployed model."}
    old_errors = None
    if args.historical_evaluation:
        historical = read_json(args.historical_evaluation)
        historical_predictions = read_predictions(args.historical_predictions, "test", historical=True)
        historical_computed = check_metric(historical_predictions, historical["test_raw"])
        _, _, _, old_prediction, label_difference = align(reference, historical_predictions)
        old_errors = np.abs(y-old_prediction)
        deployment = {"status": "paired_historical_fixed_model_comparison", "candidate": selected,
                      "candidate_test_raw": chosen["computed"], "historical_recorded_test_raw": historical["test_raw"],
                      "historical_recomputed_metrics_on_its_recorded_targets": historical_computed,
                      "historical_model_sha256": historical.get("model_sha256"),
                      "historical_evaluation_sha256": sha256(args.historical_evaluation),
                      "historical_predictions_sha256": sha256(args.historical_predictions),
                      "historical_vs_current_target_max_difference": label_difference,
                      "paired_comparison": paired_bootstrap(old_errors, errors[chosen_key], args.bootstrap, args.bootstrap_seed),
                      "comparison_scope": "One validation-selected candidate versus the saved fixed v3.1 model predictions; not a comparison against a selected R0 seed."}
    result = {**base_report("final"), "confirmation_id": confirmation["confirmation_id"], "lock_id": lock["lock_id"],
              "baseline_recipe": baseline, "winner_recipe": winner, "seeds": list(SEEDS),
              "historical_test_status": HISTORICAL_STATUS, "test_results_preexisted": True,
              "validation_gate": {k: confirmation[k] for k in ("mean_validation_mae_improvement", "minimum_mean_improvement", "validation_gate_passed")},
              "method_comparison": {"per_seed_results": rows, "three_seed_summary": method_summaries,
                                    "paired_image_bootstrap_of_mean_seed_errors": method_ci,
                                    "paired_seed_mae_improvements": [{"seed": seed,
                                        "mae_improvement": float(errors[(baseline, seed)].mean()-errors[(winner, seed)].mean())} for seed in SEEDS]},
              "deployment_comparison": deployment, "current_target_max_difference": max_target_difference,
              "evaluation_sources": [{"recipe": key[0], "seed": key[1], "path": item["path"],
                                      "evaluation_sha256": item["evaluation_sha256"], "predictions_sha256": item["predictions_sha256"]}
                                     for key, item in evaluated.items()],
              "release_decision": "Not made by this report tool; mobile integration, latency and quality gates remain separate."}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output/"final_comparison.json"
    if final_path.exists():
        raise FileExistsError("Preserve completed reports; use a new output directory")
    write_csv(output/"test_three_seed_comparison.csv", rows, ["recipe", "seed", "n", "mae", "rmse", "pearson", "spearman",
              "validation_mae", "checkpoint_sha256", "onnx_sha256", "evaluation_json"])
    paired_rows = []
    for index, name in enumerate(names):
        row = {"name": name, "target_1_5": float(y[index]), "baseline_mean_seed_absolute_error": float(baseline_mean_errors[index]),
               "winner_mean_seed_absolute_error": float(winner_mean_errors[index]),
               "method_error_improvement": float(baseline_mean_errors[index]-winner_mean_errors[index]),
               "chosen_candidate_absolute_error": float(errors[chosen_key][index])}
        if old_errors is not None:
            row.update(historical_fixed_absolute_error=float(old_errors[index]),
                       fixed_deployment_error_improvement=float(old_errors[index]-errors[chosen_key][index]))
        paired_rows.append(row)
    write_csv(output/"paired_image_errors.csv", paired_rows, list(paired_rows[0]))
    draw_seed_plot(rows, "mae", "Historical SCUT test MAE (1-5)", (baseline, winner), output/"test_three_seeds.png")
    plt = plots()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].scatter(y, predictions_by_key[chosen_key], s=9, alpha=.35)
    axes[0].plot([1, 5], [1, 5], "k--", linewidth=1)
    axes[0].set(xlabel="Recorded rater mean (1-5)", ylabel="Raw prediction", title=f"Validation-selected {chosen_key[0]} / seed {chosen_key[1]}")
    differences = baseline_mean_errors-winner_mean_errors
    histogram_range = (float(differences.mean())-.05, float(differences.mean())+.05) if np.ptp(differences) < 1e-6 else None
    axes[1].hist(differences, bins=40, range=histogram_range, color="#3c7d9b", alpha=.8)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="R0 minus candidate mean-seed absolute error", ylabel="Images", title="Paired method comparison (not ensemble)")
    fig.tight_layout()
    fig.savefig(output/"predictions_and_paired_errors.png")
    plt.close(fig)
    write_json(final_path, result)
    method_old = method_summaries[baseline]["mae"]
    method_new = method_summaries[winner]["mae"]
    text = ["SCUT历史基准最终结果说明", "", "本报告只读取已由runner候选锁绑定、且已经完成的evaluate-test输出，没有启动推理或重新选择模型。",
            "", "一、训练方法比较：本批统一连续NCHW流程，3个相同seed配对。",
            f"{baseline}：测试MAE {method_old['mean']:.6f} ± {method_old['sample_standard_deviation']:.6f}。",
            f"{winner}：测试MAE {method_new['mean']:.6f} ± {method_new['sample_standard_deviation']:.6f}。",
            f"平均MAE降低 {method_ci['mae_improvement']:.6f}；{args.bootstrap}次按图片配对bootstrap的95%区间 [{method_ci['ci95_lower']:.6f}, {method_ci['ci95_upper']:.6f}]。",
            "以上先对每个模型计算绝对误差，再平均3个seed的误差，没有平均预测来伪装成集成模型。",
            f"开发阶段平均验证MAE改善门槛 {confirmation['minimum_mean_improvement']:.6f}：{'通过' if confirmation['validation_gate_passed'] else '未通过'}。",
            "", "二、固定部署模型比较：与方法三种子比较分开。",
            f"候选由验证集预先选定：{chosen_key[0]}-seed{chosen_key[1]}；其测试MAE {chosen['computed']['mae']:.6f}。"]
    if old_errors is not None:
        ci = deployment["paired_comparison"]
        text += [f"旧固定模型历史记录测试MAE {deployment['historical_recorded_test_raw']['mae']:.6f}。",
                 f"同1100张图片配对后，候选MAE降低 {ci['mae_improvement']:.6f}；95%区间 [{ci['ci95_lower']:.6f}, {ci['ci95_upper']:.6f}]。",
                 "旧固定模型来自保存的历史逐图预测；不把新流程某个R0 seed当成旧App模型。"]
    else:
        text.append("没有提供旧固定模型的历史逐图预测，无法作此配对部署比较，不根据新R0结果代替该结论。")
    text += ["", "三、证据边界。",
             "这是已经观察过的SCUT官方fold1历史基准，不是全新盲测。数据评分有主观性。",
             "MAE/RMSE按原始1–5分量表；Pearson/Spearman是相关系数，不是百分比准确率。",
             "bootstrap以图片为单位，条件于已训练的固定模型；不能覆盖训练随机性，也无法消除未知同人多图相关性。",
             "仅3个seed，标准差用于展示波动，不夸大为充分的训练方法显著性证明。",
             "没有在本报告中测得属性F1、分割mIoU或真机识别准确率。候选是否发布还需端侧和性能验收。"]
    write_text(output/"结果说明.txt", "\n".join(text)+"\n")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--runs-root", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("confirm")
    c.add_argument("--runs-root", type=Path, required=True)
    c.add_argument("--winner", choices=RECIPES[1:], required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--minimum-improvement", type=float, default=.005)
    f = sub.add_parser("final")
    f.add_argument("--confirmation", type=Path, required=True)
    f.add_argument("--lock", type=Path, required=True)
    f.add_argument("--evaluations-root", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--historical-evaluation", type=Path)
    f.add_argument("--historical-predictions", type=Path)
    f.add_argument("--bootstrap", type=int, default=5000)
    f.add_argument("--bootstrap-seed", type=int, default=20261113)
    return p


def main():
    args = parser().parse_args()
    if args.command == "screen":
        result = screening(args)
        brief = {k: result[k] for k in ("kind", "all_six_completed", "recommended_recipe", "provisional_best_completed_recipe")}
    elif args.command == "confirm":
        result = confirmation(args)
        brief = {k: result[k] for k in ("kind", "winner_recipe", "mean_validation_mae_improvement", "validation_gate_passed", "deployment_checkpoint")}
    else:
        result = final_report(args)
        brief = {"kind": "final", "method_comparison": result["method_comparison"]["paired_image_bootstrap_of_mean_seed_errors"],
                 "deployment_comparison_status": result["deployment_comparison"]["status"]}
    print(json.dumps(brief, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
