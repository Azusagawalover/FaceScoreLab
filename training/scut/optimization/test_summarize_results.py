"""Small synthetic checks. No real SCUT test targets are read."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import summarize_results as report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runs = self.root/"runs"
        self.runs.mkdir()
        self.make_runs()

    def tearDown(self):
        self.temporary.cleanup()

    def make_predictions(self, path, split, target, prediction, historical=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            columns = ["name", "target_1_5", "prediction_raw", "prediction_clipped", "absolute_error"]
            if not historical:
                columns.insert(1, "split")
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index, (y, p) in enumerate(zip(target, prediction)):
                row = {"name": f"synthetic{index:04d}.jpg", "target_1_5": float(y), "prediction_raw": float(p),
                       "prediction_clipped": float(np.clip(p, 1, 5)), "absolute_error": float(abs(y-p))}
                if not historical:
                    row["split"] = split
                writer.writerow(row)

    def make_runs(self):
        values = {"R0": .30, "R1": .28, "R2": .25, "R3": .27, "R4": .265, "R5": .29}
        records = [(recipe, 42, error) for recipe, error in values.items()]
        records += [("R0", 17, .31), ("R0", 2026, .29), ("R2", 17, .245), ("R2", 2026, .255)]
        y = np.linspace(1.5, 4., 8)
        for recipe, seed, error in records:
            directory = self.runs/f"{recipe}-seed{seed}"
            directory.mkdir()
            checkpoint_hash = hashlib.sha256(f"{recipe}:{seed}".encode()).hexdigest()
            metadata = {"config": {"id": recipe, "seed": seed, "smoke": False}, "status": "completed",
                        "test_evaluated": False, "smoke_not_for_release": False,
                        "validation_mae": error, "best_stage": "late_finetune", "best_epoch": 3,
                        "epochs_completed": 5, "wall_seconds": 10., "fingerprint": f"fp-{recipe}-{seed}",
                        "best_checkpoint_sha256": checkpoint_hash, "counts": {"train": 16, "validation": 8}}
            report.write_json(directory/"metrics.json", metadata)
            report.write_json(directory/"training_log.json", [{"validation": {"mae": error+.1}}, {"validation": {"mae": error}}])
            self.make_predictions(directory/"validation_predictions.csv", "validation", y, y+error)
            # Screen/confirm must not open these poisoned test artifacts.
            (directory/"test_predictions.csv").write_text("DO_NOT_READ_REAL_TEST_LABELS", encoding="utf-8")
            (directory/"evaluation.json").write_text("NOT_VALID_JSON_TEST_GUARD", encoding="utf-8")

    def confirm(self):
        destination = self.root/"confirmation"
        return report.confirmation(argparse.Namespace(runs_root=self.runs, winner="R2", output=destination,
                                                       minimum_improvement=.005))

    def make_final_inputs(self, confirmation):
        entries = []
        evaluations = self.root/"evaluations"
        y = np.linspace(1.5, 4., 1100)
        for row in confirmation["runs"]:
            recipe, seed = row["recipe"], row["seed"]
            entries.append({"run": row["run"], "fingerprint": row["fingerprint"],
                            "checkpoint_sha256": row["best_checkpoint_sha256"],
                            "onnx_sha256": hashlib.sha256(f"onnx:{recipe}:{seed}".encode()).hexdigest(),
                            "validation_mae": row["validation_mae"], "config": {"id": recipe, "seed": seed, "smoke": False}})
        lock = {"format": "facescore-optimization-v1", "created_utc": report.now(),
                "selection_rationale": "Synthetic fixture only", "selection_basis": "development validation only",
                "test_status": report.HISTORICAL_STATUS, "combined_manifest_sha256": "synthetic-not-real-data", "entries": entries}
        lock["lock_id"] = report.digest(lock)
        lock_path = self.root/"lock.json"
        report.write_json(lock_path, lock)
        for entry in entries:
            recipe, seed = entry["config"]["id"], entry["config"]["seed"]
            sign = -1 if seed == 42 else 1
            error = .3 if recipe == "R0" else {17: .2, 42: .1, 2026: .3}[seed]
            prediction = y + sign*error
            destination = evaluations/f"{recipe}-seed{seed}"
            destination.mkdir(parents=True)
            self.make_predictions(destination/"test_predictions.csv", "test", y, prediction)
            report.write_json(destination/"evaluation.json", {
                "lock_id": lock["lock_id"], "checkpoint_sha256": entry["checkpoint_sha256"],
                "onnx_sha256": entry["onnx_sha256"], "test_status": report.HISTORICAL_STATUS,
                "target_scale": [1, 5], "test_raw": report.regression_metrics(y, prediction)})
        historical_predictions = self.root/"historical.csv"
        historical_evaluation = self.root/"historical.json"
        self.make_predictions(historical_predictions, "test", y, y+.34, historical=True)
        report.write_json(historical_evaluation, {"test_raw": report.regression_metrics(y, y+.34), "model_sha256": "old-fixed-model"})
        return argparse.Namespace(confirmation=self.root/"confirmation/confirmation.json", lock=lock_path,
            evaluations_root=evaluations, output=self.root/"final", historical_evaluation=historical_evaluation,
            historical_predictions=historical_predictions, bootstrap=5000, bootstrap_seed=20261113)

    def test_screen_reads_development_only_and_requires_all_six_before_recommendation(self):
        result = report.screening(argparse.Namespace(runs_root=self.runs, seed=42, output=self.root/"screen"))
        self.assertEqual(result["recommended_recipe"], "R2")
        metadata_path = self.runs/"R5-seed42/metrics.json"
        metadata = report.read_json(metadata_path)
        metadata["status"] = "running"
        report.write_json(metadata_path, metadata)
        partial = report.screening(argparse.Namespace(runs_root=self.runs, seed=42, output=self.root/"screen"))
        self.assertFalse(partial["all_six_completed"])
        self.assertIsNone(partial["recommended_recipe"])

    def test_confirmation_three_seed_gate_and_validation_only_selection(self):
        result = self.confirm()
        self.assertEqual(result["deployment_checkpoint"]["seed"], 17)
        self.assertAlmostEqual(result["mean_validation_mae_improvement"], .05)
        self.assertTrue(result["validation_gate_passed"])
        self.assertAlmostEqual(result["validation_summary"]["R0"]["sample_standard_deviation"], .01)
        with self.assertRaises(FileExistsError):
            self.confirm()

    def test_bootstrap_is_paired_and_target_mismatch_is_rejected(self):
        ci = report.paired_bootstrap(np.array([.3, .4, .5]), np.array([.2, .3, .4]), 5000, 1)
        self.assertAlmostEqual(ci["mae_improvement"], .1)
        self.assertAlmostEqual(ci["ci95_lower"], .1)
        self.assertAlmostEqual(ci["ci95_upper"], .1)
        with self.assertRaisesRegex(ValueError, "targets differ"):
            report.align({"a": (2., 3.)}, {"a": (2.1, 3.)})
        with self.assertRaisesRegex(ValueError, "same image names"):
            report.align({"a": (2., 3.)}, {"b": (2., 3.)})

    def test_final_refuses_missing_evaluation_and_tampered_lock(self):
        args = self.make_final_inputs(self.confirm())
        (args.evaluations_root/"R2-seed17/evaluation.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing completed"):
            report.final_report(args)
        lock = report.read_json(args.lock)
        lock["selection_rationale"] = "tampered"
        report.write_json(args.lock, lock)
        with self.assertRaisesRegex(ValueError, "modified"):
            report.final_report(args)

    def test_final_uses_mean_seed_errors_not_ensemble_and_keeps_validation_choice(self):
        args = self.make_final_inputs(self.confirm())
        result = report.final_report(args)
        method = result["method_comparison"]["paired_image_bootstrap_of_mean_seed_errors"]
        self.assertAlmostEqual(method["mae_improvement"], .1)
        self.assertEqual(method["repetitions"], 5000)
        deployment = result["deployment_comparison"]
        self.assertEqual(deployment["candidate"]["seed"], 17)
        self.assertAlmostEqual(deployment["candidate_test_raw"]["mae"], .2)
        self.assertAlmostEqual(deployment["paired_comparison"]["mae_improvement"], .14)
        self.assertTrue((args.output/"结果说明.txt").exists())
        self.assertTrue((args.output/"test_three_seeds.png").stat().st_size > 1000)
        self.assertFalse(result["test_evaluation_triggered"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
