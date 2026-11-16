"""Record the small boundary/resume/export tests and freeze source checksums."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ["FACESCORE_INTEGRATION"] = "1"
import runner
import test_runner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-run", type=Path)
    p.add_argument("--test-manifest", type=Path)
    args = p.parse_args()
    started = time.monotonic()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_runner)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    evidence = {"created_utc": runner.utc_now(), "tests_run": result.testsRun,
                "failures": len(result.failures), "errors": len(result.errors),
                "skipped": len(result.skipped), "passed": result.wasSuccessful(),
                "test_targets_parsed": False, "long_training_started": False,
                "test_log": log.getvalue()}
    if result.wasSuccessful() and args.baseline_run:
        directory = args.baseline_run.resolve()
        root = directory.parent.parent
        run = json.loads((directory/"run.json").read_text(encoding="utf-8"))
        summary = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed":
            raise ValueError("Compatibility check must not start/resume an unfinished baseline")
        resumed = runner.train(runner.parser().parse_args([
            "train", "--plan", str(root/"registered_plan.json"), "--config", "R0", "--seed", "42",
            "--data-dir", run["data_dir"], "--dev-manifest", run["development_manifest"],
            "--output", str(directory), "--cache-dir", str(root/"feature-cache"), "--max-hours", "3",
            "--threads", "1", "--resume"]))
        exported = json.loads((directory/"export_metadata.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary)/"compatibility-only-lock.json"
            locked = runner.create_lock(argparse.Namespace(
                output=lock_path, runs=[directory], test_manifest=args.test_manifest,
                rationale="Temporary schema compatibility check only; no test evaluation"))
            verified, entry = runner.validate_lock(lock_path, directory, args.test_manifest)
        evidence["legacy_r0_compatibility"] = {
            "completed_resume_returned_without_training": resumed["status"] == "completed",
            "best_checkpoint_sha256": exported["checkpoint_sha256"],
            "validation_export_samples": len(exported["parity"]),
            "onnx_max_absolute_difference": exported["maximum_absolute_difference"],
            "temporary_lock_created_and_verified": verified["lock_id"] == locked["lock_id"],
            "test_scores_evaluated": False}
    evidence["wall_seconds"] = time.monotonic()-started
    directory = Path(__file__).resolve().parent
    evidence["source_sha256"] = {path.name: runner.file_sha256(path) for path in sorted(directory.iterdir())
                                 if path.suffix in (".py", ".txt")}
    evidence["runner_training_semantics_frozen"] = True
    evidence["fixed_input_layout"] = "contiguous NCHW for all screening/confirmation configurations"
    runner.atomic_json(args.output, evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k != "test_log"}, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
