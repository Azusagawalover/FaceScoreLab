"""Boundary tests plus an opt-in tiny real-model uninterrupted/resume smoke."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn

import runner


def tiny_plan():
    return {"first_training_batch": {
        "batch_size": 8, "dropout": .2, "weight_decay": .01,
        "head_stage_lr": .0008, "tail_stage_head_lr": .00012, "tail_stage_backbone_lr": .00003,
        "tail_epoch_limit_except_original_baseline": 60, "early_stopping_patience_except_original_baseline": 8,
        "original_protocol_min_improvement": .00001, "new_protocol_min_improvement": .0001,
        "scheduler_factor": .5, "scheduler_patience": 3,
        "scheduler_min_head_lr": .000001, "scheduler_min_backbone_lr": .00000025,
        "single_run_compute_wall_hours_cap": 3}, "experiments": [{"id": x} for x in runner.KNOWN_CONFIGS]}


class BoundaryTests(unittest.TestCase):
    def test_stage_thresholds_and_legacy_r0_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp)/"plan.json"
            runner.atomic_json(plan, tiny_plan())
            original = runner.make_config(plan, "R0", 42)
            for name in runner.KNOWN_CONFIGS:
                config = runner.make_config(plan, name, 42)
                self.assertEqual(runner.minimum_improvement(config, 0), .00001)
                self.assertEqual(runner.minimum_improvement(config, 1), .00001 if name == "R0" else .0001)
            legacy = {k: v for k, v in original.items() if k not in ("head_min_improvement", "tail_min_improvement")}
            self.assertTrue(runner.legacy_r0_equivalent(legacy, original))
            self.assertEqual(runner.minimum_improvement(legacy, 0), .00001)
            self.assertFalse(runner.legacy_r0_equivalent(legacy, {**original, "seed": 17}))

    def test_development_boundary_rejects_combined_manifest_and_never_parses_test_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, development = root/"combined.csv", root/"development.csv"
            source.write_text("name,score,split,pixel_sha256\na.jpg,3,train,x\nb.jpg,4,validation,y\nsecret.jpg,DO_NOT_PARSE_TEST_TARGET,test,z\n", encoding="utf-8")
            record = runner.prepare_manifest(source, development)
            rows = runner.load_development_rows(development)
            self.assertEqual([r["name"] for r in rows], ["a.jpg", "b.jpg"])
            self.assertEqual(record["discarded_rows_without_parsing_targets"], 1)
            self.assertNotIn("secret", development.read_text())
            with self.assertRaisesRegex(ValueError, "development-only"):
                runner.load_development_rows(source)

    def test_cache_identity_changes_for_pixels_weights_split_and_preprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"crops").mkdir()
            image_path = root/"crops/a.jpg.png"
            Image.new("RGB", (224, 224), (1, 2, 3)).save(image_path)
            manifest = root/"dev.csv"
            manifest.write_text("fixture", encoding="utf-8")
            row = {"name": "a.jpg", "score": 3., "split": "train", "pixel_sha256": ""}
            config = {"preprocess": "legacy_facecrop_v1", "augmentation": "none", "seed": 42}
            early = nn.Conv2d(3, 2, 1).eval()
            def key(r=row, c=config):
                return runner.digest_json(runner.cache_spec(root, [r], manifest, c, early, runner.Budget(1)))
            first = key()
            Image.new("RGB", (224, 224), (1, 2, 4)).save(image_path)
            second = key()
            self.assertNotEqual(first, second)
            with torch.no_grad():
                early.weight.add_(.01)
            third = key()
            self.assertNotEqual(second, third)
            self.assertNotEqual(third, key({**row, "split": "validation"}))
            self.assertNotEqual(third, key(c={**config, "augmentation": "two_fixed_mild_views_v1"}))

    def test_square_crop_reconstructs_missing_border_from_original(self):
        original = Image.new("RGB", (100, 80), (250, 5, 5))
        points = np.array([[0, 0], [20, 20]], dtype=np.float32)
        result = runner.square_crop(original, points)
        self.assertEqual(result.size, (224, 224))
        self.assertEqual(result.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(result.getpixel((150, 150)), (250, 5, 5))

    def test_rng_and_optimizer_checkpoint_round_trip(self):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        aug = np.random.default_rng(42)
        model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2), nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x, y = torch.ones(6, 3), torch.ones(6, 1)
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x)-y)**2).mean()
            loss.backward()
            optimizer.step()
            return (random.random(), np.random.random(), aug.random())
        step()
        snapshot = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "rng": runner.capture_rng(aug)}
        expected_numbers = step()
        expected = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/"last.pt"
            runner.atomic_torch(target, snapshot)
            restored = runner.load_checkpoint(target)
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        runner.restore_rng(restored["rng"], aug)
        self.assertEqual(expected_numbers, step())
        self.assertTrue(all(torch.equal(expected[k], model.state_dict()[k]) for k in expected))

    def test_final_evaluation_checks_lock_before_opening_test_targets(self):
        args = argparse.Namespace(lock=Path("absent-lock.json"), run=Path("absent-run"),
                                  test_manifest=Path("DO_NOT_OPEN_TEST_TARGETS.csv"))
        with mock.patch.object(runner, "validate_lock", side_effect=ValueError("locked before labels")) as guard:
            with self.assertRaisesRegex(ValueError, "locked before labels"):
                runner.evaluate_test(args)
            guard.assert_called_once()

    def test_smoke_run_cannot_be_locked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner.atomic_json(root/"metrics.json", {"status": "completed", "smoke_not_for_release": True})
            runner.atomic_json(root/"export_metadata.json", {"smoke_not_for_release": True})
            args = argparse.Namespace(output=root/"lock.json", runs=[root])
            with self.assertRaisesRegex(ValueError, "non-smoke"):
                runner.create_lock(args)


@unittest.skipUnless(os.environ.get("FACESCORE_INTEGRATION") == "1", "Set FACESCORE_INTEGRATION=1 for tiny actual-model smoke")
class RealModelSmoke(unittest.TestCase):
    def test_uninterrupted_equals_mid_epoch_resume_and_export_uses_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/"data"
            (data/"crops").mkdir(parents=True)
            manifest = root/"development.csv"
            rng = np.random.default_rng(77)
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["name", "score", "split", "pixel_sha256"])
                for i in range(24):
                    name = f"fixture{i:02d}.jpg"
                    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                    Image.fromarray(image).save(data/"crops"/(name+".png"))
                    writer.writerow([name, 2.5 + (i % 5)*.2, "train" if i < 16 else "validation", ""])
            plan = root/"plan.json"
            runner.atomic_json(plan, tiny_plan())
            common = ["train", "--plan", str(plan), "--config", "R2", "--seed", "42", "--data-dir", str(data),
                      "--dev-manifest", str(manifest), "--cache-dir", str(root/"cache"), "--max-hours", ".2",
                      "--smoke", "--threads", "1"]
            full = root/"full"
            interrupted = root/"interrupted"
            first = runner.train(runner.parser().parse_args(common+["--output", str(full)]))
            stopped = runner.train(runner.parser().parse_args(common+["--output", str(interrupted), "--stop-after-batches", "1"]))
            self.assertEqual(stopped["status"], "paused_for_pilot")
            self.assertEqual(stopped["epochs_completed"], 0)
            resumed = runner.train(runner.parser().parse_args(common+["--output", str(interrupted), "--resume"]))
            self.assertEqual(first["status"], "completed")
            self.assertEqual(resumed["validation_mae"], first["validation_mae"])
            a = runner.load_checkpoint(full/"last.pt")
            b = runner.load_checkpoint(interrupted/"last.pt")
            self.assertTrue(all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"]))
            self.assertEqual(a["state"]["total_batches"], b["state"]["total_batches"])
            exported = runner.export_model(runner.parser().parse_args([
                "export", "--run", str(full), "--data-dir", str(data), "--dev-manifest", str(manifest), "--threads", "1"]))
            self.assertEqual(exported["parity_sample_split"], "validation")
            self.assertEqual(len(exported["parity"]), 8)
            self.assertFalse(exported["test_targets_opened"])
            self.assertLess(exported["maximum_absolute_difference"], 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
