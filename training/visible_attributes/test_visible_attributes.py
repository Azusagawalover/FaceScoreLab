"""Small synthetic implementation tests; no real held-out labels are accessed."""
import argparse
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn

import common
import runner


class VisibleAttributeTests(unittest.TestCase):
    def test_subset_protocol_rejects_same_count_but_different_names_or_seed(self):
        rows = [{"name": "a.jpg", "split": "train"}, {"name": "b.jpg", "split": "validation"}]
        protocol = {"seed": 42, "selected_names": {"train": ["a.jpg"], "validation": ["b.jpg"]}}
        runner.validate_subset_protocol(protocol, rows, "development")
        with self.assertRaisesRegex(ValueError, "names differ"):
            runner.validate_subset_protocol(protocol, [{"name": "wrong.jpg", "split": "train"}, rows[1]], "development")
        with self.assertRaisesRegex(ValueError, "sampling seed"):
            runner.validate_subset_protocol({**protocol, "seed": 17}, rows, "development")

    def test_zero_is_known_negative_unknown_is_masked_and_signed_values_are_rejected(self):
        self.assertEqual(common.parse_label("0"), (0., 1.))
        self.assertEqual(common.parse_label("1"), (1., 1.))
        for unknown in ("", "NA", "?"):
            self.assertEqual(common.parse_label(unknown), (0., 0.))
        self.assertEqual(common.parse_label("1", "0"), (0., 0.))
        with self.assertRaises(ValueError):
            common.parse_label("-1")
        with self.assertRaises(ValueError):
            common.parse_label("1", "2")

    def test_unknown_gradients_are_zero_and_class_counts_exclude_unknowns(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        masks = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
        loss, valid = common.masked_bce(logits, labels, masks, torch.tensor([1., 3., 1.]))
        loss.backward()
        self.assertEqual(valid.item(), 4)
        self.assertEqual(logits.grad[0, 1].item(), 0)
        self.assertEqual(logits.grad[1, 2].item(), 0)
        counts = common.class_counts(labels.numpy(), masks.numpy())
        self.assertEqual(counts[1]["positive"], 1)
        self.assertEqual(counts[1]["negative"], 0)
        self.assertEqual(counts[1]["unknown"], 1)

    def test_thresholds_use_masks_and_no_face_is_coverage_loss_not_negative_prediction(self):
        y = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], float)
        masks = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], float)
        scores = np.array([[.2, .3, .4], [.8, .7, .6], [.99, .99, .99]])
        thresholds = common.choose_thresholds(y, masks, scores)
        measured = common.evaluate_scores(y, masks, scores, thresholds)
        self.assertEqual(measured["macro_f1"], 1)
        scores[2] = np.nan
        masks[2] = 1
        measured = common.evaluate_scores(y, masks, scores, thresholds, [True, True, False])
        self.assertEqual(measured["coverage"]["images_total"], 3)
        self.assertEqual(measured["coverage"]["images_with_output"], 2)
        self.assertEqual(measured["per_label"][0]["known_available"], 2)
        self.assertEqual(measured["per_label"][0]["known_total"], 3)
        with self.assertRaisesRegex(ValueError, "available scores"):
            common.evaluate_scores(y, masks, scores, thresholds, [True, True, True])

    def test_prepare_uses_mediapipe_largest_face_and_retains_failed_detection_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/"dev.csv"
            protocol = root/"protocol.json"
            protocol.write_text('{"synthetic_fixture": true}')
            task = root/"face.task"
            task.write_bytes(b"synthetic fixture; detector factory is mocked")
            fields = ["name", "split", "image_path", *common.LABELS]
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                for i, split in enumerate(("train", "validation")):
                    image = root/f"image{i}.jpg"
                    Image.new("RGB", (80, 60), (50+i*20, 100, 150)).save(image)
                    writer.writerow([image.name, split, str(image.resolve()), i, 1-i, i])
            small = [SimpleNamespace(x=.2, y=.2), SimpleNamespace(x=.3, y=.3)]
            large = [SimpleNamespace(x=.1, y=.1), SimpleNamespace(x=.8, y=.8)]
            detector = mock.Mock()
            detector.detect.side_effect = [SimpleNamespace(face_landmarks=[small, large]), SimpleNamespace(face_landmarks=[])]
            context = mock.MagicMock()
            context.__enter__.return_value = detector
            args = argparse.Namespace(manifest=manifest, subset_protocol=protocol, face_task=task, output=root/"prepared", max_hours=.1, smoke=True)
            with mock.patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", return_value=context):
                result = runner.prepare(args)
            metadata, rows = runner.read_prepared(root/"prepared", "development")
            self.assertEqual(result["detected_counts"], {"train": 1, "validation": 0})
            self.assertEqual(len(rows), 2)
            self.assertFalse(rows[1]["detected"])
            self.assertEqual(rows[1]["masks"], [1., 1., 1.])
            record = common.read_json(root/"prepared/crop_records.json")["image0.jpg"]
            expected = common.face_crop_rect(np.array([[8., 6.], [64., 48.]], np.float32), 80, 60)
            self.assertEqual(record["crop"], expected)
            with Image.open(rows[0]["crop_path"]) as crop:
                self.assertEqual(crop.size, (224, 224))

    def test_heldout_lock_is_checked_before_manifest_or_labels(self):
        args = argparse.Namespace(lock=Path("missing-lock"))
        with mock.patch.object(runner, "check_lock", side_effect=ValueError("frozen boundary first")):
            with mock.patch.object(runner, "load_manifest", side_effect=AssertionError("must not read labels")):
                with self.assertRaisesRegex(ValueError, "frozen boundary"):
                    runner.prepare(args, heldout=True)

    def test_feature_cache_invalidates_when_crop_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/"crop.png"
            Image.new("RGB", (224, 224), (1, 2, 3)).save(path)
            row = {"name": "synthetic.jpg", "split": "train", "crop_path": str(path), "crop_sha256": common.sha256(path)}
            metadata = {"manifest_sha256": "synthetic", "preprocessing": {"synthetic": True}}
            model = nn.Module()
            model.features = nn.Sequential(nn.Conv2d(3, 2, 1))
            model.pool = nn.AdaptiveAvgPool2d(1)
            torch.set_num_threads(1)
            _, first = runner.feature_cache([row], metadata, model, root/"cache", False, common.Budget(.1))
            Image.new("RGB", (224, 224), (4, 5, 6)).save(path)
            with self.assertRaisesRegex(ValueError, "content changed"):
                runner.feature_cache([row], metadata, model, root/"cache", False, common.Budget(.1))
            row["crop_sha256"] = common.sha256(path)
            _, second = runner.feature_cache([row], metadata, model, root/"cache", False, common.Budget(.1))
            self.assertNotEqual(first["key"], second["key"])

    def synthetic_prepared(self, root, mode="development"):
        prepared = root/"prepared"
        (prepared/"crops").mkdir(parents=True)
        rng = np.random.default_rng(7)
        rows = []
        for i in range(88):
            path = prepared/"crops"/f"synthetic{i:03d}.png"
            Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(path)
            rows.append({"name": f"synthetic{i:03d}.jpg", "split": "train" if i < 80 else "validation",
                         "image_path": str(path.resolve()), "detected": i not in (0, 80), "crop_path": str(path.resolve()),
                         "source_sha256": common.sha256(path), "crop_sha256": common.sha256(path),
                         "labels": [float(i%2), float((i//2)%2), float((i//3)%2)],
                         "masks": [0. if i%7 == 0 else 1., 1., 1.]})
        runner.write_prepared_manifest(prepared/"development.csv", rows)
        common.atomic_json(prepared/"metadata.json", {
            "format": common.VERSION, "mode": "development", "status": "completed", "smoke": True,
            "manifest_name": "development.csv", "manifest_sha256": common.sha256(prepared/"development.csv"),
            "preprocessing": common.preprocessing_contract("synthetic-task-only"),
            "counts": {"train": 80, "validation": 8}, "detected_counts": {"train": 79, "validation": 7}, "wall_seconds": 0.})
        return prepared

    def test_real_model_small_training_resume_and_validation_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.synthetic_prepared(root)
            base = ["train", "--prepared", str(prepared), "--cache-dir", str(root/"cache"), "--threads", "1",
                    "--max-hours", ".1", "--smoke", "--tail-epochs", "1"]
            full = runner.train(runner.parser().parse_args(base+["--output", str(root/"full")]))
            paused = runner.train(runner.parser().parse_args(base+["--output", str(root/"paused"), "--stop-after-steps", "1"]))
            self.assertEqual(paused["status"], "paused_for_pilot")
            self.assertEqual(paused["epochs_completed"], 0)
            resumed = runner.train(runner.parser().parse_args(base+["--output", str(root/"paused"), "--resume"]))
            self.assertEqual(resumed["best_validation_macro_f1"], full["best_validation_macro_f1"])
            self.assertEqual(resumed["thresholds"], full["thresholds"])
            a = common.load_checkpoint(root/"full/last.pt")
            b = common.load_checkpoint(root/"paused/last.pt")
            self.assertTrue(all(torch.equal(a["model_state"][name], b["model_state"][name]) for name in a["model_state"]))
            self.assertEqual(full["validation"]["coverage"]["images_total"], 8)
            self.assertEqual(full["validation"]["coverage"]["images_with_output"], 7)
            export = runner.export_model(runner.parser().parse_args(["export", "--run", str(root/"full"), "--prepared", str(prepared), "--threads", "1"]))
            self.assertLess(export["max_logit_difference"], 1e-4)
            self.assertEqual(len(export["parity"]), 7)
            android = common.read_json(root/"full/android_metadata.json")
            self.assertEqual(android["labels"], list(common.LABELS))
            self.assertEqual(android["output"]["shape"], ["N", 3])
            with self.assertRaisesRegex(ValueError, "non-smoke"):
                runner.lock_candidate(runner.parser().parse_args(["lock", "--run", str(root/"full"), "--heldout-count", "2000",
                    "--heldout-seed", "42", "--output", str(root/"lock.json")]))

    def test_heldout_evaluation_keeps_frozen_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/"crop.png"
            Image.new("RGB", (224, 224), (20, 40, 60)).save(path)
            rows = [{"name": "synthetic.jpg", "split": "test", "detected": True, "crop_path": str(path),
                     "crop_sha256": common.sha256(path), "labels": [0., 1., 1.], "masks": [1., 1., 1.]}]
            lock = {"lock_id": "synthetic-lock", "preprocessing": {"synthetic": True},
                    "heldout_selection": {"sample_count": 1}, "thresholds": [.3, .6, .7], "onnx_sha256": "synthetic-onnx"}
            metadata = {"lock_id": lock["lock_id"], "preprocessing": lock["preprocessing"], "manifest_sha256": "synthetic"}
            session = mock.Mock()
            session.run.return_value = [np.array([[-2., 2., 2.]], np.float32)]
            args = argparse.Namespace(lock=root/"lock", run=root/"run", prepared=root/"prepared", output=root/"out", max_hours=.1, threads=1)
            with mock.patch.object(runner, "check_lock", return_value=lock), mock.patch.object(runner, "read_prepared", return_value=(metadata, rows)):
                with mock.patch("onnxruntime.InferenceSession", return_value=session), mock.patch.object(runner, "choose_thresholds", side_effect=AssertionError("Never tune on held-out targets")):
                    result = runner.evaluate_heldout(args)
            self.assertEqual(result["thresholds"], [.3, .6, .7])
            self.assertFalse(result["thresholds_retuned_on_test"])
            self.assertEqual(result["heldout"]["macro_f1"], 2/3)  # No positive Smiling support: F1 uses zero-division=0.


if __name__ == "__main__":
    unittest.main(verbosity=2)
