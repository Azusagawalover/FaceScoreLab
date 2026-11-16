"""Local synthetic safety checks; never reads the actual CelebA test files."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from PIL import Image

import prepare_celeba_heldout as heldout


class HeldoutTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        data, run = root / "synthetic-data", root / "synthetic-model"
        data.mkdir()
        run.mkdir()
        (run / "best.pt").write_bytes(b"synthetic checkpoint, not a model")
        (run / "visible_attributes.onnx").write_bytes(b"synthetic ONNX, not a model")
        model_hash = heldout.sha256(run / "visible_attributes.onnx")
        android = {"thresholds": [.4, .5, .6], "preprocessing": {"fixture": True}, "model_sha256": model_hash}
        (run / "android_metadata.json").write_text(json.dumps(android), encoding="utf-8")
        lock = {"format": heldout.LOCK_FORMAT, "run": str(run),
                "checkpoint_sha256": heldout.sha256(run / "best.pt"), "onnx_sha256": model_hash,
                "android_metadata_sha256": heldout.sha256(run / "android_metadata.json"),
                "thresholds": android["thresholds"], "preprocessing": android["preprocessing"],
                "heldout_selection": {"pool": "official CelebA test only", "sample_count": 3,
                                      "seed": heldout.HELDOUT_SEED, "identity_independent": False}}
        lock["lock_id"] = heldout.digest(lock)
        lock_path = root / "synthetic-lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        names = [f"{i:06d}.jpg" for i in range(1, 9)]
        splits = [0, 0, 1, 2, 2, 2, 2, 2]
        (data / "list_eval_partition.txt").write_text("".join(f"{name} {split}\n" for name, split in zip(names, splits)), encoding="utf-8")
        columns = list(heldout.SAFE_ATTRIBUTES) + [f"fixture_{i}" for i in range(37)]
        lines = ["8", " ".join(columns)]
        for i, name in enumerate(names):
            lines.append(" ".join([name] + ["1" if (i + j) % 2 else "-1" for j in range(40)]))
        (data / "list_attr_celeba.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with zipfile.ZipFile(data / "img_align_celeba.zip", "w") as archive:
            for i, name in enumerate(names):
                stream = io.BytesIO()
                Image.new("RGB", (4, 4), (i * 20, 10, 40)).save(stream, format="JPEG")
                archive.writestr("img_align_celeba/" + name, stream.getvalue())
        with (data / "dev.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=heldout.FIELDS)
            writer.writeheader()
            for i in range(3):
                writer.writerow({"name": names[i], "split": "train" if i < 2 else "validation", "image_path": "unused-fixture-path",
                                 **{label: i % 2 for label in heldout.SAFE_ATTRIBUTES}})
        expected = {name: heldout.hashes(data / name) for name in heldout.EXPECTED}
        return data, run, lock_path, expected

    def test_missing_lock_precedes_dataset_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(heldout, "hashes", side_effect=AssertionError("dataset was read")):
                with self.assertRaises(FileNotFoundError):
                    heldout.prepare(Path(temporary) / "missing-lock", Path(temporary) / "missing-data")

    def test_tampered_lock_and_changed_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, run, lock_path, expected = self.fixture(temporary)
            with mock.patch.object(heldout, "HELDOUT_COUNT", 3):
                original = lock_path.read_text()
                lock = json.loads(original)
                lock["thresholds"][0] = .7
                lock_path.write_text(json.dumps(lock))
                with self.assertRaisesRegex(ValueError, "modified"):
                    heldout.prepare(lock_path, data)
                lock_path.write_text(original)
                (run / "best.pt").write_bytes(b"changed after freeze")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    heldout.prepare(lock_path, data)
            self.assertFalse((data / "images").exists())

    def test_wrong_seed_rejected_even_with_valid_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, run, lock_path, expected = self.fixture(temporary)
            lock = json.loads(lock_path.read_text())
            lock.pop("lock_id")
            lock["heldout_selection"]["seed"] = 42
            lock["lock_id"] = heldout.digest(lock)
            lock_path.write_text(json.dumps(lock))
            with mock.patch.object(heldout, "HELDOUT_COUNT", 3):
                with mock.patch.object(heldout, "hashes", side_effect=AssertionError("artifact/data read before seed gate")):
                    with self.assertRaisesRegex(ValueError, "preregister"):
                        heldout.prepare(lock_path, data)

    def test_extract_reuse_and_detect_modified_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, run, lock_path, expected = self.fixture(temporary)
            with mock.patch.multiple(heldout, HELDOUT_COUNT=3, OFFICIAL_COUNTS={0: 2, 1: 1, 2: 5}, EXPECTED=expected):
                result = heldout.prepare(lock_path, data)
                protocol = json.loads((data / "heldout_protocol.json").read_text())
                names = protocol["selected_names"]["test"]
                self.assertEqual(len(names), 3)
                self.assertEqual(protocol["seed"], 20261113)
                self.assertTrue(set(names).isdisjoint(["000001.jpg", "000002.jpg", "000003.jpg"]))
                with (data / "heldout.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertTrue(all(Path(row["image_path"]).is_absolute() for row in rows))
                self.assertTrue(all(row[label] in ("0", "1") for row in rows for label in heldout.SAFE_ATTRIBUTES))
                timestamps = {path: path.stat().st_mtime_ns for path in data.rglob("*") if path.is_file()}
                reused = heldout.prepare(lock_path, data)
                self.assertEqual(reused["status"], "reused_verified_same_protocol")
                self.assertEqual(result["manifest_sha256"], reused["manifest_sha256"])
                self.assertEqual(timestamps, {path: path.stat().st_mtime_ns for path in data.rglob("*") if path.is_file()})
                (data / "images/test" / names[0]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "differs from original"):
                    heldout.prepare(lock_path, data)

    def test_wrong_original_md5_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, run, lock_path, expected = self.fixture(temporary)
            (data / "list_attr_celeba.txt").write_text("wrong file")
            with mock.patch.multiple(heldout, HELDOUT_COUNT=3, OFFICIAL_COUNTS={0: 2, 1: 1, 2: 5}, EXPECTED=expected):
                with self.assertRaisesRegex(ValueError, "MD5/size"):
                    heldout.prepare(lock_path, data)
            self.assertFalse((data / "images").exists())

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "escapes"):
                heldout.safe_path(Path(temporary).resolve(), "..", "outside.jpg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
