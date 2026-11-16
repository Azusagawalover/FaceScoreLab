"""Synthetic correctness/safety tests only. None of these establish model accuracy."""
import unittest

import numpy as np

from evaluate import binary_metrics, confusion_matrix, evaluate_attributes, segmentation_metrics
from evaluation_common import paired_group_bootstrap
from manifest import build_splits, import_identities, validate_manifest


def annotation(image, group, target, attribute="smiling"):
    return {"image_id": image, "group_id": group, "split": "validation", "dataset_role": "functional_fixture",
            "label_source": "synthetic_test_fixture", "attribute": attribute, "target": target}


def sample(image, group, digest=None, split=""):
    return {"image_id": image, "group_id": group, "group_id_origin": "participant_code",
            "sha256": digest or image, "decoded_pixel_sha256": digest or image,
            "duplicate_group_id": "", "split": split}


class MetricTests(unittest.TestCase):
    def test_binary_manual_counts(self):
        result = binary_metrics([2, 1, 1, 4])
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_absent_binary_label_is_not_perfect(self):
        result = binary_metrics([0, 0, 0, 5])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])

    def test_mask_ignore_and_false_positive_absence(self):
        target = np.array([[0, 1], [255, 1]], np.uint8)
        pred = np.array([[0, 2], [99, 1]], np.uint8)
        matrix = confusion_matrix(target, pred)
        result = segmentation_metrics(matrix)
        self.assertEqual(int(matrix.sum()), 3)
        self.assertEqual(result["skin"]["iou"], .5)
        self.assertEqual(result["left_brow"]["iou"], 0.)
        self.assertIsNone(result["hat"]["iou"])
        self.assertAlmostEqual(result["skin"]["dice"], 2 / 3)

    def test_invalid_mask_class_at_valid_pixel_rejected(self):
        with self.assertRaisesRegex(ValueError, "Out-of-range"):
            confusion_matrix(np.array([[1]], np.uint8), np.array([[255]], np.uint8))

    def test_mask_shape_and_rgb_fail_instead_of_resize(self):
        with self.assertRaisesRegex(ValueError, "identical size"):
            confusion_matrix(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "2D"):
            confusion_matrix(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8))

    def test_paired_bootstrap_identity_and_cluster_extremes(self):
        clusters = np.array([[1, 0, 0, 0], [0, 0, 3, 0]])
        metrics = lambda cells: {"f1": binary_metrics(cells)["f1"]}
        result = paired_group_bootstrap(clusters, metrics, clusters, repeats=500, seed=10)
        self.assertEqual(result["intervals"]["delta_b_minus_a"]["f1"]["low"], 0.)
        self.assertEqual(result["intervals"]["delta_b_minus_a"]["f1"]["high"], 0.)
        self.assertEqual(result["intervals"]["a"]["f1"]["low"], 0.)
        self.assertEqual(result["intervals"]["a"]["f1"]["high"], 1.)
        self.assertEqual(result, paired_group_bootstrap(clusters, metrics, clusters, repeats=500, seed=10))

    def test_fixtures_require_explicit_nonaccuracy_mode(self):
        with self.assertRaisesRegex(ValueError, "not an accuracy set"):
            evaluate_attributes([annotation("a", "g", 1)], [{"image_id": "a", "attribute": "smiling", "score": .8}], {"smiling": .5})

    def test_model_outputs_cannot_declare_ground_truth(self):
        row = annotation("a", "g", 1)
        row.update(dataset_role="accuracy_set", label_source="model_prediction")
        with self.assertRaisesRegex(ValueError, "never model predictions"):
            evaluate_attributes([row], [{"image_id": "a", "attribute": "smiling", "score": .8}], {"smiling": .5})

    def test_human_review_needs_two_reviewers(self):
        row = annotation("a", "g", 1)
        row.update(dataset_role="accuracy_set", label_source="human_double_reviewed", reviewer_1="r1", reviewer_2="r1", review_status="agreed", source_reference="synthetic-review-ledger")
        with self.assertRaisesRegex(ValueError, "distinct reviewer"):
            evaluate_attributes([row], [{"image_id": "a", "attribute": "smiling", "score": .8}], {"smiling": .5})

    def test_agreed_reviewers_cannot_disagree_silently(self):
        row = annotation("a", "g", 1)
        row.update(dataset_role="accuracy_set", label_source="human_double_reviewed", reviewer_1="r1", reviewer_2="r2", review_status="agreed", source_reference="synthetic-review-ledger", reviewer_1_target=1, reviewer_2_target=0)
        with self.assertRaisesRegex(ValueError, "do not match"):
            evaluate_attributes([row], [{"image_id": "a", "attribute": "smiling", "score": .8}], {"smiling": .5})

    def test_unknown_and_unavailable_preserve_coverage(self):
        rows = [annotation("a", "g1", 1), annotation("b", "g2", 0), annotation("c", "g3", "unknown")]
        preds = [{"image_id": "a", "attribute": "smiling", "score": .8},
                 {"image_id": "b", "attribute": "smiling", "score": "", "prediction_status": "unavailable"},
                 {"image_id": "c", "attribute": "smiling", "score": .1}]
        result = evaluate_attributes(rows, preds, {"smiling": .5}, allow_fixtures=True, repeats=20)
        self.assertEqual(result["coverage"]["smiling"]["unknown_truth"], 1)
        self.assertEqual(result["coverage"]["smiling"]["a_coverage"], .5)
        self.assertEqual(result["model_a"]["smiling"]["tn"], 0)
        self.assertEqual(result["evidence_kind"], "functional_fixture_not_accuracy")

    def test_missing_prediction_is_error(self):
        with self.assertRaisesRegex(ValueError, "Missing prediction"):
            evaluate_attributes([annotation("a", "g", 1)], [], {"smiling": .5}, allow_fixtures=True)

    def test_nan_score_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_attributes([annotation("a", "g", 1)], [{"image_id": "a", "attribute": "smiling", "score": float("nan")}], {"smiling": .5}, allow_fixtures=True)

    def test_paired_predictions_use_same_cohort(self):
        rows = [annotation("a", "g1", 1), annotation("b", "g2", 0)]
        first = [{"image_id": "a", "attribute": "smiling", "score": .8}, {"image_id": "b", "attribute": "smiling", "score": .2}]
        second = [{"image_id": "a", "attribute": "smiling", "score": .8}, {"image_id": "b", "attribute": "smiling", "score": "", "prediction_status": "unavailable"}]
        result = evaluate_attributes(rows, first, {"smiling": .5}, second, allow_fixtures=True, repeats=20)
        self.assertEqual(result["model_a"]["smiling"]["negative_support"], 0)
        self.assertEqual(result["model_b"]["smiling"]["negative_support"], 0)
        self.assertEqual(result["coverage"]["smiling"]["common_evaluated"], 1)

    def test_separately_fixed_model_thresholds(self):
        rows = [annotation("a", "g1", 1)]
        predictions = [{"image_id": "a", "attribute": "smiling", "score": .4}]
        result = evaluate_attributes(rows, predictions, {"smiling": .5}, predictions, allow_fixtures=True,
                                     thresholds_b={"smiling": .3}, repeats=10)
        self.assertEqual(result["model_a"]["smiling"]["f1"], 0.)
        self.assertEqual(result["model_b"]["smiling"]["f1"], 1.)


class SplitTests(unittest.TestCase):
    def test_identity_and_duplicate_components_stay_together(self):
        rows = [sample("a", "g1", "same"), sample("b", "g2", "same"), sample("c", "g2", "other"), sample("d", "g3")]
        result, audit = build_splits(rows)
        self.assertEqual(len({row["split"] for row in result[:3]}), 1)
        self.assertEqual(audit["n_connected_groups"], 2)
        self.assertEqual(result, build_splits(list(reversed(rows)))[0][::-1])

    def test_cross_split_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "spans splits"):
            validate_manifest([sample("a", "g1", split="train"), sample("b", "g1", split="test")])

    def test_unknown_identities_fail_closed(self):
        row = sample("a", "")
        row["group_id_origin"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Missing identity"):
            build_splits([row])
        _, audit = build_splits([row], allow_image_level=True)
        self.assertFalse(audit["identity_isolation_declared"])

    def test_previous_locked_split_preserved(self):
        old = [sample("a", "g1", split="test"), sample("b", "g2", split="train")]
        updated, _ = build_splits(old + [sample("c", "g3")], previous=old, additions_train_only=True)
        self.assertEqual([row["split"] for row in updated], ["test", "train", "train"])

    def test_error_mined_locked_identity_rejected(self):
        old = [sample("a", "g1", split="test")]
        with self.assertRaisesRegex(ValueError, "Error-mined additions"):
            build_splits(old + [sample("b", "g1")], previous=old, additions_train_only=True)

    def test_changed_frozen_hash_rejected(self):
        old = [sample("a", "g1", split="test")]
        changed = [dict(old[0], sha256="changed")]
        with self.assertRaisesRegex(ValueError, "Frozen identity/hash"):
            build_splits(changed, previous=old)

    def test_pseudonyms_stable_namespace_scoped_and_no_raw_ids(self):
        images = [{"image_id": "a"}, {"image_id": "b"}]
        mapping = [{"image_id": "a", "source_identity_id": "participant123"}, {"image_id": "b", "source_identity_id": "participant123"}]
        first = import_identities(images, mapping, b"x" * 32, "study")
        second = import_identities(images, mapping, b"x" * 32, "another")
        self.assertEqual(first[0]["group_id"], first[1]["group_id"])
        self.assertNotEqual(first[0]["group_id"], second[0]["group_id"])
        self.assertNotIn("source_identity_id", first[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
