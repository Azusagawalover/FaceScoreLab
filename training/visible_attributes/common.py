"""Shared contracts for three independent, visible CelebA labels only."""
from __future__ import annotations
import csv
import hashlib
import inspect
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

LABELS = ("Smiling", "Eyeglasses", "Mouth_Slightly_Open")
DISPLAY_LABELS = ("微笑", "眼镜", "嘴巴张开")
VERSION = "visible-attributes-mnv3-v1"
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
THRESHOLD_GRID = np.round(np.arange(.05, .951, .025), 3).tolist()
UNKNOWN = {"", "na", "n/a", "?", "unknown", "null", "none", "nan"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, hours, consumed=0.):
        if not 0 < hours <= 2:
            raise ValueError("This first protocol allows at most two hours per command/run")
        self.limit = hours*3600
        self.started = time.monotonic()
        self.consumed = consumed

    def elapsed(self):
        return self.consumed + time.monotonic()-self.started

    def check(self):
        if self.elapsed() >= self.limit:
            raise BudgetExceeded("The registered wall-time budget is exhausted")


def parse_label(value, explicit_mask=None):
    """The data-agent manifest uses 0/1; zero is a KNOWN negative."""
    if explicit_mask is not None:
        mask_text = str(explicit_mask).strip().lower()
        if mask_text in UNKNOWN or mask_text in ("0", "0.0"):
            return 0., 0.
        if mask_text not in ("1", "1.0"):
            raise ValueError("Explicit label masks must be 0 or 1")
    text = str(value).strip().lower()
    if text in UNKNOWN:
        return 0., 0.
    if text not in ("0", "1", "0.0", "1.0"):
        raise ValueError("Expected normalized 0/1 label, or blank/NA/? for unknown; signed CelebA values are not accepted")
    return float(text), 1.


def load_manifest(path, mode, smoke=False):
    allowed = {"train", "validation"} if mode == "development" else {"test"}
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"name", "split", "image_path", *LABELS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain {sorted(required)}")
        for raw in reader:
            if raw["split"] not in allowed:
                raise ValueError("Split boundary violation; test rows cannot enter development preparation")
            name = raw["name"]
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Unsafe image name")
            labels, masks = zip(*(parse_label(raw[label], raw.get(label+"_mask")) for label in LABELS))
            image_path = Path(raw["image_path"])
            if not image_path.is_absolute():
                raise ValueError("image_path must be an explicit absolute path")
            if image_path.name != name:
                raise ValueError("Image path basename must match its labelled manifest name")
            rows.append({"name": name, "split": raw["split"], "image_path": str(image_path),
                         "labels": list(labels), "masks": list(masks)})
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image names in manifest")
    if smoke:
        limited, counts = [], {}
        for row in rows:
            limit = 16 if row["split"] == "train" else 8
            if counts.get(row["split"], 0) < limit:
                limited.append(row)
                counts[row["split"]] = counts.get(row["split"], 0)+1
        rows = limited
    expected = {"train": 20000, "validation": 2000}
    if mode == "development" and not smoke:
        actual = {s: sum(r["split"] == s for r in rows) for s in allowed}
        if actual != expected:
            raise ValueError(f"First development protocol requires {expected}, got {actual}")
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def face_crop_rect(points, width, height):
    # Same float32 maximum-landmark-box and clipped faceCrop v1 as the app.
    p = np.asarray(points, dtype=np.float32)
    left, top = p.min(axis=0)
    right, bottom = p.max(axis=0)
    size = np.float32(max(right-left, bottom-top)*np.float32(1.45))
    cx = np.float32((left+right)/np.float32(2))
    cy = np.float32((top+bottom)/np.float32(2)-size*np.float32(.06))
    return [max(0, math.floor(float(cx-size/2))), max(0, math.floor(float(cy-size/2))),
            min(width, math.ceil(float(cx+size/2))), min(height, math.ceil(float(cy+size/2)))]


def preprocessing_contract(face_task_hash):
    return {"version": "mediapipe-largest-face-facecrop-v1-rgb224", "face_task_sha256": face_task_hash,
            "num_faces": 3, "min_face_detection_confidence": .5, "min_face_presence_confidence": .5,
            "min_tracking_confidence": .5, "face_selection": "largest landmark bounding-box area; first on ties",
            "crop": "faceCrop v1: box max edge *1.45, center_y minus .06*size, clipped to image bounds",
            "crop_implementation_sha256": hashlib.sha256(inspect.getsource(face_crop_rect).encode()).hexdigest(),
            "resize": "224x224 bilinear", "color": "RGB", "dtype": "float32", "layout": "NCHW contiguous",
            "pixel_scale": "divide by 255", "mean": MEAN.tolist(), "std": STD.tolist()}


def tensor_input(crop_path):
    with Image.open(crop_path) as opened:
        image = opened.convert("RGB")
    if image.size != (224, 224):
        raise ValueError("Prepared crop must be 224x224")
    values = np.asarray(image, np.float32)/255
    return ((values-MEAN)/STD).transpose(2, 0, 1).copy()


class AttributeNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared_head = nn.Sequential(nn.LayerNorm(576), nn.Linear(576, 128), nn.Hardswish(), nn.Dropout(.2))
        self.output_heads = nn.ModuleList([nn.Linear(128, 1) for _ in LABELS])
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def classify(self, vectors):
        shared = self.shared_head(vectors)
        return torch.cat([head(shared) for head in self.output_heads], dim=1)

    def forward(self, x):
        return self.classify(self.pool(self.features(x)).flatten(1))

    def head_parameters(self):
        return list(self.shared_head.parameters())+list(self.output_heads.parameters())

    def train_heads(self):
        self.shared_head.train()
        self.output_heads.train()
        self.features.eval()


def weights_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(canonical({"name": name, "shape": list(a.shape), "dtype": str(a.dtype)}))
        h.update(a.tobytes())
    return h.hexdigest()


def masked_bce(logits, labels, masks, pos_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    valid = masks.sum()
    return (loss*masks).sum()/valid.clamp_min(1), valid


def sigmoid(logits):
    values = np.asarray(logits, np.float64)
    return 1/(1+np.exp(-np.clip(values, -80, 80)))


def class_counts(labels, masks):
    y, m = np.asarray(labels), np.asarray(masks, bool)
    return [{"label": label, "known": int(m[:, i].sum()), "positive": int(((y[:, i] == 1)&m[:, i]).sum()),
             "negative": int(((y[:, i] == 0)&m[:, i]).sum()), "unknown": int((~m[:, i]).sum())} for i, label in enumerate(LABELS)]


def binary_metrics(y, prediction):
    y, prediction = np.asarray(y, bool), np.asarray(prediction, bool)
    tp, fp = int((y&prediction).sum()), int((~y&prediction).sum())
    fn, tn = int((y&~prediction).sum()), int((~y&~prediction).sum())
    n = len(y)
    if not n:
        return {"known_available": 0, "positive_support": 0, "negative_support": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "precision": None, "recall": None, "f1": None, "balanced_accuracy": None}
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return {"known_available": n, "positive_support": tp+fn, "negative_support": tn+fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "balanced_accuracy": .5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None}


def choose_thresholds(labels, masks, scores):
    y, m, scores = np.asarray(labels), np.asarray(masks, bool), np.asarray(scores)
    if y.shape != m.shape or scores.shape != y.shape or y.ndim != 2 or y.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError("Invalid validation label/mask/score arrays")
    thresholds = []
    for i, label in enumerate(LABELS):
        known = m[:, i]
        positives = int(y[known, i].sum())
        if positives == 0 or positives == int(known.sum()):
            raise ValueError(f"Validation must contain known positives and negatives for {label}")
        choices = [(binary_metrics(y[known, i], scores[known, i] >= t)["f1"], -abs(t-.5), t) for t in THRESHOLD_GRID]
        thresholds.append(max(choices)[2])
    return thresholds


def evaluate_scores(labels, masks, scores, thresholds, available=None):
    y, m, scores = np.asarray(labels), np.asarray(masks, bool), np.asarray(scores)
    present = np.ones(len(y), bool) if available is None else np.asarray(available, bool)
    if y.shape != m.shape or scores.shape != y.shape or y.ndim != 2 or y.shape[1] != 3 or present.shape != (len(y),):
        raise ValueError("Invalid label/mask/score shapes")
    if len(thresholds) != 3 or not all(0 < value < 1 for value in thresholds) or not np.isfinite(scores[present]).all():
        raise ValueError("Invalid frozen thresholds or available scores")
    per_label = []
    for i, label in enumerate(LABELS):
        chosen = m[:, i]&present
        metrics = binary_metrics(y[chosen, i], scores[chosen, i] >= thresholds[i])
        metrics.update(label=label, threshold=float(thresholds[i]), unknown_labels=int((~m[:, i]).sum()),
                       known_total=int(m[:, i].sum()), coverage_of_known=float(chosen.sum()/m[:, i].sum()) if m[:, i].sum() else None)
        per_label.append(metrics)
    f1s = [item["f1"] for item in per_label if item["f1"] is not None]
    return {"macro_f1": float(np.mean(f1s)) if f1s else None, "macro_f1_label_count": len(f1s), "per_label": per_label,
            "coverage": {"images_total": len(y), "images_with_output": int(present.sum()),
                         "face_detection_output_rate": float(present.mean()) if len(present) else None},
            "score_type": "uncalibrated independent sigmoid scores; not calibrated probabilities"}


def write_predictions(path, rows, scores, thresholds):
    stream = io.StringIO(newline="")
    columns = ["name", "split", "detected"]
    for label in LABELS:
        columns += [label+"_label", label+"_known", label+"_score", label+"_prediction"]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row, values in zip(rows, scores):
        item = {"name": row["name"], "split": row["split"], "detected": int(row["detected"])}
        for i, label in enumerate(LABELS):
            item[label+"_label"] = int(row["labels"][i]) if row["masks"][i] else ""
            item[label+"_known"] = int(row["masks"][i])
            item[label+"_score"] = float(values[i]) if row["detected"] else ""
            item[label+"_prediction"] = int(values[i] >= thresholds[i]) if row["detected"] else ""
        writer.writerow(item)
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
