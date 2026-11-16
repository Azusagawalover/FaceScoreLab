"""Isolated, budgeted SCUT development, export, and explicitly locked evaluation.

The training entry requires a development-only manifest. It never opens the
original combined manifest or test image/label files. All artifacts are written
outside the existing Android assets and historical results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import inspect
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


FORMAT_VERSION = "facescore-optimization-v1"
PREPROCESS_VERSION = "scut-rgb224-imagenet-v2"
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
KNOWN_CONFIGS = ("R0", "R1", "R2", "R3", "R4", "R5")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest_json(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Only load checkpoints produced by this local runner. RNG/optimizer state
    # requires the explicit non-weights-only mode on PyTorch 2.6.
    return torch.load(path, map_location="cpu", weights_only=False)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, hours, consumed=0.):
        self.started = time.monotonic()
        self.consumed = float(consumed)
        self.limit = float(hours) * 3600
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("max-hours must be positive and finite")

    def elapsed(self):
        return self.consumed + time.monotonic() - self.started

    def check(self):
        if self.elapsed() >= self.limit:
            raise BudgetExceeded("The cumulative wall-time budget for this run is exhausted")


def prepare_manifest(source, output):
    """A separate data preparation entry; discarded test scores are not parsed."""
    source, output = Path(source), Path(output)
    if output.resolve() == source.resolve():
        raise ValueError("Development manifest must be a new file")
    accepted = []
    excluded = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["split"] not in ("train", "validation"):
                excluded += 1
                continue
            name = row["name"]
            if Path(name).name != name or "\\" in name or "/" in name:
                raise ValueError("Unsafe image name")
            accepted.append({"name": name, "score": float(row["score"]), "split": row["split"],
                             "pixel_sha256": row.get("pixel_sha256", "")})
    if not accepted:
        raise ValueError("No development samples")
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=["name", "score", "split", "pixel_sha256"])
    writer.writeheader()
    writer.writerows(accepted)
    raw = content.getvalue().encode("utf-8")
    if output.exists() and output.read_bytes() != raw:
        raise FileExistsError("Refusing to replace a different development manifest")
    atomic_bytes(output, raw)
    record = {"version": FORMAT_VERSION, "created_utc": utc_now(),
              "source_manifest_sha256": file_sha256(source),
              "development_manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "counts": {s: sum(r["split"] == s for r in accepted) for s in ("train", "validation")},
              "discarded_rows_without_parsing_targets": excluded,
              "note": "Training reads only the resulting development-only CSV."}
    atomic_json(output.with_suffix(".provenance.json"), record)
    return record


def load_development_rows(path, smoke=False):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            # Reject a combined manifest before converting any prohibited target.
            if row["split"] not in ("train", "validation"):
                raise ValueError("Training requires a development-only manifest; run prepare-manifest first")
            name = row["name"]
            if Path(name).name != name or "\\" in name or "/" in name:
                raise ValueError("Unsafe image name")
            target = float(row["score"])
            if not math.isfinite(target) or not 1 <= target <= 5:
                raise ValueError("SCUT targets must be finite values on the 1-5 scale")
            rows.append({"name": name, "score": target, "split": row["split"],
                         "pixel_sha256": row.get("pixel_sha256", "")})
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Duplicate names in development manifest")
    if smoke:
        counts = {"train": 0, "validation": 0}
        limited = []
        for row in rows:
            maximum = 16 if row["split"] == "train" else 8
            if counts[row["split"]] < maximum:
                limited.append(row)
                counts[row["split"]] += 1
        rows = limited
    if not all(any(r["split"] == s for r in rows) for s in ("train", "validation")):
        raise ValueError("Both train and validation are required")
    return rows


def make_config(plan_path, config_id, seed, smoke=False):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    registered = {r["id"] for r in plan["experiments"]}
    if config_id not in KNOWN_CONFIGS or config_id not in registered:
        raise ValueError("Unknown preregistered configuration")
    p = plan["first_training_batch"]
    c = {"format": FORMAT_VERSION, "id": config_id, "seed": int(seed),
         "plan_sha256": file_sha256(plan_path), "architecture": "MobileNetV3Small-LayerNorm128",
         "head_epochs": 100, "head_patience": 16, "batch_size": p["batch_size"],
         "dropout": p["dropout"], "weight_decay": p["weight_decay"],
         "head_lr": p["head_stage_lr"], "tail_head_lr": p["tail_stage_head_lr"],
         "tail_backbone_lr": p["tail_stage_backbone_lr"],
         "tail_epochs": 16 if config_id == "R0" else p["tail_epoch_limit_except_original_baseline"],
         "tail_patience": 6 if config_id == "R0" else p["early_stopping_patience_except_original_baseline"],
         "min_improvement": p["original_protocol_min_improvement"] if config_id == "R0" else p["new_protocol_min_improvement"],
         "head_min_improvement": p["original_protocol_min_improvement"],
         "tail_min_improvement": p["original_protocol_min_improvement"] if config_id == "R0" else p["new_protocol_min_improvement"],
         "scheduler": config_id in ("R2", "R3", "R4", "R5"),
         "scheduler_factor": p["scheduler_factor"], "scheduler_patience": p["scheduler_patience"],
         "scheduler_min_head_lr": p["scheduler_min_head_lr"],
         "scheduler_min_backbone_lr": p["scheduler_min_backbone_lr"],
         "loss": "huber" if config_id == "R3" else "mse", "huber_delta": .3,
         "preprocess": "square_padding_from_original_v1" if config_id == "R5" else "legacy_facecrop_v1",
         "augmentation": "two_fixed_mild_views_v1" if config_id == "R4" else "none",
         "augmentation_stage": "tail_only", "frozen_prefix_modules": 9,
         "smoke": bool(smoke), "single_run_cap_hours": p["single_run_compute_wall_hours_cap"]}
    if smoke:
        c.update(head_epochs=2, tail_epochs=2)
    return c


def minimum_improvement(config, stage_index):
    key = "head_min_improvement" if stage_index == 0 else "tail_min_improvement"
    return config.get(key, config["min_improvement"])


def legacy_r0_equivalent(old_config, new_config):
    """The first active R0 run predates explicit head/tail threshold fields."""
    if old_config.get("id") != "R0" or new_config.get("id") != "R0":
        return False
    expanded = copy.deepcopy(old_config)
    expanded.setdefault("head_min_improvement", old_config["min_improvement"])
    expanded.setdefault("tail_min_improvement", old_config["min_improvement"])
    return expanded == new_config


class BeautyNet(nn.Module):
    def __init__(self, features, mean, dropout=.2):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.LayerNorm(576), nn.Linear(576, 128), nn.Hardswish(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("target_mean", torch.tensor(float(mean), dtype=torch.float32))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1)) + self.target_mean


def initialize_model(mean, config, pretrained=True):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    # Preserve the original factory/head initialization order and RNG consumption.
    backbone = mobilenet_v3_small(weights=weights)
    return BeautyNet(backbone.features, mean, config["dropout"])


def state_sha256(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(canonical({"name": name, "shape": list(a.shape), "dtype": str(a.dtype)}))
        h.update(a.tobytes())
    return h.hexdigest()


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    boundaries = np.concatenate(([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1, [len(values)]))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + end - 1)/2 + 1
    return ranks


def metrics(target, prediction):
    y, p = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid metric inputs")
    def correlation(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-10 and b.std() > 1e-10 else None
    return {"n": len(y), "mae": float(np.abs(y-p).mean()),
            "rmse": float(np.sqrt(np.mean((y-p)**2))), "pearson": correlation(y, p),
            "spearman": correlation(average_ranks(y), average_ranks(p))}


def image_sources(data_dir, row, config):
    data_dir = Path(data_dir)
    if config["preprocess"] == "legacy_facecrop_v1":
        return [data_dir / "crops" / (row["name"] + ".png")]
    return [data_dir / "images" / row["name"], data_dir / "landmarks" / (row["name"] + ".npy")]


def square_crop(image, points):
    """Preserve the desired face box before image-boundary clipping.

    The larger integer edge is used for both sides; any area outside the source
    is RGB128 padding. The continuous center uses the existing float32 faceCrop
    formula. Android must implement this exact contract before R5 deployment.
    """
    p = np.asarray(points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Invalid landmark coordinates")
    left, top = p.min(axis=0)
    right, bottom = p.max(axis=0)
    size = np.float32(max(right-left, bottom-top) * np.float32(1.45))
    cx = np.float32((left+right) / np.float32(2))
    cy = np.float32((top+bottom) / np.float32(2) - size*np.float32(.06))
    x0, y0 = math.floor(float(cx-size/2)), math.floor(float(cy-size/2))
    x1, y1 = math.ceil(float(cx+size/2)), math.ceil(float(cy+size/2))
    side = max(x1-x0, y1-y0)
    if side <= 0 or side > max(image.size)*10:
        raise ValueError("Invalid face rectangle")
    result = Image.new("RGB", (side, side), (128, 128, 128))
    source = (max(0, x0), max(0, y0), min(image.width, x0+side), min(image.height, y0+side))
    if source[2] <= source[0] or source[3] <= source[1]:
        raise ValueError("Face rectangle does not overlap image")
    result.paste(image.crop(source), (source[0]-x0, source[1]-y0))
    return result.resize((224, 224), Image.Resampling.BILINEAR)


def load_image(data_dir, row, config, view=0):
    sources = image_sources(data_dir, row, config)
    with Image.open(sources[0]) as opened:
        image = opened.convert("RGB")
    if config["preprocess"] == "square_padding_from_original_v1":
        image = square_crop(image, np.load(sources[1], allow_pickle=False))
    elif image.size != (224, 224):
        raise ValueError("Legacy crop must already be 224x224")
    if view:
        if row["split"] != "train" or config["augmentation"] == "none":
            raise ValueError("Augmentation is restricted to training samples")
        seed_raw = hashlib.sha256(f'{config["seed"]}:{row["name"]}:{view}:mild-v1'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed_raw[:8], "little"))
        if rng.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.rotate(float(rng.uniform(-7, 7)), resample=Image.Resampling.BILINEAR,
                             fillcolor=(128, 128, 128))
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(.9, 1.1)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(.9, 1.1)))
    return image


def input_array(data_dir, row, config, view=0):
    array = np.asarray(load_image(data_dir, row, config, view), dtype=np.float32) / 255
    return ((array-MEAN)/STD).transpose(2, 0, 1).copy()


def cache_spec(data_dir, rows, manifest_path, config, early, budget):
    inputs = []
    for index, row in enumerate(rows):
        budget.check()
        inputs.append({"name": row["name"], "split": row["split"],
                       "source_sha256": [file_sha256(p) for p in image_sources(data_dir, row, config)]})
        if index and index % 1000 == 0:
            print(json.dumps({"event": "cache_input_audit", "count": index}), flush=True)
    return {"format": FORMAT_VERSION, "preprocess_version": PREPROCESS_VERSION,
            "preprocessing_implementation_sha256": hashlib.sha256((inspect.getsource(square_crop) + inspect.getsource(load_image) + inspect.getsource(input_array)).encode("utf-8")).hexdigest(),
            "preprocess": config["preprocess"], "augmentation": config["augmentation"],
            "augmentation_seed": config["seed"] if config["augmentation"] != "none" else None,
            "dtype": "float32", "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
            "manifest_sha256": file_sha256(manifest_path), "rows_sha256": digest_json(rows),
            "inputs": inputs, "early_weights_sha256": state_sha256(early.state_dict()),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "pillow": Image.__version__, "numpy": np.__version__, "cache_batch_size": 32}


def ensure_cache(data_dir, rows, manifest_path, config, early, cache_dir, budget):
    spec = cache_spec(data_dir, rows, manifest_path, config, early, budget)
    key = digest_json(spec)
    directory = Path(cache_dir) / key
    directory.mkdir(parents=True, exist_ok=True)
    path, meta_path = directory / "early_maps.npy", directory / "cache_manifest.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key and meta.get("npy_sha256") == file_sha256(path):
            bank = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(bank.shape) == meta["shape"] and bank.dtype == np.float32:
                return bank, {"key": key, "path": str(path), "reused": True, "npy_sha256": meta["npy_sha256"]}
        raise ValueError("Existing content-addressed cache is corrupt; preserve it and investigate")
    temporary = directory / "early_maps.partial.npy"
    nviews = 3 if config["augmentation"] != "none" else 1
    bank = None
    for view in range(nviews):
        chosen = [i for i, row in enumerate(rows) if view == 0 or row["split"] == "train"]
        if view and bank is not None:
            # Validation is never augmented; unused slots are identical copies.
            bank[view] = bank[0]
        for start in range(0, len(chosen), 32):
            budget.check()
            batch = chosen[start:start+32]
            x = np.stack([input_array(data_dir, rows[i], config, view) for i in batch])
            with torch.inference_mode():
                output = early(torch.from_numpy(x)).numpy()
            if bank is None:
                bank = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                                shape=(nviews, len(rows), *output.shape[1:]))
            bank[view, batch] = output
            if start % 320 == 0:
                print(json.dumps({"event": "cache_build", "view": view, "samples": start+len(batch),
                                  "total": len(chosen), "wall_seconds": budget.elapsed()}), flush=True)
    bank.flush()
    shape = list(bank.shape)
    del bank
    temporary.replace(path)
    checksum = file_sha256(path)
    atomic_json(meta_path, {"key": key, "spec": spec, "shape": shape, "npy_sha256": checksum,
                           "created_utc": utc_now(), "legacy_unverified_cache_imported": False})
    return np.load(path, mmap_mode="r", allow_pickle=False), {
        "key": key, "path": str(path), "reused": False, "npy_sha256": checksum}


def capture_rng(augmentation_rng):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "augmentation": copy.deepcopy(augmentation_rng.bit_generator.state)}


def restore_rng(state, augmentation_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    augmentation_rng.bit_generator.state = state["augmentation"]


def write_predictions(path, rows, indices, predictions):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["name", "split", "target_1_5", "prediction_raw", "prediction_clipped", "absolute_error"])
    for i, prediction in zip(indices, predictions):
        row = rows[int(i)]
        writer.writerow([row["name"], row["split"], row["score"], float(prediction),
                         float(np.clip(prediction, 1, 5)), abs(row["score"]-float(prediction))])
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


def setup_threads(count):
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Allows isolated smoke invocations in the same Python process.
    torch.use_deterministic_algorithms(True)


def train(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = make_config(args.plan, args.config, args.seed, args.smoke)
    if args.max_hours > config["single_run_cap_hours"]:
        raise ValueError("Requested run budget exceeds the preregistered single-run cap")
    manifest = Path(args.dev_manifest).resolve()
    rows = load_development_rows(manifest, args.smoke)
    identity = {"config": config, "development_manifest_sha256": file_sha256(manifest),
                "selected_rows_sha256": digest_json(rows)}
    fingerprint = digest_json(identity)
    last_path = output / "last.pt"
    saved = None
    if last_path.exists():
        if not args.resume:
            raise FileExistsError("Run has a checkpoint; use --resume or a new output directory")
        saved = load_checkpoint(last_path)
        if saved["fingerprint"] != fingerprint:
            old_identity = {**identity, "config": saved["config"]}
            if legacy_r0_equivalent(saved["config"], config) and digest_json(old_identity) == saved["fingerprint"]:
                config = saved["config"]
                identity = old_identity
                fingerprint = saved["fingerprint"]
            else:
                raise ValueError("Resume configuration or development data does not match checkpoint")
        if saved["status"] == "completed":
            return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    elif args.resume and not (output / "metrics.json").exists():
        raise FileNotFoundError("No completed boundary or preparation status to resume")
    elif (output / "run.json").exists():
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Output directory belongs to a different run")
    consumed = saved["wall_seconds"] if saved else 0.
    if (output / "metrics.json").exists():
        prior_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if prior_metrics["fingerprint"] != fingerprint:
            raise ValueError("Output metrics belong to another run")
        if not args.resume:
            raise FileExistsError("Existing run accounting requires --resume")
        consumed = max(consumed, prior_metrics["wall_seconds"])
    budget = Budget(args.max_hours, consumed)
    setup_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    augmentation_rng = np.random.default_rng(args.seed + 120613)
    indices = {s: np.array([i for i, row in enumerate(rows) if row["split"] == s], dtype=np.int64)
               for s in ("train", "validation")}
    train_idx, val_idx = indices["train"], indices["validation"]
    y = torch.tensor([row["score"] for row in rows], dtype=torch.float32).reshape(-1, 1)
    mean = y[train_idx].mean().item()
    model = initialize_model(mean, config)
    early = nn.Sequential(*list(model.features.children())[:9]).eval()
    tail = nn.Sequential(*list(model.features.children())[9:])
    for parameter in early.parameters():
        parameter.requires_grad = False
    initial_tail = copy.deepcopy(tail.state_dict())
    run = {"format": FORMAT_VERSION, "created_utc": utc_now(), "fingerprint": fingerprint,
           **identity, "data_dir": str(Path(args.data_dir).resolve()), "development_manifest": str(manifest),
           "source_code_sha256": file_sha256(__file__), "counts": {s: len(v) for s, v in indices.items()},
           "seed": args.seed, "threads": args.threads, "max_hours": args.max_hours,
           "test_targets_opened": False, "pretrained_weights": str(MobileNet_V3_Small_Weights.IMAGENET1K_V1.url),
           "pretrained_prefix_sha256": state_sha256(early.state_dict()),
           "effective_min_improvement": {"head": minimum_improvement(config, 0), "tail": minimum_improvement(config, 1)},
           "r0_alignment_note": "Original sample order and RNG initialization retained. Development-only forward batches can introduce floating point differences from historical all-sample caches."}
    if (output / "run.json").exists():
        old_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        run["created_utc"] = old_run["created_utc"]
        run["resume_events"] = old_run.get("resume_events", []) + [{"utc": utc_now(), "source_code_sha256": run["source_code_sha256"]}]
    atomic_json(output / "run.json", run)
    preparation_started = time.monotonic()
    optimizer = scheduler = None
    state = {"stage_index": 0, "epoch": 1, "initialized": False, "stage_complete": False,
             "cursor": 0, "order": None, "views": None, "loss_sum": 0., "sample_count": 0,
             "stale": 0, "best_mae": None, "best_state": None, "best_stage": None, "best_epoch": None,
             "logs": [], "total_batches": 0, "total_epochs": 0}
    cache_info = None
    session_batches = session_epochs = 0
    status = "preparing"

    def save_last(current_status):
        atomic_torch(last_path, {"format": FORMAT_VERSION, "fingerprint": fingerprint,
                     "config": config, "model_state": model.state_dict(), "state": state,
                     "optimizer": optimizer.state_dict() if optimizer else None,
                     "scheduler": scheduler.state_dict() if scheduler else None,
                     "rng": capture_rng(augmentation_rng), "wall_seconds": budget.elapsed(),
                     "cache_key": cache_info["key"] if cache_info else None, "status": current_status,
                     "source_code_sha256": run["source_code_sha256"]})

    def publish(current_status):
        result = {"format": FORMAT_VERSION, "status": current_status, "fingerprint": fingerprint,
                  "config": config, "counts": run["counts"], "best_stage": state["best_stage"],
                  "best_epoch": state["best_epoch"], "validation_mae": state["best_mae"],
                  "selection_metric": "validation_raw_mae_1_to_5", "test_evaluated": False,
                  "wall_seconds": budget.elapsed(), "timings": run.get("timings", {}),
                  "epochs_completed": state["total_epochs"], "batches_completed": state["total_batches"],
                  "cache": cache_info, "smoke_not_for_release": config["smoke"]}
        if state["best_state"] is not None:
            result["best_checkpoint_sha256"] = file_sha256(output / "best.pt")
        atomic_json(output / "training_log.json", state["logs"])
        atomic_json(output / "metrics.json", result)
        atomic_json(output / "run.json", run)
        return result

    def make_optimizer(stage_index):
        if stage_index == 0:
            opt = torch.optim.AdamW(model.head.parameters(), lr=config["head_lr"], weight_decay=config["weight_decay"])
        else:
            opt = torch.optim.AdamW([
                {"params": tail.parameters(), "lr": config["tail_backbone_lr"]},
                {"params": model.head.parameters(), "lr": config["tail_head_lr"]}],
                lr=config["tail_head_lr"], weight_decay=config["weight_decay"])
        sched = None
        if stage_index == 1 and config["scheduler"]:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=config["scheduler_factor"],
                       patience=config["scheduler_patience"], threshold=minimum_improvement(config, 1), threshold_mode="abs",
                       min_lr=[config["scheduler_min_backbone_lr"], config["scheduler_min_head_lr"]])
        return opt, sched

    try:
        bank, cache_info = ensure_cache(args.data_dir, rows, manifest, config, early, args.cache_dir, budget)
        maps = torch.from_numpy(np.asarray(bank).copy())
        bank._mmap.close()
        del bank
        tail.eval()
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(rows), 64):
                budget.check()
                vectors.append(model.pool(tail(maps[0, start:start+64])).flatten(1))
        vectors = torch.cat(vectors)
        run["timings"] = {"preparation_this_process_seconds": time.monotonic()-preparation_started,
                          "previous_process_seconds": consumed}
        if saved:
            if saved["cache_key"] != cache_info["key"]:
                raise ValueError("Resume inputs or prefix weights changed; cache key differs")
            model.load_state_dict(saved["model_state"])
            state = saved["state"]
            if state["initialized"] and state["stage_index"] < 2:
                optimizer, scheduler = make_optimizer(state["stage_index"])
                optimizer.load_state_dict(saved["optimizer"])
                if scheduler:
                    scheduler.load_state_dict(saved["scheduler"])
            restore_rng(saved["rng"], augmentation_rng)
        loss_fn = nn.HuberLoss(delta=config["huber_delta"]) if config["loss"] == "huber" else nn.MSELoss()

        def evaluate(stage_index):
            model.eval()
            parts = []
            with torch.inference_mode():
                for start in range(0, len(val_idx), 128):
                    batch = val_idx[start:start+128]
                    feat = vectors[batch] if stage_index == 0 else model.pool(tail(maps[0, batch])).flatten(1)
                    parts.append(model.head(feat) + model.target_mean)
            return torch.cat(parts).numpy().ravel()

        status = "running"
        while state["stage_index"] < 2:
            stage_index = state["stage_index"]
            stage = "head" if stage_index == 0 else "late_finetune"
            if state["stage_complete"]:
                state.update(stage_index=stage_index+1, epoch=1, initialized=False, stage_complete=False,
                             cursor=0, order=None, views=None, loss_sum=0., sample_count=0, stale=0)
                optimizer = scheduler = None
                continue
            if not state["initialized"]:
                if stage_index == 1:
                    model.load_state_dict(state["best_state"])
                optimizer, scheduler = make_optimizer(stage_index)
                state["initialized"] = True
            limit = config["head_epochs"] if stage_index == 0 else config["tail_epochs"]
            patience = config["head_patience"] if stage_index == 0 else config["tail_patience"]
            if state["order"] is None:
                state["order"] = train_idx[np.random.permutation(len(train_idx))]
                state["views"] = augmentation_rng.integers(0, 3, len(train_idx)) if stage_index == 1 and config["augmentation"] != "none" else np.zeros(len(train_idx), dtype=np.int64)
                state.update(cursor=0, loss_sum=0., sample_count=0)
            epoch_started = time.monotonic()
            while state["cursor"] < len(state["order"]):
                budget.check()
                if args.stop_after_batches and session_batches >= args.stop_after_batches:
                    status = "paused_for_pilot"
                    save_last(status)
                    return publish(status)
                start = state["cursor"]
                batch = state["order"][start:start+config["batch_size"]]
                views = state["views"][start:start+len(batch)]
                model.head.train()
                tail.eval()
                optimizer.zero_grad(set_to_none=True)
                feat = vectors[batch] if stage_index == 0 else model.pool(tail(maps[views, batch])).flatten(1)
                prediction = model.head(feat) + model.target_mean
                loss = loss_fn(prediction, y[batch])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                parameters = list(model.head.parameters()) + (list(tail.parameters()) if stage_index == 1 else [])
                torch.nn.utils.clip_grad_norm_(parameters, 5)
                optimizer.step()
                state["loss_sum"] += loss.item()*len(batch)
                state["sample_count"] += len(batch)
                state["cursor"] += len(batch)
                state["total_batches"] += 1
                session_batches += 1
            prediction = evaluate(stage_index)
            valid = metrics(y[val_idx].numpy().ravel(), prediction)
            before_lr = [group["lr"] for group in optimizer.param_groups]
            improved = state["best_mae"] is None or valid["mae"] < state["best_mae"]-minimum_improvement(config, stage_index)
            if improved:
                state.update(best_mae=valid["mae"], best_state=copy.deepcopy(model.state_dict()),
                             best_stage=stage, best_epoch=state["epoch"], stale=0)
                best = {"format": FORMAT_VERSION, "fingerprint": fingerprint, "config": config,
                        "model_state": state["best_state"], "target_mean": mean, "best_stage": stage,
                        "best_epoch": state["epoch"], "validation": valid,
                        "development_manifest_sha256": identity["development_manifest_sha256"],
                        "cache_key": cache_info["key"], "test_evaluated": False}
                atomic_torch(output / "best.pt", best)
                write_predictions(output / "validation_predictions.csv", rows, val_idx, prediction)
            else:
                state["stale"] += 1
            if scheduler:
                scheduler.step(valid["mae"])
            log = {"stage": stage, "epoch": state["epoch"], "loss_type": config["loss"],
                   "train_loss": state["loss_sum"]/state["sample_count"], "validation": valid,
                   "learning_rates_before_scheduler": before_lr,
                   "learning_rates_next": [group["lr"] for group in optimizer.param_groups],
                   "improved": improved, "stale_epochs": state["stale"],
                   "epoch_this_process_seconds": time.monotonic()-epoch_started,
                   "cumulative_wall_seconds": budget.elapsed()}
            state["logs"].append(log)
            print(json.dumps(log), flush=True)
            state["total_epochs"] += 1
            session_epochs += 1
            state["stage_complete"] = state["stale"] >= patience or state["epoch"] >= limit
            state.update(epoch=state["epoch"]+1, order=None, views=None, cursor=0, loss_sum=0., sample_count=0)
            save_last("running")
            publish("running")
            if args.stop_after_epochs and session_epochs >= args.stop_after_epochs:
                status = "paused_for_pilot"
                save_last(status)
                return publish(status)
        model.load_state_dict(state["best_state"])
        model.eval()
        run["late_block_weight_change_l2"] = float(np.sqrt(sum(float(torch.sum((tail.state_dict()[key]-value).float()**2)) for key, value in initial_tail.items())))
        status = "completed"
        save_last(status)
        return publish(status)
    except BudgetExceeded:
        status = "budget_exhausted"
        if cache_info is not None:
            save_last(status)
        return publish(status)
    except KeyboardInterrupt:
        # The last completed epoch/budget-boundary checkpoint is intact. Do not
        # save potentially half-executed optimizer work from an arbitrary signal.
        atomic_json(output / "interruption.json", {"utc": utc_now(), "wall_seconds": budget.elapsed(),
                    "note": "Resume last.pt, which is an atomic completed boundary; uncommitted work is discarded."})
        raise


def model_from_best(best):
    model = initialize_model(best["target_mean"], best["config"], pretrained=False)
    model.load_state_dict(best["model_state"])
    return model.eval()


def export_model(args):
    import onnx
    import onnxruntime as ort
    run_dir = Path(args.run).resolve()
    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("Export requires a completed run")
    best_path = run_dir / "best.pt"
    best = load_checkpoint(best_path)
    rows = load_development_rows(args.dev_manifest, best["config"]["smoke"])
    if file_sha256(args.dev_manifest) != best["development_manifest_sha256"]:
        raise ValueError("Export development manifest differs from selected checkpoint")
    setup_threads(args.threads)
    model = model_from_best(best)
    output = run_dir / "beauty_mobilenetv3_small.onnx"
    temporary = output.with_name(output.stem + ".tmp.onnx")
    sample = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, sample, str(temporary), input_names=["input"], output_names=["score"],
                      opset_version=17, dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(temporary))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
    parity = []
    for row in [r for r in rows if r["split"] == "validation"][:20]:
        x = input_array(args.data_dir, row, best["config"])[None]
        with torch.inference_mode():
            native = float(model(torch.from_numpy(x)).item())
        converted = float(session.run(None, {"input": x})[0].item())
        difference = abs(native-converted)
        if difference >= 1e-4:
            raise AssertionError("ONNX validation parity failed")
        parity.append({"name": row["name"], "pytorch": native, "onnx": converted, "absolute_difference": difference})
    del session
    temporary.replace(output)
    result = {"format": FORMAT_VERSION, "created_utc": utc_now(), "checkpoint_sha256": file_sha256(best_path),
              "onnx_sha256": file_sha256(output), "onnx_bytes": output.stat().st_size, "parity": parity,
              "maximum_absolute_difference": max(r["absolute_difference"] for r in parity),
              "parity_sample_split": "validation", "test_targets_opened": False,
              "preprocessing": best["config"]["preprocess"], "smoke_not_for_release": best["config"]["smoke"]}
    atomic_json(run_dir / "export_metadata.json", result)
    return result


def create_lock(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("A lock is immutable; use a new review artifact")
    entries = []
    for path in args.runs:
        directory = Path(path).resolve()
        summary = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        exported = json.loads((directory / "export_metadata.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed" or summary["smoke_not_for_release"] or exported["smoke_not_for_release"]:
            raise ValueError("Only completed non-smoke runs can be locked for final evaluation")
        checkpoint_hash = file_sha256(directory / "best.pt")
        onnx_hash = file_sha256(directory / "beauty_mobilenetv3_small.onnx")
        if checkpoint_hash != exported["checkpoint_sha256"] or onnx_hash != exported["onnx_sha256"]:
            raise ValueError("Export no longer matches selected checkpoint")
        entries.append({"run": str(directory), "fingerprint": summary["fingerprint"],
                        "checkpoint_sha256": checkpoint_hash, "onnx_sha256": onnx_hash,
                        "validation_mae": summary["validation_mae"], "config": summary["config"]})
    lock = {"format": FORMAT_VERSION, "created_utc": utc_now(), "selection_rationale": args.rationale,
            "selection_basis": "development validation only", "test_status": "previously_observed_official_fold1_historical_benchmark",
            "combined_manifest_sha256": file_sha256(args.test_manifest), "entries": entries}
    lock["lock_id"] = digest_json(lock)
    atomic_json(output, lock)
    return lock


def validate_lock(lock_path, run_dir, manifest_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    expected = lock.pop("lock_id")
    if digest_json(lock) != expected:
        raise ValueError("Candidate lock was modified")
    lock["lock_id"] = expected
    run_dir = Path(run_dir).resolve()
    entry = next((x for x in lock["entries"] if Path(x["run"]).resolve() == run_dir), None)
    if entry is None:
        raise ValueError("Run is not in the frozen candidate lock")
    if file_sha256(run_dir / "best.pt") != entry["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after lock")
    if file_sha256(run_dir / "beauty_mobilenetv3_small.onnx") != entry["onnx_sha256"]:
        raise ValueError("ONNX changed after lock")
    if file_sha256(manifest_path) != lock["combined_manifest_sha256"]:
        raise ValueError("Evaluation manifest changed after lock")
    return lock, entry


def evaluate_test(args):
    import onnxruntime as ort
    # This validation happens before test targets are parsed or images loaded.
    lock, entry = validate_lock(args.lock, args.run, args.test_manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "evaluation.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous["lock_id"] == lock["lock_id"] and previous["onnx_sha256"] == entry["onnx_sha256"]:
            return previous
        raise FileExistsError("Refusing to overwrite an evaluation")
    rows = []
    with Path(args.test_manifest).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "test":
                rows.append({"name": row["name"], "split": "test", "score": float(row["score"])})
    if len(rows) != 1100:
        raise ValueError("Official fold 1 final evaluation requires exactly 1100 test images")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(Path(args.run)/"beauty_mobilenetv3_small.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    predictions = []
    started = time.monotonic()
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        x = np.stack([input_array(args.data_dir, row, entry["config"]) for row in batch])
        predictions.extend(session.run(None, {"input": x})[0].reshape(-1).tolist())
    targets = [row["score"] for row in rows]
    result = {"format": FORMAT_VERSION, "created_utc": utc_now(), "lock_id": lock["lock_id"],
              "onnx_sha256": entry["onnx_sha256"], "checkpoint_sha256": entry["checkpoint_sha256"],
              "test_status": lock["test_status"], "test_raw": metrics(targets, predictions),
              "test_clipped": metrics(targets, np.clip(predictions, 1, 5)),
              "target_scale": [1, 5], "wall_seconds": time.monotonic()-started,
              "limitation": "Previously observed SCUT fold 1; not a newly unseen or real-phone prospective test."}
    write_predictions(output / "test_predictions.csv", rows, np.arange(len(rows)), predictions)
    atomic_json(result_path, result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-manifest")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    t = commands.add_parser("train")
    t.add_argument("--plan", type=Path, required=True)
    t.add_argument("--config", choices=KNOWN_CONFIGS, required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--data-dir", type=Path, required=True)
    t.add_argument("--dev-manifest", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--cache-dir", type=Path, required=True)
    t.add_argument("--max-hours", type=float, default=3.)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--smoke", action="store_true", help="16 train/8 validation, 2+2 epochs; cannot be locked/released")
    t.add_argument("--stop-after-batches", type=int, default=0, help="Pause at a complete minibatch boundary for a pilot/resume check")
    t.add_argument("--stop-after-epochs", type=int, default=0)
    e = commands.add_parser("export")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--data-dir", type=Path, required=True)
    e.add_argument("--dev-manifest", type=Path, required=True)
    e.add_argument("--threads", type=int, default=4)
    lock = commands.add_parser("lock")
    lock.add_argument("--runs", type=Path, nargs="+", required=True)
    lock.add_argument("--test-manifest", type=Path, required=True)
    lock.add_argument("--rationale", required=True)
    lock.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("evaluate-test")
    final.add_argument("--lock", type=Path, required=True)
    final.add_argument("--run", type=Path, required=True)
    final.add_argument("--data-dir", type=Path, required=True)
    final.add_argument("--test-manifest", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--threads", type=int, default=4)
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare-manifest":
        result = prepare_manifest(args.source, args.output)
    elif args.command == "train":
        result = train(args)
    elif args.command == "export":
        result = export_model(args)
    elif args.command == "lock":
        result = create_lock(args)
    else:
        result = evaluate_test(args)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
