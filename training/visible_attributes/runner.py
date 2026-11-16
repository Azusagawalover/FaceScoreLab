"""Prepare, train and export three visible labels without touching test targets.

Held-out preparation/evaluation are separate commands gated by a frozen model
lock. No gender, ethnicity, health, emotion or orientation targets are used.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import inspect
import io
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from common import *


def now():
    return datetime.now(timezone.utc).isoformat()


def check_lock(lock_path, run=None):
    lock = read_json(lock_path)
    lock_id = lock.pop("lock_id")
    if digest(lock) != lock_id:
        raise ValueError("Visible-label candidate lock was modified")
    lock["lock_id"] = lock_id
    directory = Path(run or lock["run"]).resolve()
    if directory != Path(lock["run"]).resolve():
        raise ValueError("Wrong model run for this lock")
    for file, expected in (("best.pt", lock["checkpoint_sha256"]), ("visible_attributes.onnx", lock["onnx_sha256"]),
                           ("android_metadata.json", lock["android_metadata_sha256"])):
        if sha256(directory/file) != expected:
            raise ValueError(f"{file} changed after candidate freeze")
    return lock


def write_prepared_manifest(path, rows):
    fields = ["name", "split", "image_path", "detected", "crop_path", "source_sha256", "crop_sha256", "face_count", "reason"]
    for label in LABELS:
        fields += [label, label+"_mask"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        value = {key: row.get(key, "") for key in fields if key not in LABELS and not key.endswith("_mask")}
        value["detected"] = int(row["detected"])
        for index, label in enumerate(LABELS):
            value[label] = int(row["labels"][index]) if row["masks"][index] else ""
            value[label+"_mask"] = int(row["masks"][index])
        writer.writerow(value)
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


def validate_subset_protocol(protocol, rows, mode, smoke=False, locked_selection=None):
    expected_seed = 42 if mode == "development" else locked_selection["seed"]
    if smoke and protocol.get("synthetic_fixture"):
        return
    if protocol.get("seed") != expected_seed:
        raise ValueError("Subset sampling seed differs from the registered protocol")
    selections = protocol.get("selected_names", {})
    splits = ("train", "validation") if mode == "development" else ("test",)
    for split in splits:
        names = selections.get(split)
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError("Subset protocol must list unique selected_names for each required split")
        actual = {row["name"] for row in rows if row["split"] == split}
        if (not smoke and actual != set(names)) or (smoke and not actual.issubset(names)):
            raise ValueError("Manifest names differ from the frozen subset selection")


def prepare(args, heldout=False):
    # Check the model lock before any held-out manifest/targets are parsed.
    lock = check_lock(args.lock) if heldout else None
    mode = "heldout" if heldout else "development"
    rows = load_manifest(args.manifest, mode, smoke=args.smoke if not heldout else False)
    validate_subset_protocol(read_json(args.subset_protocol), rows, mode, smoke=args.smoke if not heldout else False,
                             locked_selection=lock["heldout_selection"] if lock else None)
    if heldout and len(rows) != lock["heldout_selection"]["sample_count"]:
        raise ValueError("Held-out sample count differs from the preregistered lock")
    task_hash = sha256(args.face_task)
    contract = preprocessing_contract(task_hash)
    if heldout and contract != lock["preprocessing"]:
        raise ValueError("Held-out preprocessing differs from the frozen candidate")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    crop_dir = output/"crops"
    crop_dir.mkdir(exist_ok=True)
    cache_path = output/"crop_records.json"
    cached = read_json(cache_path) if cache_path.exists() else {}
    account_path = output/"prepare_accounting.json"
    consumed = read_json(account_path).get("wall_seconds", 0.) if account_path.exists() else 0.
    budget = Budget(args.max_hours, consumed)
    source_hash = sha256(args.manifest)
    protocol_hash = sha256(args.subset_protocol)
    metadata = {"format": VERSION, "mode": mode, "status": "preparing", "created_utc": now(), "labels": list(LABELS),
                "source_manifest_sha256": source_hash, "subset_protocol_sha256": protocol_hash,
                "subset_protocol_path": str(Path(args.subset_protocol).resolve()), "preprocessing": contract,
                "smoke": bool(args.smoke) if not heldout else False, "lock_id": lock["lock_id"] if lock else None,
                "identity_note": "Official image-level split; identity independence is not verified.",
                "label_encoding": "0 and 1 are known labels; blank/NA/? are unknown; optional masks applied"}
    previous_metadata = read_json(output/"metadata.json") if (output/"metadata.json").exists() else None
    if previous_metadata:
        for key in ("mode", "source_manifest_sha256", "subset_protocol_sha256", "preprocessing", "smoke", "lock_id"):
            if previous_metadata[key] != metadata[key]:
                raise FileExistsError("Prepared output belongs to another data/protocol version; choose a new directory")
    atomic_json(output/"metadata.json", metadata)
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.face_task)),
                    running_mode=vision.RunningMode.IMAGE, num_faces=3, min_face_detection_confidence=.5,
                    min_face_presence_confidence=.5, min_tracking_confidence=.5)
        contract_key = digest({"contract": contract, "mediapipe": mp.__version__, "pillow": Image.__version__})
        with vision.FaceLandmarker.create_from_options(options) as detector:
            for index, row in enumerate(rows):
                budget.check()
                raw_hash = sha256(row["image_path"])
                old = cached.get(row["name"])
                if previous_metadata and previous_metadata["status"] == "completed" and old and old["source_sha256"] != raw_hash:
                    raise ValueError("Raw image changed since audited preparation; create a new prepared data version")
                can_reuse = old and old.get("source_sha256") == raw_hash and old.get("contract_key") == contract_key
                if can_reuse and old["detected"]:
                    can_reuse = Path(old["crop_path"]).exists() and sha256(old["crop_path"]) == old["crop_sha256"]
                if can_reuse:
                    row.update(old)
                    continue
                with Image.open(row["image_path"]) as opened:
                    image = opened.convert("RGB")
                array = np.asarray(image)
                found = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=array))
                record = {"source_sha256": raw_hash, "contract_key": contract_key, "face_count": len(found.face_landmarks),
                          "detected": False, "crop_path": "", "crop_sha256": "", "reason": "No face detected"}
                if found.face_landmarks:
                    def coordinates(face):
                        return np.asarray([(p.x*image.width, p.y*image.height) for p in face], np.float32)
                    points = max((coordinates(face) for face in found.face_landmarks), key=lambda p: float(np.prod(p.max(0)-p.min(0))))
                    rect = face_crop_rect(points, image.width, image.height)
                    if rect[2] <= rect[0] or rect[3] <= rect[1]:
                        record["reason"] = "Invalid face crop"
                    else:
                        target = crop_dir/(row["name"]+".png")
                        temporary = target.with_name(target.name+".tmp")
                        image.crop(rect).resize((224, 224), Image.Resampling.BILINEAR).save(temporary, format="PNG")
                        temporary.replace(target)
                        record.update(detected=True, crop_path=str(target), crop_sha256=sha256(target), crop=rect, reason="")
                cached[row["name"]] = record
                row.update(record)
                if index % 100 == 0:
                    atomic_json(cache_path, cached)
                    atomic_json(account_path, {"wall_seconds": budget.elapsed(), "completed_images": index+1})
                    print(json.dumps({"event": "face_crops", "count": index+1, "total": len(rows), "wall_seconds": budget.elapsed()}), flush=True)
        manifest_name = "heldout.csv" if heldout else "development.csv"
        write_prepared_manifest(output/manifest_name, rows)
        metadata.update(status="completed", manifest_name=manifest_name, manifest_sha256=sha256(output/manifest_name),
                        counts={s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})},
                        detected_counts={s: sum(r["split"] == s and r["detected"] for r in rows) for s in sorted({r["split"] for r in rows})},
                        wall_seconds=budget.elapsed(), detector_version=mp.__version__)
        atomic_json(output/"metadata.json", metadata)
        return metadata
    except BudgetExceeded:
        metadata.update(status="budget_exhausted", wall_seconds=budget.elapsed())
        atomic_json(output/"metadata.json", metadata)
        return metadata
    finally:
        atomic_json(cache_path, cached)
        atomic_json(account_path, {"wall_seconds": budget.elapsed()})


def read_prepared(directory, mode):
    directory = Path(directory).resolve()
    metadata = read_json(directory/"metadata.json")
    if metadata["mode"] != mode or metadata["status"] != "completed":
        raise ValueError("Prepared data must be completed and belong to the expected development/heldout boundary")
    filename = "development.csv" if mode == "development" else "heldout.csv"
    if metadata["manifest_name"] != filename or sha256(directory/filename) != metadata["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    # Split is checked before any label is converted.
    allowed = {"train", "validation"} if mode == "development" else {"test"}
    rows = []
    with (directory/filename).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in allowed:
                raise ValueError("Prepared split boundary violation")
            values = [parse_label(row[label], row[label+"_mask"]) for label in LABELS]
            rows.append({**{key: row[key] for key in ("name", "split", "image_path", "crop_path", "source_sha256", "crop_sha256")},
                         "detected": row["detected"] == "1", "labels": [v[0] for v in values], "masks": [v[1] for v in values]})
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prepared image names")
    return metadata, rows


def setup(seed, threads):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def feature_cache(rows, metadata, model, directory, use_tail, budget):
    inputs = []
    for row in rows:
        budget.check()
        current_hash = sha256(row["crop_path"])
        if current_hash != row["crop_sha256"]:
            raise ValueError("Prepared crop content changed; prepare a new audited dataset")
        inputs.append({"name": row["name"], "split": row["split"], "crop_sha256": current_hash})
    spec = {"format": VERSION, "manifest_sha256": metadata["manifest_sha256"], "preprocessing": metadata["preprocessing"],
            "tensor_preprocessing_code_sha256": hashlib.sha256(inspect.getsource(tensor_input).encode()).hexdigest(),
            "backbone_weights_sha256": weights_hash(model.features.state_dict()), "inputs": inputs,
            "representation": "prefix_0_8_maps" if use_tail else "pooled_576_vectors", "dtype": "float32",
            "torch": torch.__version__, "numpy": np.__version__, "pillow": Image.__version__, "batch_size": 32}
    key = digest(spec)
    root = Path(directory)/key
    root.mkdir(parents=True, exist_ok=True)
    path, sidecar = root/"features.npy", root/"cache_manifest.json"
    if path.exists() and sidecar.exists():
        saved = read_json(sidecar)
        if saved["key"] != key or saved["npy_sha256"] != sha256(path):
            raise ValueError("Corrupt feature cache")
        values = np.load(path, allow_pickle=False)
        if values.dtype != np.float32 or values.shape[0] != len(rows) or list(values.shape) != saved["shape"]:
            raise ValueError("Feature cache shape/dtype differs from its manifest")
        return torch.from_numpy(values), {"key": key, "path": str(path), "reused": True, "spec": spec}
    prefix = nn.Sequential(*list(model.features.children())[:9]).eval()
    model.eval()
    bank = None
    temporary = root/"features.partial.npy"
    for start in range(0, len(rows), 32):
        budget.check()
        x = torch.from_numpy(np.stack([tensor_input(r["crop_path"]) for r in rows[start:start+32]]))
        with torch.inference_mode():
            result = prefix(x) if use_tail else model.pool(model.features(x)).flatten(1)
        values = result.numpy()
        if bank is None:
            bank = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(rows), *values.shape[1:]))
        bank[start:start+len(values)] = values
        if start % 320 == 0:
            print(json.dumps({"event": "feature_cache", "count": start+len(values), "total": len(rows), "wall_seconds": budget.elapsed()}), flush=True)
    bank.flush()
    del bank
    temporary.replace(path)
    shape = list(np.load(path, mmap_mode="r", allow_pickle=False).shape)
    atomic_json(sidecar, {"key": key, "spec": spec, "shape": shape, "npy_sha256": sha256(path)})
    return torch.from_numpy(np.load(path, allow_pickle=False)), {"key": key, "path": str(path), "reused": False, "spec": spec}


def train(args):
    metadata, all_rows = read_prepared(args.prepared, "development")
    if args.seed != 42 or not 0 <= args.tail_epochs <= 8:
        raise ValueError("Initial protocol fixes seed42 and optional tail0..8")
    if bool(args.smoke) != bool(metadata["smoke"]):
        raise ValueError("Smoke flag must match prepared data")
    config = {"format": VERSION, "labels": list(LABELS), "seed": 42, "batch_size": 64,
              "head_epochs": 2 if args.smoke else 50, "head_patience": 8,
              "tail_epochs": min(args.tail_epochs, 1) if args.smoke else args.tail_epochs, "tail_patience": 3,
              "head_lr": .0008, "tail_head_lr": .0001, "tail_backbone_lr": .000025,
              "weight_decay": .01, "min_f1_improvement": .0001, "pos_weight": True,
              "threshold_grid": THRESHOLD_GRID, "smoke": bool(args.smoke), "selection": "validation_macro_f1_with_validation_only_per_label_thresholds"}
    implementation = {"runner.py": sha256(__file__), "common.py": sha256(Path(__file__).with_name("common.py"))}
    fingerprint = digest({"config": config, "prepared_manifest_sha256": metadata["manifest_sha256"],
                          "preprocessing": metadata["preprocessing"], "implementation": implementation})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    last_path = output/"last.pt"
    saved = None
    if last_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume or a new output directory")
        saved = load_checkpoint(last_path)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Resume protocol/data mismatch")
        if saved["status"] == "completed":
            return read_json(output/"metrics.json")
    elif args.resume:
        raise FileNotFoundError("No complete checkpoint boundary to resume")
    consumed = saved["phase_wall_seconds"] if saved else metadata["wall_seconds"]
    if args.resume and (output/"interruption.json").exists():
        consumed = max(consumed, read_json(output/"interruption.json")["phase_wall_seconds"])
    budget = Budget(args.max_hours, consumed)
    setup(args.seed, args.threads)
    rows = [row for row in all_rows if row["detected"]]
    indices = {s: np.array([i for i, row in enumerate(rows) if row["split"] == s], np.int64) for s in ("train", "validation")}
    if not all(len(v) for v in indices.values()):
        raise ValueError("Detected training and validation images are both required")
    y = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    masks = torch.tensor([r["masks"] for r in rows], dtype=torch.float32)
    counts = class_counts(y[indices["train"]].numpy(), masks[indices["train"]].numpy())
    if any(c["positive"] == 0 or c["negative"] == 0 for c in counts):
        raise ValueError("Every training label must have known positive and negative examples")
    pos_weight = torch.tensor([c["negative"]/c["positive"] for c in counts], dtype=torch.float32)
    # Validate both-class support without fitting a threshold yet.
    validation_counts = class_counts(y[indices["validation"]].numpy(), masks[indices["validation"]].numpy())
    if any(c["positive"] == 0 or c["negative"] == 0 for c in validation_counts):
        raise ValueError("Every validation label must have known positives and negatives")
    model = AttributeNet(pretrained=True)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    tail = nn.Sequential(*list(model.features.children())[9:])
    run = {"format": VERSION, "created_utc": now(), "config": config, "fingerprint": fingerprint,
           "prepared": str(Path(args.prepared).resolve()), "prepared_manifest_sha256": metadata["manifest_sha256"],
           "preprocessing": metadata["preprocessing"], "training_class_counts": counts,
           "validation_class_counts": validation_counts, "positive_weights": pos_weight.tolist(),
           "dataset_total_counts": metadata["counts"], "dataset_detected_counts": metadata["detected_counts"],
           "initialization": "torchvision ImageNet1K V1 only; no CelebA attribute classification checkpoint",
           "test_targets_opened": False, "preparation_charged_seconds": metadata["wall_seconds"],
           "source_sha256": implementation}
    atomic_json(output/"run.json", run)
    state = {"stage_index": 0, "initialized": False, "stage_complete": False, "epoch": 1, "order": None,
             "cursor": 0, "loss_sum": 0., "valid_labels": 0., "best_f1": None, "best_state": None,
             "thresholds": None, "best_stage": None, "best_epoch": None, "stale": 0,
             "logs": [], "steps": 0, "epochs": 0}
    optimizer = None
    cache = None
    process_steps = process_epochs = 0
    val_all_rows = [row for row in all_rows if row["split"] == "validation"]
    val_all_y, val_all_m = np.array([r["labels"] for r in val_all_rows]), np.array([r["masks"] for r in val_all_rows])
    val_all_available = np.array([r["detected"] for r in val_all_rows])
    val_lookup = {row["name"]: i for i, row in enumerate(val_all_rows)}

    def save_last(status):
        atomic_torch(last_path, {"format": VERSION, "fingerprint": fingerprint, "config": config,
            "model_state": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": None, "rng": rng_state(), "state": state, "phase_wall_seconds": budget.elapsed(),
            "cache_key": cache["key"] if cache else None, "status": status})

    def publish(status):
        result = {"format": VERSION, "status": status, "fingerprint": fingerprint, "config": config,
                  "best_validation_macro_f1": state["best_f1"], "best_stage": state["best_stage"],
                  "best_epoch": state["best_epoch"], "thresholds": state["thresholds"],
                  "steps_completed": state["steps"], "epochs_completed": state["epochs"],
                  "phase_wall_seconds": budget.elapsed(), "preparation_charged_seconds": metadata["wall_seconds"],
                  "test_evaluated": False, "smoke_not_for_release": config["smoke"],
                  "positive_weights": pos_weight.tolist(), "training_class_counts": counts,
                  "cache_key": cache["key"] if cache else None}
        if state["best_state"] is not None:
            result["best_checkpoint_sha256"] = sha256(output/"best.pt")
            result["validation"] = read_json(output/"best_validation_metrics.json")
        atomic_json(output/"training_log.json", state["logs"])
        atomic_json(output/"metrics.json", result)
        return result

    def make_optimizer(stage_index):
        for parameter in tail.parameters():
            parameter.requires_grad = stage_index == 1
        if stage_index == 0:
            return torch.optim.AdamW(model.head_parameters(), lr=config["head_lr"], weight_decay=config["weight_decay"])
        return torch.optim.AdamW([{"params": tail.parameters(), "lr": config["tail_backbone_lr"]},
                                  {"params": model.head_parameters(), "lr": config["tail_head_lr"]}], weight_decay=config["weight_decay"])

    try:
        cached, cache = feature_cache(rows, metadata, model, args.cache_dir, bool(config["tail_epochs"]), budget)
        if config["tail_epochs"]:
            tail.eval()
            vectors = []
            with torch.inference_mode():
                for start in range(0, len(rows), 64):
                    budget.check()
                    vectors.append(model.pool(tail(cached[start:start+64])).flatten(1))
            vectors = torch.cat(vectors)
        else:
            vectors = cached
        if saved:
            if saved["cache_key"] != cache["key"]:
                raise ValueError("Resume cache identity changed")
            model.load_state_dict(saved["model_state"])
            state = saved["state"]
            if state["initialized"]:
                optimizer = make_optimizer(state["stage_index"])
                optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"])
        while state["stage_index"] < (2 if config["tail_epochs"] else 1):
            stage_index = state["stage_index"]
            stage = "head" if stage_index == 0 else "tail"
            if state["stage_complete"]:
                state.update(stage_index=stage_index+1, initialized=False, stage_complete=False, epoch=1,
                             order=None, cursor=0, loss_sum=0., valid_labels=0., stale=0)
                optimizer = None
                continue
            if not state["initialized"]:
                if stage_index == 1:
                    model.load_state_dict(state["best_state"])
                optimizer = make_optimizer(stage_index)
                state["initialized"] = True
            if state["order"] is None:
                state["order"] = indices["train"][np.random.permutation(len(indices["train"]))]
                state.update(cursor=0, loss_sum=0., valid_labels=0.)
            started = time.monotonic()
            while state["cursor"] < len(state["order"]):
                budget.check()
                if args.stop_after_steps and process_steps >= args.stop_after_steps:
                    save_last("paused_for_pilot")
                    return publish("paused_for_pilot")
                batch = state["order"][state["cursor"]:state["cursor"]+config["batch_size"]]
                model.train_heads()
                optimizer.zero_grad(set_to_none=True)
                feature = vectors[batch] if stage_index == 0 else model.pool(tail(cached[batch])).flatten(1)
                logits = model.classify(feature)
                loss, valid_count = masked_bce(logits, y[batch], masks[batch], pos_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite masked BCE")
                if valid_count.item():
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.head_parameters()+(list(tail.parameters()) if stage_index == 1 else []), 5)
                    optimizer.step()
                state["loss_sum"] += float(loss.item()*valid_count.item())
                state["valid_labels"] += float(valid_count.item())
                state["cursor"] += len(batch)
                state["steps"] += 1
                process_steps += 1
            model.eval()
            outputs = []
            with torch.inference_mode():
                for start in range(0, len(indices["validation"]), 128):
                    batch = indices["validation"][start:start+128]
                    feature = vectors[batch] if stage_index == 0 else model.pool(tail(cached[batch])).flatten(1)
                    outputs.append(model.classify(feature).numpy())
            scores = sigmoid(np.concatenate(outputs))
            val_y, val_m = y[indices["validation"]].numpy(), masks[indices["validation"]].numpy()
            thresholds = choose_thresholds(val_y, val_m, scores)
            full_scores = np.full((len(val_all_rows), 3), np.nan)
            for index, score in zip(indices["validation"], scores):
                full_scores[val_lookup[rows[index]["name"]]] = score
            measured = evaluate_scores(val_all_y, val_all_m, full_scores, thresholds, val_all_available)
            macro_f1 = measured["macro_f1"]
            improved = state["best_f1"] is None or macro_f1 > state["best_f1"]+config["min_f1_improvement"]
            if improved:
                state.update(best_f1=macro_f1, best_state=copy.deepcopy(model.state_dict()), thresholds=thresholds,
                             best_stage=stage, best_epoch=state["epoch"], stale=0)
                atomic_torch(output/"best.pt", {"format": VERSION, "fingerprint": fingerprint, "config": config,
                    "model_state": state["best_state"], "thresholds": thresholds, "validation": measured,
                    "preprocessing": metadata["preprocessing"], "prepared_manifest_sha256": metadata["manifest_sha256"],
                    "best_stage": stage, "best_epoch": state["epoch"], "test_evaluated": False})
                atomic_json(output/"best_validation_metrics.json", measured)
                write_predictions(output/"validation_predictions.csv", val_all_rows, full_scores, thresholds)
            else:
                state["stale"] += 1
            log = {"stage": stage, "epoch": state["epoch"], "masked_train_bce": state["loss_sum"]/max(1, state["valid_labels"]),
                   "validation_macro_f1": macro_f1, "validation": measured, "thresholds": thresholds,
                   "improved": improved, "steps": state["steps"], "epoch_this_process_seconds": time.monotonic()-started,
                   "phase_wall_seconds": budget.elapsed()}
            state["logs"].append(log)
            print(json.dumps({key:value for key,value in log.items() if key != "validation"}), flush=True)
            state["epochs"] += 1
            process_epochs += 1
            limit = config["head_epochs"] if stage_index == 0 else config["tail_epochs"]
            patience = config["head_patience"] if stage_index == 0 else config["tail_patience"]
            state["stage_complete"] = state["epoch"] >= limit or state["stale"] >= patience
            state.update(epoch=state["epoch"]+1, order=None, cursor=0, loss_sum=0., valid_labels=0.)
            save_last("running")
            publish("running")
            if args.stop_after_epochs and process_epochs >= args.stop_after_epochs:
                save_last("paused_for_pilot")
                return publish("paused_for_pilot")
        model.load_state_dict(state["best_state"])
        save_last("completed")
        return publish("completed")
    except BudgetExceeded:
        if cache:
            save_last("budget_exhausted")
        return publish("budget_exhausted")
    except KeyboardInterrupt:
        atomic_json(output/"interruption.json", {"phase_wall_seconds": budget.elapsed(), "note": "Resume last complete checkpoint boundary"})
        raise


def export_model(args):
    import onnx
    import onnxruntime as ort
    directory = Path(args.run).resolve()
    if read_json(directory/"metrics.json")["status"] != "completed":
        raise ValueError("Export requires a completed run")
    best = load_checkpoint(directory/"best.pt")
    metadata, rows = read_prepared(args.prepared, "development")
    if best["prepared_manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Export development data differs from checkpoint")
    setup(42, args.threads)
    model = AttributeNet(pretrained=False).eval()
    model.load_state_dict(best["model_state"])
    path = directory/"visible_attributes.onnx"
    temporary = directory/"visible_attributes.tmp.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(temporary), input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(temporary))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
    parity = []
    for row in [r for r in rows if r["split"] == "validation" and r["detected"]][:20]:
        if sha256(row["crop_path"]) != row["crop_sha256"]:
            raise ValueError("Validation crop changed before export")
        x = tensor_input(row["crop_path"])[None]
        with torch.inference_mode():
            native = model(torch.from_numpy(x)).numpy()
        converted = session.run(None, {"input": x})[0]
        difference = float(np.max(np.abs(native-converted)))
        decisions_match = bool(np.array_equal(sigmoid(native) >= best["thresholds"], sigmoid(converted) >= best["thresholds"]))
        if difference >= 1e-4 or not decisions_match:
            raise AssertionError("Visible-label export validation parity failed")
        parity.append({"name": row["name"], "max_logit_difference": difference, "decisions_match": decisions_match})
    del session
    temporary.replace(path)
    android = {"format": VERSION, "model_file": path.name, "model_sha256": sha256(path),
               "input": {"name": "input", "shape": ["N", 3, 224, 224], "dtype": "float32", "layout": "NCHW"},
               "output": {"name": "logits", "shape": ["N", 3], "activation": "independent sigmoid, NOT softmax"},
               "labels": list(LABELS), "display_labels": list(DISPLAY_LABELS), "thresholds": best["thresholds"],
               "decision": "sigmoid(logit[i]) >= thresholds[i]", "threshold_fit_split": "validation",
               "preprocessing": best["preprocessing"], "score_type": "uncalibrated sigmoid score; not calibrated probability",
               "no_face_behavior": "No output; preserve detection failure/coverage rather than return negative labels",
               "existing_models": "Separate optional model; existing MediaPipe mouth actions and FaceAttribNet five labels remain distinct",
               "smoke_not_for_release": best["config"]["smoke"]}
    atomic_json(directory/"android_metadata.json", android)
    result = {"format": VERSION, "checkpoint_sha256": sha256(directory/"best.pt"), "onnx_sha256": sha256(path),
              "android_metadata_sha256": sha256(directory/"android_metadata.json"), "parity": parity,
              "max_logit_difference": max(p["max_logit_difference"] for p in parity),
              "parity_split": "validation", "test_targets_opened": False, "model_bytes": path.stat().st_size,
              "smoke_not_for_release": best["config"]["smoke"]}
    atomic_json(directory/"export_metadata.json", result)
    return result


def lock_candidate(args):
    directory = Path(args.run).resolve()
    metric = read_json(directory/"metrics.json")
    exported = read_json(directory/"export_metadata.json")
    android = read_json(directory/"android_metadata.json")
    if metric["status"] != "completed" or metric["smoke_not_for_release"] or exported["smoke_not_for_release"]:
        raise ValueError("Only a completed non-smoke candidate can be locked")
    if sha256(directory/"best.pt") != exported["checkpoint_sha256"] or sha256(directory/"visible_attributes.onnx") != exported["onnx_sha256"]:
        raise ValueError("Export identity mismatch")
    if sha256(directory/"android_metadata.json") != exported["android_metadata_sha256"]:
        raise ValueError("Threshold/preprocessing metadata changed after export")
    if args.heldout_count <= 0:
        raise ValueError("Held-out count must be explicitly positive")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Candidate locks are immutable")
    lock = {"format": VERSION, "created_utc": now(), "run": str(directory),
            "checkpoint_sha256": exported["checkpoint_sha256"], "onnx_sha256": exported["onnx_sha256"],
            "android_metadata_sha256": exported["android_metadata_sha256"], "thresholds": android["thresholds"],
            "preprocessing": android["preprocessing"], "selection_basis": "validation macro F1 and validation-only class thresholds",
            "heldout_selection": {"pool": "official CelebA test only", "sample_count": args.heldout_count,
                                  "seed": args.heldout_seed, "identity_independent": False}}
    lock["lock_id"] = digest(lock)
    atomic_json(destination, lock)
    return lock


def evaluate_heldout(args):
    import onnxruntime as ort
    lock = check_lock(args.lock, args.run)
    metadata, rows = read_prepared(args.prepared, "heldout")
    if metadata["lock_id"] != lock["lock_id"] or metadata["preprocessing"] != lock["preprocessing"]:
        raise ValueError("Held-out preparation does not belong to this frozen candidate")
    if len(rows) != lock["heldout_selection"]["sample_count"]:
        raise ValueError("Held-out count differs from lock")
    output = Path(args.output)
    if (output/"evaluation.json").exists():
        raise FileExistsError("Preserve the original held-out evaluation; choose a new audited output path")
    output.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_hours)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(Path(args.run)/"visible_attributes.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    scores = np.full((len(rows), 3), np.nan)
    detected = [i for i,r in enumerate(rows) if r["detected"]]
    for start in range(0, len(detected), 32):
        budget.check()
        batch = detected[start:start+32]
        for i in batch:
            if sha256(rows[i]["crop_path"]) != rows[i]["crop_sha256"]:
                raise ValueError("Held-out crop content changed")
        x = np.stack([tensor_input(rows[i]["crop_path"]) for i in batch])
        scores[batch] = sigmoid(session.run(None, {"input": x})[0])
    metrics = evaluate_scores(np.array([r["labels"] for r in rows]), np.array([r["masks"] for r in rows]), scores,
                              lock["thresholds"], np.array([r["detected"] for r in rows]))
    result = {"format": VERSION, "lock_id": lock["lock_id"], "onnx_sha256": lock["onnx_sha256"],
              "prepared_manifest_sha256": metadata["manifest_sha256"], "thresholds": lock["thresholds"],
              "thresholds_retuned_on_test": False, "heldout": metrics, "wall_seconds": budget.elapsed(),
              "scope": "Three visible CelebA labels on official image-level held-out samples; not real-phone accuracy or identity-disjoint evaluation"}
    write_predictions(output/"heldout_predictions.csv", rows, scores, lock["thresholds"])
    atomic_json(output/"evaluation.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "prepare-heldout"):
        q = commands.add_parser(name)
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--subset-protocol", type=Path, required=True)
        q.add_argument("--face-task", type=Path, required=True)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--max-hours", type=float, default=2.)
        if name == "prepare-heldout":
            q.add_argument("--lock", type=Path, required=True)
        else:
            q.add_argument("--smoke", action="store_true")
    t = commands.add_parser("train")
    t.add_argument("--prepared", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--cache-dir", type=Path, required=True)
    t.add_argument("--seed", type=int, choices=[42], default=42)
    t.add_argument("--tail-epochs", type=int, default=0)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--max-hours", type=float, default=2.)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--stop-after-steps", type=int, default=0)
    t.add_argument("--stop-after-epochs", type=int, default=0)
    e = commands.add_parser("export")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--prepared", type=Path, required=True)
    e.add_argument("--threads", type=int, default=4)
    lock = commands.add_parser("lock")
    lock.add_argument("--run", type=Path, required=True)
    lock.add_argument("--heldout-count", type=int, required=True)
    lock.add_argument("--heldout-seed", type=int, required=True)
    lock.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("evaluate-heldout")
    final.add_argument("--lock", type=Path, required=True)
    final.add_argument("--run", type=Path, required=True)
    final.add_argument("--prepared", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--threads", type=int, default=4)
    final.add_argument("--max-hours", type=float, default=2.)
    return p


def main():
    args = parser().parse_args()
    if args.command in ("prepare", "prepare-heldout"):
        result = prepare(args, args.command == "prepare-heldout")
    elif args.command == "train":
        result = train(args)
    elif args.command == "export":
        result = export_model(args)
    elif args.command == "lock":
        result = lock_candidate(args)
    else:
        result = evaluate_heldout(args)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
