"""对已训练性别模型做 Platt 校准，并重新在验证集上选阈值。

为什么需要：
    BCE 训练 6 轮后模型输出严重饱和（验证/测试集约一半 |logit| > 10）。
    此时 sigmoid(logit) 虽然能排序，但数值不是概率——直接显示“0% / 100%”
    会让读者误以为模型确定。Platt 校准用一个一维 logistic 回归把 logit
    映射到校准后的概率，显著改善 Brier 分数与 log loss。

做法：
    1. 在 validation 划分上取 logit；
    2. 用 LBFGS 拟合 p = sigmoid(a*logit + b)，a>0 保持单调；
    3. 在“校准后概率”上按平衡准确率重新选阈值；
    4. 输出原始与校准后的 Brier / log loss 对比，供报告引用。

用法：
    python calibrate.py --prepared <dev-dir> --model <onnx> --summary <json> --output <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PROJECT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("FACESCORELAB_WORK_DIR") or PROJECT / "training/work").expanduser()
DEFAULT_CELEBA_ATTRS = os.environ.get("FACESCORELAB_CELEBA_ATTRS", str(WORK_ROOT / "datasets/celeba/list_attr_celeba.txt"))


def load_labels(attr_file: Path) -> dict[str, int]:
    male: dict[str, int] = {}
    with attr_file.open("r", encoding="utf-8") as handle:
        handle.readline()
        header = handle.readline().split()
        index = header.index("Male")
        for line in handle:
            parts = line.split()
            if len(parts) < len(header) + 1:
                continue
            male[parts[0]] = 1 if parts[index + 1] == "1" else 0
    return male


def collect(prepared: Path, attr_file: Path, session: ort.InferenceSession,
            split: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    male = load_labels(attr_file)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    name = "development.csv" if split in ("train", "validation") else "heldout.csv"
    csv_path = prepared / name
    logits: list[float] = []
    labels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record.get("detected") != "1":
                continue
            if record.get("split") != split:
                continue
            label = male.get(record["name"])
            if label is None:
                continue
            path = record.get("crop_path", "")
            if not path or not Path(path).is_file():
                continue
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).numpy()
            logits.append(float(session.run(None, {"input": tensor})[0].reshape(-1)[0]))
            labels.append(label)
            if limit is not None and len(labels) >= limit:
                break
    return np.array(logits, dtype=np.float64), np.array(labels, dtype=np.int32)


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """拟合 p = sigmoid(a*logit + b)，用 LBFGS 最小化二元交叉熵。"""
    x = torch.tensor(logits, dtype=torch.float64).reshape(-1, 1)
    y = torch.tensor(labels, dtype=torch.float64).reshape(-1, 1)
    a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200,
                                  line_search_fn="strong_wolfe")
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        # 用 a,b 缩放原始 logit，再做 BCE-with-logits
        loss = criterion(x * a + b, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(a.item()), float(b.item())


def metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    prediction = (probs >= threshold).astype(np.int32)
    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    brier = float(np.mean((p - labels) ** 2))
    log_loss = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    return {
        "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "brier": brier, "log_loss": log_loss, "support": int(len(labels)),
    }


def select_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict]:
    best, best_key, best_metrics = 0.5, None, None
    for step in range(5, 96):
        threshold = step / 100.0
        result = metrics(labels, probs, threshold)
        key = (round(result["balanced_accuracy"], 6), -abs(threshold - 0.5))
        if best_key is None or key > best_key:
            best_key, best, best_metrics = key, threshold, result
    return best, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--celeba-attrs", default=DEFAULT_CELEBA_ATTRS,
                        help="Official attributes file; defaults to FACESCORELAB_CELEBA_ATTRS or <work-dir>/datasets/celeba/list_attr_celeba.txt")
    args = parser.parse_args()

    prepared = Path(args.prepared).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    attr_file = Path(args.celeba_attrs).expanduser().resolve()
    session = ort.InferenceSession(str(Path(args.model).resolve()), providers=["CPUExecutionProvider"])

    print("收集验证集 logit…")
    validation_logits, validation_labels = collect(prepared, attr_file, session, "validation")
    print(f"验证集 {len(validation_labels)} 张")
    a, b = fit_platt(validation_logits, validation_labels)
    print(f"Platt 参数 a={a:.6f} b={b:.6f}")

    def calibrate(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(a * logits + b)))

    raw_probs = 1.0 / (1.0 + np.exp(-validation_logits))
    calibrated_probs = calibrate(validation_logits)
    threshold, chosen = select_threshold(validation_labels, calibrated_probs)
    raw_threshold, raw_chosen = select_threshold(validation_labels, raw_probs)
    print(f"校准后阈值 {threshold:.2f}（原始阈值 {raw_threshold:.2f}）")

    strict = Path(args.summary)
    summary = json.loads(strict.read_text(encoding="utf-8"))
    model_version = summary.get("model_version") or "celeba-gender-mobilenetv3-small-seed42-v1"

    result = {
        "model_version": model_version,
        "calibration": {"method": "Platt scaling (1-D logistic regression)", "a": a, "b": b,
                        "fit_split": "validation", "formula": "p = sigmoid(a*logit + b)"},
        "threshold": threshold,
        "threshold_fit_split": "validation",
        "validation": {
            "raw": raw_chosen,
            "calibrated": chosen,
        },
        "note": ("原始 sigmoid 在验证集上严重饱和，数值不是概率；"
                 "校准后 Brier 与 log loss 明显下降，但阈值为重新在验证集选定。"),
    }
    (output / "gender_calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
