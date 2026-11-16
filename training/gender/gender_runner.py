"""CelebA 性别（Male）二分类训练、验证与 ONNX 导出。

设计原则（与项目既有实验规范一致）：
1. 只使用 CelebA 官方 Male 标签；标签 +1 -> 1（男），-1 -> 0（女）。
2. 复用 visible_attributes 已生成的 MediaPipe 人脸裁剪图，保证与 App 端
   预处理（faceCrop v1 + 224x224 + ImageNet 归一化）完全一致。
3. 数据划分沿用既有 subset_protocol.json：训练只用 train 划分，
   验证只用 validation 划分；官方 test 划分冻结，仅在 lock 阶段评测一次。
4. 阈值只在验证集上按平衡准确率选择，选定后冻结。
5. 导出 ONNX 时固定单输入单输出，输入名与现有模型一致。

本脚本不修改 App 资产；部署是单独步骤。

用法：
    python gender_runner.py train   --prepared <dir> --output <dir> [--seed 42]
    python gender_runner.py lock    --prepared <heldout-dir> --model <onnx> --output <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
LABEL_COLUMN = "Male"
PROJECT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("FACESCORELAB_WORK_DIR") or PROJECT / "training/work").expanduser()
DEFAULT_CELEBA_ATTRS = os.environ.get("FACESCORELAB_CELEBA_ATTRS", str(WORK_ROOT / "datasets/celeba/list_attr_celeba.txt"))


# --------------------------------------------------------------------- 数据


class CropDataset(Dataset):
    """读取 prepared 目录下的裁剪图与标签。"""

    def __init__(self, rows: list[dict], augment: bool):
        self.rows = rows
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as handle:
            image = handle.convert("RGB")
        tensor = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label


def read_prepared(csv_path: Path, split: str | None) -> list[dict]:
    """读取 development.csv，按 split 过滤；Male 标签可能不在 CSV 内，后续再补。"""
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            if record.get("detected", "1") not in ("1", "true", "True"):
                continue
            if split is not None and record.get("split") != split:
                continue
            crop_path = record.get("crop_path", "")
            if not crop_path or not Path(crop_path).is_file():
                continue
            raw = record.get(LABEL_COLUMN, "")
            label = int(raw) if raw in ("0", "1") else None
            rows.append({"name": record["name"], "crop_path": crop_path, "label": label})
    return rows


def attach_labels_from_celeba(rows: list[dict], attr_file: Path) -> int:
    """从 CelebA 官方属性表补 Male 标签；返回成功匹配数。"""
    male: dict[str, int] = {}
    with attr_file.open("r", encoding="utf-8") as handle:
        handle.readline()  # 行数
        header = handle.readline().split()
        index = header.index(LABEL_COLUMN)
        for line in handle:
            parts = line.split()
            if len(parts) < len(header) + 1:
                continue
            male[parts[0]] = 1 if parts[index + 1] == "1" else 0
    matched = 0
    for row in rows:
        value = male.get(row["name"])
        if value is not None:
            row["label"] = value
            matched += 1
    return matched


def keep_labelled(rows: list[dict]) -> list[dict]:
    """丢弃没有有效 Male 标签的行，避免把 None 当作阴性。"""
    return [row for row in rows if row.get("label") in (0, 1)]


# --------------------------------------------------------------------- 模型


def build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


# --------------------------------------------------------------------- 指标


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    prediction = (scores >= threshold).astype(np.int32)
    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    accuracy = (tp + tn) / max(1, len(labels))
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    balanced = (sensitivity + specificity) / 2 if np.isfinite(sensitivity) and np.isfinite(specificity) else float("nan")
    return {
        "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1, "accuracy": accuracy,
        "balanced_accuracy": balanced, "support": int(len(labels)),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    """在验证集上按平衡准确率选阈值；平局取最接近 0.5 的。"""
    best_threshold, best_metrics = 0.5, None
    best_key = None
    for step in range(5, 96, 1):
        threshold = step / 100.0
        metrics = binary_metrics(labels, scores, threshold)
        key = (round(metrics["balanced_accuracy"], 6), -abs(threshold - 0.5))
        if best_key is None or key > best_key:
            best_key, best_threshold, best_metrics = key, threshold, metrics
    return best_threshold, best_metrics


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """秩和法计算 AUROC；不做 pROC 的平滑近似。"""
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 并列取平均秩
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            average = (i + j) / 2.0 + 1
            ranks[order[i:j + 1]] = average
        i = j + 1
    positive_rank_sum = float(ranks[labels == 1].sum())
    n1, n0 = len(positives), len(negatives)
    return (positive_rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------- 训练


@dataclass
class TrainState:
    history: list[dict] = field(default_factory=list)


def run_train(args: argparse.Namespace) -> int:
    prepared = Path(args.prepared).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = prepared / "development.csv"
    if not csv_path.is_file():
        print(f"缺少 development.csv: {csv_path}", file=sys.stderr)
        return 2

    train_rows = read_prepared(csv_path, "train")
    validation_rows = read_prepared(csv_path, "validation")
    attr_file = Path(args.celeba_attrs).expanduser().resolve()
    if attr_file.is_file():
        matched_train = attach_labels_from_celeba(train_rows, attr_file)
        matched_validation = attach_labels_from_celeba(validation_rows, attr_file)
        print(f"从 CelebA 属性表补充 Male 标签：train {matched_train}，validation {matched_validation}")
    else:
        print(f"未找到 CelebA 属性表 {attr_file}，仅使用 CSV 内的 Male 列。", file=sys.stderr)

    train_rows = keep_labelled(train_rows)
    validation_rows = keep_labelled(validation_rows)
    if not train_rows or not validation_rows:
        print(f"样本不足：train={len(train_rows)} validation={len(validation_rows)}", file=sys.stderr)
        return 2
    positives = sum(row["label"] for row in train_rows)
    ratio = positives / len(train_rows)
    print(f"训练样本 {len(train_rows)}（男 {positives}，女 {len(train_rows) - positives}，男占比 {ratio:.3f}）")
    print(f"验证样本 {len(validation_rows)}")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, args.threads))

    train_dataset = CropDataset(train_rows, augment=True)
    validation_dataset = CropDataset(validation_rows, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0)

    device = torch.device("cpu")
    model = build_model().to(device)

    # 处理类别不平衡：男（正类）占比约 0.42，给正类更大权重。
    positive_weight = torch.tensor([(1 - ratio) / max(ratio, 1e-6)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    state = TrainState()
    best_score = -1.0
    best_path = output / "best.pt"
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        train_loss = total_loss / max(1, seen)

        labels_list, score_list = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in validation_loader:
                logits = model(images.to(device)).squeeze(1)
                score_list.append(torch.sigmoid(logits).cpu().numpy())
                labels_list.append(labels.numpy())
        validation_labels = np.concatenate(labels_list)
        validation_scores = np.concatenate(score_list)
        threshold, metrics = select_threshold(validation_labels, validation_scores)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics["auroc"] = auroc(validation_labels, validation_scores)
        metrics["elapsed_seconds"] = round(time.time() - started, 1)
        state.history.append(metrics)
        print(f"epoch {epoch:02d} loss {train_loss:.4f} "
              f"balAcc {metrics['balanced_accuracy']:.4f} AUROC {metrics['auroc']:.4f} "
              f"thr {threshold:.2f} sens {metrics['sensitivity']:.4f} spec {metrics['specificity']:.4f}")

        # 用平衡准确率选择 checkpoint；AUROC 作为并列时的次序依据。
        score = (round(metrics["balanced_accuracy"], 6), round(metrics["auroc"], 6))
        if score > (round(best_score, 6), -1.0):
            best_score = metrics["balanced_accuracy"]
            torch.save({"model": model.state_dict(), "threshold": threshold,
                        "metrics": metrics, "seed": seed}, best_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"验证指标 {args.patience} 轮未提升，提前结束。")
            break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    frozen_threshold = float(checkpoint["threshold"])
    print(f"选中 checkpoint：平衡准确率 {checkpoint['metrics']['balanced_accuracy']:.4f}，"
          f"冻结阈值 {frozen_threshold:.2f}")

    export_onnx(model, output / "gender_mobilenetv3_small.onnx")

    summary = {
        "task": "CelebA Male binary classification",
        "label": LABEL_COLUMN,
        "label_mapping": "+1 -> 1 (male), -1 -> 0 (female)",
        "seed": seed,
        "epochs_requested": args.epochs,
        "epochs_run": len(state.history),
        "train_samples": len(train_rows),
        "train_male": int(positives),
        "train_female": int(len(train_rows) - positives),
        "validation_samples": len(validation_rows),
        "preprocessing": {
            "source": "visible_attributes prepared crops",
            "resize": "224x224",
            "mean": list(MEAN), "std": list(STD), "layout": "NCHW", "dtype": "float32",
        },
        "frozen_threshold": frozen_threshold,
        "selected_metrics": checkpoint["metrics"],
        "history": state.history,
        "onnx": "gender_mobilenetv3_small.onnx",
        "scope_note": (
            "CelebA 官方图片级划分；身份表未取得，不能宣称身份独立。"
            "阈值只在验证集上选定并冻结；官方 test 划分未参与训练或选型。"
            "F1 不是总体准确率；sigmoid 输出未经概率校准。"
        ),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {output / 'training_summary.json'}")
    return 0


def export_onnx(model: nn.Module, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logit"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
    )
    size = path.stat().st_size
    print(f"已导出 {path.name}（{size} 字节）")
    # 用 ONNX Runtime 复核一次，确保导出可加载且数值一致。
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            reference = float(model(dummy).squeeze().item())
        value = float(session.run(None, {"input": dummy.numpy()})[0].reshape(-1)[0])
        print(f"ONNX 复核：PyTorch {reference:.6f} / ORT {value:.6f} 差 {abs(reference - value):.2e}")
    except Exception as error:  # noqa: BLE001
        print(f"ONNX 复核跳过：{error}", file=sys.stderr)


# --------------------------------------------------------------------- 锁定评测


def run_lock(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    prepared = Path(args.prepared).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 开发集用 development.csv，锁定测试集用 heldout.csv；两者列结构相同。
    csv_path = None
    for name in ("heldout.csv", "development.csv"):
        candidate = prepared / name
        if candidate.is_file():
            csv_path = candidate
            break
    if csv_path is None:
        print(f"目录内没有 heldout.csv 或 development.csv：{prepared}", file=sys.stderr)
        return 2
    print(f"锁定评测清单：{csv_path.name}")
    rows = read_prepared(csv_path, "test")
    attr_file = Path(args.celeba_attrs).expanduser().resolve()
    if attr_file.is_file():
        attach_labels_from_celeba(rows, attr_file)
    rows = keep_labelled(rows)
    if not rows:
        print("测试集为空。", file=sys.stderr)
        return 2

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    threshold = float(summary["frozen_threshold"])

    session = ort.InferenceSession(str(Path(args.model).resolve()), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    labels_list, scores_list = [], []
    failures = 0
    for row in rows:
        try:
            with Image.open(row["crop_path"]) as handle:
                tensor = transform(handle.convert("RGB")).unsqueeze(0).numpy()
            score = float(session.run(None, {input_name: tensor})[0].reshape(-1)[0])
            labels_list.append(row["label"])
            # 冻结阈值作用在 sigmoid 输出上
            scores_list.append(1.0 / (1.0 + np.exp(-score)))
        except Exception:  # noqa: BLE001
            failures += 1
    labels = np.array(labels_list, dtype=np.int32)
    scores = np.array(scores_list, dtype=np.float64)
    metrics = binary_metrics(labels, scores, threshold)
    metrics["auroc"] = auroc(labels, scores)
    metrics["total_rows"] = len(rows)
    metrics["valid_outputs"] = len(labels)
    metrics["coverage"] = len(labels) / max(1, len(rows))
    metrics["failures"] = failures
    result = {"frozen_threshold": threshold, "heldout": metrics,
              "scope_note": "官方 test 划分只在此处评测一次，不参与训练或阈值选择。"}
    (output / "heldout_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CelebA 性别二分类训练与评测")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="训练并导出 ONNX")
    train.add_argument("--prepared", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--celeba-attrs", default=DEFAULT_CELEBA_ATTRS,
                       help="Official attributes file; defaults to FACESCORELAB_CELEBA_ATTRS or <work-dir>/datasets/celeba/list_attr_celeba.txt")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=48)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--threads", type=int, default=8)
    train.add_argument("--patience", type=int, default=3)
    train.set_defaults(func=run_train)

    lock = sub.add_parser("lock", help="在官方 test 划分上锁定评测一次")
    lock.add_argument("--prepared", required=True)
    lock.add_argument("--model", required=True)
    lock.add_argument("--summary", required=True)
    lock.add_argument("--output", required=True)
    lock.add_argument("--celeba-attrs", default=DEFAULT_CELEBA_ATTRS,
                      help="Official attributes file; defaults to FACESCORELAB_CELEBA_ATTRS or <work-dir>/datasets/celeba/list_attr_celeba.txt")
    lock.set_defaults(func=run_lock)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
