#!/usr/bin/env python3
"""
Huấn luyện và đánh giá Perceptron trên split không trùng vector đặc trưng.

Model chỉ học từ train. Validation và test được báo cáo riêng; việc chọn epoch
bằng validation sẽ được thực hiện ở bước tiếp theo.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from feature_contract import (
    FEATURE_DOMAINS,
    FEATURE_NAMES,
    INTERNAL_CLASS_NAMES,
    RAW_LABEL_TO_INTERNAL,
    SCHEMA_VERSION,
)
from perceptron import Perceptron

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = INTERNAL_CLASS_NAMES
SPLIT_NAMES = ("train", "validation", "test")


def load_data(output_directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Đọc ba tập đã xử lý và giữ đúng thứ tự 30 đặc trưng."""
    datasets = {}
    expected_columns = FEATURE_NAMES + ["Result"]
    for split_name in SPLIT_NAMES:
        path = output_directory / f"{split_name}_processed.csv"
        frame = pd.read_csv(path)
        if list(frame.columns) != expected_columns:
            raise ValueError(f"Schema không đúng trong {path}")
        x = frame[FEATURE_NAMES].to_numpy(dtype=float)
        y = frame["Result"].to_numpy(dtype=int)
        datasets[split_name] = (x, y)
    return datasets


def confusion_matrix(y_true, y_pred, n_classes=2):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    return matrix


def metrics(matrix):
    """matrix[i][j]: thực tế i, dự đoán j; lớp 1 là phishing."""
    tn, fp, fn, tp = (
        matrix[0, 0],
        matrix[0, 1],
        matrix[1, 0],
        matrix[1, 1],
    )
    total = int(matrix.sum())
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "accuracy": float(accuracy),
        "misclassification_rate": float(1.0 - accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate(model: Perceptron, x: np.ndarray, y: np.ndarray) -> dict:
    predicted = model.predict(x)
    matrix = confusion_matrix(y, predicted)
    return {
        "confusion_matrix": matrix.tolist(),
        "metrics": metrics(matrix),
    }


def log_evaluation(name: str, evaluation: dict) -> None:
    matrix = np.asarray(evaluation["confusion_matrix"])
    log.info("%s - confusion matrix (hàng=thực tế, cột=dự đoán):", name)
    log.info("            %-12s %-12s", CLASS_NAMES[0], CLASS_NAMES[1])
    log.info("%-12s %-12d %-12d", CLASS_NAMES[0], matrix[0, 0], matrix[0, 1])
    log.info("%-12s %-12d %-12d", CLASS_NAMES[1], matrix[1, 0], matrix[1, 1])
    for metric_name, value in evaluation["metrics"].items():
        log.info("%s %-24s %.4f", name, metric_name, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá Perceptron.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
        help="Thư mục chứa dữ liệu đã xử lý và nơi lưu kết quả",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Tốc độ học")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = load_data(args.out_dir)
    for split_name in SPLIT_NAMES:
        _, labels = datasets[split_name]
        log.info(
            "%s: %d mẫu (%d phishing, %d legitimate)",
            split_name,
            len(labels),
            int((labels == 1).sum()),
            int((labels == 0).sum()),
        )

    x_train, y_train = datasets["train"]
    model = Perceptron(
        n_classes=2,
        alpha=args.alpha,
        max_epochs=args.max_epochs,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    evaluations = {
        split_name: evaluate(model, *datasets[split_name])
        for split_name in ("validation", "test")
    }
    for split_name, evaluation in evaluations.items():
        log_evaluation(split_name, evaluation)

    manifest_path = args.out_dir / "split_manifest.json"
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_domains": {
            name: sorted(values) for name, values in FEATURE_DOMAINS.items()
        },
        "raw_label_to_internal": {
            str(key): value for key, value in RAW_LABEL_TO_INTERNAL.items()
        },
        "internal_class_names": {
            str(key): value for key, value in INTERNAL_CLASS_NAMES.items()
        },
        "data_split": split_manifest,
        "training": {
            "alpha": args.alpha,
            "max_epochs": args.max_epochs,
            "seed": args.seed,
            "epochs_run": len(model.history),
            "updates_per_epoch": model.history,
        },
        "evaluations": evaluations,
    }

    results_path = args.out_dir / "eval_results.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(args.out_dir / "weights.npy", model.W)
    log.info("Đã lưu eval_results.json và weights.npy vào %s", args.out_dir)


if __name__ == "__main__":
    main()
