#!/usr/bin/env python3
"""
train_eval.py — Giai đoạn 2 (GĐ2): Huấn luyện và đánh giá Perceptron.

Quy trình:
  1. Đọc dữ liệu đã xử lý (output/train_processed.csv, output/test_processed.csv).
  2. Huấn luyện Perceptron (from scratch, src/perceptron.py) trên tập train.
  3. Đánh giá trên tập test bằng tỷ lệ phân loại sai (misclassification rate)
     theo đúng tiêu chí trong giáo trình, kèm confusion matrix,
     precision / recall / F1.

Nhãn gốc trong dataset: -1 = phishing (lừa đảo), 1 = legitimate (hợp lệ).
Sau tiền xử lý, nhãn được mã hoá thành 0 (legitimate) / 1 (phishing).
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from perceptron import Perceptron
from feature_contract import (
    FEATURE_DOMAINS,
    FEATURE_NAMES,
    INTERNAL_CLASS_NAMES,
    RAW_LABEL_TO_INTERNAL,
    SCHEMA_VERSION,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = INTERNAL_CLASS_NAMES


def load_data(out_dir: Path):
    train = pd.read_csv(out_dir / "train_processed.csv")
    test = pd.read_csv(out_dir / "test_processed.csv")
    X_train = train.drop(columns=["Result"]).to_numpy(dtype=float)
    y_train = train["Result"].to_numpy(dtype=int)
    X_test = test.drop(columns=["Result"]).to_numpy(dtype=float)
    y_test = test["Result"].to_numpy(dtype=int)
    return X_train, y_train, X_test, y_test


def confusion_matrix(y_true, y_pred, n_classes=2):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics(cm):
    """cm[i][j]: thực tế i, dự đoán j. Lớp 1 = phishing (positive)."""
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (tp + tn) / cm.sum()
    misclass_rate = (fp + fn) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "misclassification_rate": misclass_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện & đánh giá Perceptron.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
        help="Thư mục chứa dữ liệu đã xử lý và nơi lưu kết quả",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Tốc độ học")
    parser.add_argument("--max-epochs", type=int, default=100, help="Số epoch tối đa")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.out_dir)
    log.info(
        "Train: %d mẫu (%d phishing) | Test: %d mẫu (%d phishing)",
        len(y_train), int((y_train == 1).sum()),
        len(y_test), int((y_test == 1).sum()),
    )

    clf = Perceptron(n_classes=2, alpha=args.alpha,
                     max_epochs=args.max_epochs, random_state=args.seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    m = metrics(cm)

    log.info("Confusion matrix (hàng = thực tế, cột = dự đoán):")
    log.info("            %-12s %-12s", CLASS_NAMES[0], CLASS_NAMES[1])
    log.info("%-12s %-12d %-12d", CLASS_NAMES[0], cm[0, 0], cm[0, 1])
    log.info("%-12s %-12d %-12d", CLASS_NAMES[1], cm[1, 0], cm[1, 1])
    for k, v in m.items():
        log.info("%-24s %.4f", k, v)

    # Lưu kết quả
    result = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_domains": {name: sorted(values) for name, values in FEATURE_DOMAINS.items()},
        "raw_label_to_internal": {str(k): v for k, v in RAW_LABEL_TO_INTERNAL.items()},
        "internal_class_names": {str(k): v for k, v in INTERNAL_CLASS_NAMES.items()},
        "alpha": args.alpha,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "converged_epoch": len(clf.history),
        "updates_per_epoch": clf.history,
        "confusion_matrix": cm.tolist(),
        "metrics": m,
    }
    with open(args.out_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.save(args.out_dir / "weights.npy", clf.W)
    log.info("Đã lưu eval_results.json và weights.npy vào %s", args.out_dir)


if __name__ == "__main__":
    main()
