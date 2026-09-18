#!/usr/bin/env python3
"""
Chọn checkpoint Perceptron bằng validation và đánh giá cuối trên test.

Giai đoạn chọn model chỉ sử dụng train và validation. Sau khi chọn epoch có
misclassification validation thấp nhất, model cuối được huấn luyện lại trên
train + validation đúng số epoch đó. Test chỉ được dùng cho đánh giá cuối.
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
SELECTION_METRIC = "misclassification_rate"


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


def select_checkpoint(
    model: Perceptron,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict:
    """Chọn epoch có validation misclassification thấp nhất; hòa thì lấy sớm nhất."""
    best_key = None
    best_weights = None
    best_evaluation = None
    best_epoch = None
    validation_history = []

    def inspect_epoch(epoch: int, updates: int, current_model: Perceptron) -> None:
        nonlocal best_key, best_weights, best_evaluation, best_epoch
        evaluation = evaluate(current_model, x_validation, y_validation)
        score = evaluation["metrics"][SELECTION_METRIC]
        validation_history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "misclassification_rate": score,
                "accuracy": evaluation["metrics"]["accuracy"],
                "f1": evaluation["metrics"]["f1"],
            }
        )
        candidate_key = (score, epoch)
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_epoch = epoch
            best_weights = current_model.W.copy()
            best_evaluation = evaluation

    model.fit(x_train, y_train, epoch_callback=inspect_epoch)
    if best_weights is None:
        raise RuntimeError("Không tạo được checkpoint từ quá trình huấn luyện")

    model.W = best_weights
    return {
        "best_epoch": best_epoch,
        "best_validation_evaluation": best_evaluation,
        "updates_per_epoch": model.history.copy(),
        "validation_history": validation_history,
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
    parser = argparse.ArgumentParser(
        description="Chọn checkpoint và đánh giá Perceptron."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
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
    x_validation, y_validation = datasets["validation"]
    x_test, y_test = datasets["test"]

    selection_model = Perceptron(
        n_classes=2,
        alpha=args.alpha,
        max_epochs=args.max_epochs,
        random_state=args.seed,
    )
    selection = select_checkpoint(
        selection_model,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )
    best_epoch = selection["best_epoch"]
    validation_evaluation = selection["best_validation_evaluation"]
    log.info(
        "Checkpoint được chọn: epoch %d, validation %s=%.4f",
        best_epoch,
        SELECTION_METRIC,
        validation_evaluation["metrics"][SELECTION_METRIC],
    )
    log_evaluation("validation tại checkpoint", validation_evaluation)

    x_final = np.vstack([x_train, x_validation])
    y_final = np.concatenate([y_train, y_validation])
    permutation = np.random.default_rng(args.seed).permutation(len(y_final))
    x_final = x_final[permutation]
    y_final = y_final[permutation]

    final_model = Perceptron(
        n_classes=2,
        alpha=args.alpha,
        max_epochs=best_epoch,
        random_state=args.seed,
    )
    final_model.fit(x_final, y_final)
    test_evaluation = evaluate(final_model, x_test, y_test)
    log_evaluation("test cuối", test_evaluation)

    split_manifest = json.loads(
        (args.out_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
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
            "max_epochs_examined": args.max_epochs,
            "seed": args.seed,
            "selection_metric": SELECTION_METRIC,
            "tie_break": "earliest_epoch",
            "best_epoch": best_epoch,
            "selection_train_rows": len(y_train),
            "selection_validation_rows": len(y_validation),
            "selection_updates_per_epoch": selection["updates_per_epoch"],
            "validation_history": selection["validation_history"],
            "final_fit_rows": len(y_final),
            "final_fit_sources": ["train", "validation"],
            "final_fit_epochs": len(final_model.history),
            "final_fit_updates_per_epoch": final_model.history,
        },
        "evaluations": {
            "validation": validation_evaluation,
            "test": test_evaluation,
        },
    }

    (args.out_dir / "eval_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(args.out_dir / "weights.npy", final_model.W)
    log.info("Đã lưu model cuối và báo cáo vào %s", args.out_dir)


if __name__ == "__main__":
    main()
