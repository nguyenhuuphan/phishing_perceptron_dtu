#!/usr/bin/env python3
"""
Tiền xử lý bộ UCI Phishing Websites.

Đọc tập ARFF đầy đủ, kiểm tra schema, mã hóa nhãn và tạo ba tập
train/validation/test. Các hàng có cùng vector 30 đặc trưng luôn được đặt trong
cùng một tập để không làm rò rỉ một đầu vào giống hệt sang tập đánh giá.
"""

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from feature_contract import (
    FEATURE_NAMES,
    RAW_LABEL_TO_INTERNAL,
    validate_feature_values,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLUMNS = FEATURE_NAMES
LABEL_COLUMN = "Result"
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_SPLIT_SEED = 42


def load_arff(path: Path) -> pd.DataFrame:
    """Đọc file ARFF cố định của dự án mà không phụ thuộc SciPy."""
    expected = FEATURE_COLUMNS + [LABEL_COLUMN]
    columns: list[str] = []
    rows: list[list[float]] = []
    in_data = False

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                    continue
                match = re.match(
                    r"""(?i)^@attribute\s+(?:'([^']+)'|"([^"]+)"|(\S+))""",
                    line,
                )
                if match:
                    columns.append(next(value for value in match.groups() if value))
                continue

            values = next(csv.reader([line], skipinitialspace=True))
            if len(values) != len(columns):
                raise ValueError(
                    f"Dòng {line_number} có {len(values)} giá trị; "
                    f"schema khai báo {len(columns)} cột"
                )
            parsed = [
                np.nan if value.strip() == "?" else int(value.strip())
                for value in values
            ]
            rows.append(parsed)

    if not in_data:
        raise ValueError(f"Không tìm thấy phần @data trong {path}")
    if columns != expected:
        raise ValueError(f"Cột không khớp mong đợi: {columns}")

    frame = pd.DataFrame(rows, columns=columns)
    validate_feature_values(frame)
    raw_labels = set(frame[LABEL_COLUMN].dropna().unique())
    if not raw_labels <= set(RAW_LABEL_TO_INTERNAL):
        raise ValueError(
            f"Nhãn chứa giá trị ngoài {{-1, 1}}: {sorted(raw_labels)}"
        )
    if frame[LABEL_COLUMN].isna().any():
        raise ValueError("Nhãn Result không được thiếu")
    return frame


def encode_label(series: pd.Series) -> pd.Series:
    """Mã hóa Result: UCI -1/+1 thành nội bộ phishing=1/legitimate=0."""
    encoded = series.map(RAW_LABEL_TO_INTERNAL)
    if encoded.isna().any():
        raise ValueError("Nhãn chứa giá trị ngoài {-1, 1}")
    return encoded.astype(int)


def report_quality(frame: pd.DataFrame, name: str) -> None:
    """Ghi số mẫu, dữ liệu thiếu và phân bố nhãn."""
    n_missing = int(frame.isna().sum().sum())
    log.info("[%s] Số mẫu: %d, số cột: %d", name, len(frame), frame.shape[1])
    log.info("[%s] Tổng ô thiếu (NaN): %d", name, n_missing)
    if n_missing:
        missing_columns = frame.columns[frame.isna().any()].tolist()
        log.warning("[%s] Cột có giá trị thiếu: %s", name, missing_columns)
    distribution = frame[LABEL_COLUMN].value_counts().sort_index()
    log.info("[%s] Phân bố nhãn:\n%s", name, distribution.to_string())


def _validate_ratios(ratios: tuple[float, float, float]) -> np.ndarray:
    ratio_array = np.asarray(ratios, dtype=float)
    if ratio_array.shape != (3,) or np.any(ratio_array <= 0):
        raise ValueError("Ba tỷ lệ train/validation/test phải lớn hơn 0")
    if not np.isclose(ratio_array.sum(), 1.0):
        raise ValueError("Tổng tỷ lệ train/validation/test phải bằng 1")
    return ratio_array


def grouped_stratified_split(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, pd.DataFrame]:
    """
    Chia dữ liệu theo nhóm vector đặc trưng và cân bằng số hàng/lớp.

    Mỗi vector 30 đặc trưng là một nhóm không thể tách. Thứ tự nhóm được xáo
    trộn bằng seed, sau đó thuật toán tham lam gán từng nhóm vào tập làm giảm
    sai lệch chuẩn hóa so với mục tiêu về số hàng, hai lớp và số nhóm.
    """
    ratio_array = _validate_ratios(ratios)
    labels = set(frame[LABEL_COLUMN].dropna().astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(f"Nhãn nội bộ phải có đủ hai lớp {{0, 1}}, nhận được {labels}")

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    grouped = frame.groupby(FEATURE_COLUMNS, sort=True, dropna=False)
    for _, group in grouped:
        class_counts = np.bincount(
            group[LABEL_COLUMN].to_numpy(dtype=int), minlength=2
        ).astype(float)
        groups.append((group.index.to_numpy(), class_counts))

    if len(groups) < len(SPLIT_NAMES):
        raise ValueError("Không đủ nhóm đặc trưng để tạo ba tập dữ liệu")

    total_class_counts = np.bincount(
        frame[LABEL_COLUMN].to_numpy(dtype=int), minlength=2
    ).astype(float)
    totals = np.array(
        [len(frame), total_class_counts[0], total_class_counts[1], len(groups)],
        dtype=float,
    )
    targets = ratio_array[:, None] * totals[None, :]
    current = np.zeros_like(targets)
    assigned_indices: list[list[int]] = [[] for _ in SPLIT_NAMES]

    rng = np.random.default_rng(seed)
    for group_index in rng.permutation(len(groups)):
        row_indices, class_counts = groups[group_index]
        group_stats = np.array(
            [len(row_indices), class_counts[0], class_counts[1], 1],
            dtype=float,
        )

        objectives = []
        for split_index in range(len(SPLIT_NAMES)):
            candidate = current.copy()
            candidate[split_index] += group_stats
            normalized_error = (candidate - targets) / np.maximum(targets, 1.0)
            objectives.append(float(np.square(normalized_error).sum()))

        selected = int(np.argmin(objectives))
        current[selected] += group_stats
        assigned_indices[selected].extend(row_indices.tolist())

    result: dict[str, pd.DataFrame] = {}
    for split_index, split_name in enumerate(SPLIT_NAMES):
        part = frame.loc[assigned_indices[split_index]].copy()
        part = part.sample(frac=1.0, random_state=seed + split_index)
        result[split_name] = part.reset_index(drop=True)

    return result


def impute_mode(train_frame: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Điền thiếu bằng mode của tập train, không dùng validation/test."""
    output = frame.copy()
    for column in FEATURE_COLUMNS:
        if output[column].isna().any():
            mode = train_frame[column].mode(dropna=True)
            if mode.empty:
                raise ValueError(f"Không thể tính mode cho đặc trưng {column}")
            mode_value = mode.iloc[0]
            missing_count = int(output[column].isna().sum())
            output[column] = output[column].fillna(mode_value)
            log.info(
                "Điền %d ô thiếu ở '%s' bằng mode train=%s",
                missing_count,
                column,
                mode_value,
            )
    return output


def _feature_key_set(frame: pd.DataFrame) -> set[tuple]:
    return set(frame[FEATURE_COLUMNS].itertuples(index=False, name=None))


def build_split_manifest(
    full_frame: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    ratios: tuple[float, float, float],
    seed: int,
    source_file: str,
) -> dict:
    """Tạo thông tin kiểm toán cho phép tái lập và kiểm tra split."""
    feature_group_labels = full_frame.groupby(
        FEATURE_COLUMNS, sort=True, dropna=False
    )[LABEL_COLUMN].nunique()
    feature_sets = {name: _feature_key_set(part) for name, part in splits.items()}

    overlaps = {}
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            key = f"{left_name}__{right_name}"
            overlaps[key] = len(feature_sets[left_name] & feature_sets[right_name])

    split_stats = {}
    for name, ratio in zip(SPLIT_NAMES, ratios):
        part = splits[name]
        counts = part[LABEL_COLUMN].value_counts().to_dict()
        split_stats[name] = {
            "target_ratio": ratio,
            "rows": len(part),
            "actual_ratio": len(part) / len(full_frame),
            "class_counts": {
                "legitimate": int(counts.get(0, 0)),
                "phishing": int(counts.get(1, 0)),
            },
            "unique_feature_vectors": len(feature_sets[name]),
        }

    return {
        "source_file": source_file,
        "strategy": "grouped_stratified_by_30_feature_vector",
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "total_rows": len(full_frame),
        "unique_feature_vectors": len(_feature_key_set(full_frame)),
        "exact_duplicate_rows_beyond_first": (
            len(full_frame) - len(full_frame.drop_duplicates())
        ),
        "conflicting_label_feature_vectors": int((feature_group_labels > 1).sum()),
        "cross_split_feature_vector_overlap": overlaps,
        "splits": split_stats,
    }


def save_csv(frame: pd.DataFrame, output_directory: Path, name: str) -> None:
    path = output_directory / f"{name}_processed.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    log.info("Đã lưu %s (%d dòng) -> %s", name, len(frame), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiền xử lý và chia train/validation/test cho UCI Phishing Websites."
    )
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--data-file",
        type=Path,
        default=project_root / "data" / "Phishing_Dataset_full.arff",
        help="File ARFF đầy đủ dùng làm nguồn duy nhất",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "output",
        help="Thư mục lưu CSV và split_manifest.json",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full_frame = load_arff(args.data_file)
    report_quality(full_frame, "UCI full")
    full_frame[LABEL_COLUMN] = encode_label(full_frame[LABEL_COLUMN])

    splits = grouped_stratified_split(
        full_frame,
        ratios=DEFAULT_SPLIT_RATIOS,
        seed=args.split_seed,
    )

    train_reference = splits["train"]
    clean_splits = {
        name: impute_mode(train_reference, part)
        for name, part in splits.items()
    }
    for name, part in clean_splits.items():
        report_quality(part, name)
        save_csv(part, args.out_dir, name)

    manifest = build_split_manifest(
        full_frame=full_frame,
        splits=clean_splits,
        ratios=DEFAULT_SPLIT_RATIOS,
        seed=args.split_seed,
        source_file=args.data_file.name,
    )
    overlap_total = sum(manifest["cross_split_feature_vector_overlap"].values())
    if overlap_total:
        raise RuntimeError("Phát hiện vector đặc trưng xuất hiện ở nhiều tập")

    manifest_path = args.out_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Đã lưu thông tin kiểm toán split -> %s", manifest_path)


if __name__ == "__main__":
    main()
