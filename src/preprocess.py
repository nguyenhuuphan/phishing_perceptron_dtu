#!/usr/bin/env python3
"""
preprocess.py — Giai đoạn 1 (GĐ1): Tiền xử lý dữ liệu Phishing Websites.

Đọc 2 file ARFF (Training.arff / Testing.arff) của bộ dữ liệu Phishing
Websites (UCI, 30 đặc trưng + nhãn Result), chuyển thành DataFrame,
kiểm tra chất lượng dữ liệu (thiếu giá trị, phân bố nhãn) và lưu ra
thư mục output/ dưới dạng CSV để các giai đoạn sau (huấn luyện Perceptron,
đánh giá, demo) sử dụng.

Giá trị đặc trưng nằm trong tập {-1, 0, 1} theo đúng mô tả dataset;
nhãn Result được mã hoá lại thành {0, 1} để thuận tiện cho phân loại nhị phân.
"""

import argparse
import logging
import os
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

# Backward-compatible name used by the preprocessing pipeline.
FEATURE_COLUMNS = FEATURE_NAMES

LABEL_COLUMN = "Result"


def load_arff(path: Path) -> pd.DataFrame:
    """Đọc file ARFF và trả về DataFrame với 30 đặc trưng + nhãn (kiểu int)."""
    from scipy.io import arff

    log.info("Đang đọc %s ...", path)
    data, meta = arff.loadarff(str(path))
    df = pd.DataFrame(data)

    # scipy trả về bytes cho các thuộc tính nominal -> giải mã sang int
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, bytes) else v
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    expected = FEATURE_COLUMNS + [LABEL_COLUMN]
    if list(df.columns) != expected:
        raise ValueError(f"Cột không khớp mong đợi: {list(df.columns)}")
    validate_feature_values(df)
    raw_labels = set(df[LABEL_COLUMN].dropna().unique())
    if not raw_labels <= set(RAW_LABEL_TO_INTERNAL):
        raise ValueError(f"Nhãn chứa giá trị ngoài {{-1, 1}}: {sorted(raw_labels)}")
    return df


def encode_label(series: pd.Series) -> pd.Series:
    """Mã hoá nhãn {-1, 1} -> {0, 1} để dùng cho phân loại nhị phân."""
    encoded = series.map(RAW_LABEL_TO_INTERNAL)
    assert encoded.notna().all(), "Nhãn chứa giá trị ngoài {-1, 1}"
    return encoded.astype(int)


def report_quality(df: pd.DataFrame, name: str) -> None:
    """In báo cáo chất lượng dữ liệu: thiếu giá trị + phân bố nhãn."""
    n_missing = int(df.isna().sum().sum())
    log.info("[%s] Số mẫu: %d, số cột: %d", name, len(df), df.shape[1])
    log.info("[%s] Tổng ô thiếu giá trị (NaN): %d", name, n_missing)
    if n_missing > 0:
        miss_cols = df.columns[df.isna().any()].tolist()
        log.warning("[%s] Cột có giá trị thiếu: %s", name, miss_cols)
    dist = df[LABEL_COLUMN].value_counts().sort_index()
    log.info("[%s] Phân bố nhãn:\n%s", name, dist.to_string())


def impute_mode(train_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Điền giá trị thiếu bằng mode (giá trị xuất hiện nhiều nhất) tính từ
    tập huấn luyện — tránh rò rỉ thông tin từ tập kiểm tra.
    """
    out = df.copy()
    for col in FEATURE_COLUMNS:
        if out[col].isna().any():
            mode_val = train_df[col].mode().iloc[0]
            n = int(out[col].isna().sum())
            out[col] = out[col].fillna(mode_val)
            log.info("Điền %d ô thiếu ở '%s' bằng mode=%s", n, col, mode_val)
    return out


def save_csv(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info("Đã lưu %s (%d dòng) -> %s", name, len(df), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu Phishing Websites.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Thư mục chứa Training.arff / Testing.arff",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output",
        help="Thư mục lưu dữ liệu đã xử lý",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_raw = load_arff(args.data_dir / "Training.arff")
    test_raw = load_arff(args.data_dir / "Testing.arff")

    report_quality(train_raw, "Training")
    report_quality(test_raw, "Testing")

    # Mã hoá nhãn trước khi điền thiếu (nhãn không được thiếu)
    train_raw[LABEL_COLUMN] = encode_label(train_raw[LABEL_COLUMN])
    test_raw[LABEL_COLUMN] = encode_label(test_raw[LABEL_COLUMN])

    # Điền thiếu bằng mode từ tập huấn luyện
    train_clean = impute_mode(train_raw, train_raw)
    test_clean = impute_mode(train_raw, test_raw)

    save_csv(train_clean, args.out_dir, "train_processed")
    save_csv(test_clean, args.out_dir, "test_processed")

    log.info("Hoàn tất tiền xử lý. Dữ liệu sạch nằm trong: %s", args.out_dir)


if __name__ == "__main__":
    main()
