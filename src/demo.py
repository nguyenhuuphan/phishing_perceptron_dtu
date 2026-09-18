#!/usr/bin/env python3
"""
demo.py — CLI mỏng gọi DetectService để kiểm tra một URL.

Cách dùng:
    python3 demo.py https://duytan.edu.vn/
    python3 demo.py --url https://example.com --weights ../output/weights.npy

Toàn bộ logic suy luận nằm trong detect_service.py (Application Service);
file này chỉ là giao diện dòng lệnh.
"""

import argparse
import json
import logging
from pathlib import Path

from detect_service import DetectService

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo phát hiện phishing bằng Perceptron.")
    parser.add_argument("url", nargs="?", help="URL cần kiểm tra")
    parser.add_argument("--weights", type=Path,
                        default=Path(__file__).resolve().parent.parent / "output" / "weights.npy")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--json", action="store_true", help="Xuất kết quả dạng JSON")
    args = parser.parse_args()

    if not args.url:
        parser.error("Cần truyền URL, vd: python3 demo.py https://duytan.edu.vn/")

    svc = DetectService(weights_path=args.weights, timeout=args.timeout)
    r = svc.analyze_url(args.url)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 64)
    print(f"URL kiểm tra : {r['url']}")
    print(f"HTTP status  : {r['status_code']}")
    print("=" * 64)
    print(f"PHÁN ĐOÁN    : {r['verdict']}")
    print(f"  Xác suất legitimate : {r['probability']['legitimate']*100:.2f}%")
    print(f"  Xác suất phishing    : {r['probability']['phishing']*100:.2f}%")
    print(f"  Biên tin cậy (margin): {r['margin']}")
    print(f"  Số đặc trưng trích   : {r['features_count']['computed']}/"
          f"{r['features_count']['total']} (thiếu {r['features_count']['missing']})")
    print("-" * 64)
    print("Top đặc trưng đóng góp:")
    for t in r["top_contributors"]:
        print(f"  {t['feature']:<28} {t['contribution']:+.4f} -> {t['direction']}")
    if r["approximated"]:
        print("-" * 64)
        print("Đặc trưng dùng giá trị gần đúng (0):", ", ".join(r["approximated"]))
    print("=" * 64)


if __name__ == "__main__":
    main()