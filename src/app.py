#!/usr/bin/env python3
"""
app.py — Web MVP (Flask) cho engine phát hiện phishing bằng Perceptron.

Routes:
    GET  /            → form nhập domain cần check
    POST /api/check   → trả JSON verdict (gọi DetectService)
    GET  /dashboard   → dashboard stub (nội dung bổ sung sau)

Chạy:  python3 src/app.py   rồi mở http://127.0.0.1:5000/
"""

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from detect_service import DetectService

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=BASE_DIR / "templates",
            static_folder=BASE_DIR / "static")

# Nạp model + service một lần lúc khởi động (singleton)
_service = DetectService(phishtank_key=os.environ.get("PHISHTANK_API_KEY"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or request.form
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Thiếu tham số 'url'"}), 400

    try:
        result = _service.analyze_url(url)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Lỗi khi phân tích %s", url)
        return jsonify({"error": f"Lỗi khi phân tích URL: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    # debug=True chỉ dùng khi phát triển; khi demo thật nên tắt
    app.run(host="127.0.0.1", port=5000, debug=True)