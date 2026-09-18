#!/usr/bin/env python3
"""
detect_service.py — Lớp Application Service.

Điều phối toàn bộ luồng suy luận cho một URL:
    trích 30 đặc trưng → build vector → predict → confidence → explainability
và trả về một dict JSON chuẩn để web layer (app.py) hoặc CLI (demo.py) dùng.

Tách hẳn khỏi web layer để tái dùng được: web, CLI, batch, API sau này.
"""

import logging
from pathlib import Path

import numpy as np

from feature_extractor import FEATURE_ORDER, FeatureExtractor
from perceptron import Perceptron

log = logging.getLogger(__name__)

CLASS_NAMES = {0: "LEGITIMATE", 1: "PHISHING"}
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "output" / "weights.npy"


def load_model(weights_path: Path = WEIGHTS_PATH) -> Perceptron:
    """Nạp trọng số đã huấn luyện vào một Perceptron (chưa fit)."""
    W = np.load(weights_path)
    clf = Perceptron(n_classes=W.shape[0])
    clf.W = W
    clf.n_features = W.shape[1] - 1
    return clf


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class DetectService:
    """Service suy luận phishing cho một URL."""

    def __init__(self, model: Perceptron | None = None,
                 weights_path: Path = WEIGHTS_PATH,
                 phishtank_key: str | None = None,
                 timeout: float = 15.0):
        self.model = model or load_model(weights_path)
        self.extractor = FeatureExtractor(timeout=timeout,
                                          phishtank_key=phishtank_key)

    def _explain(self, vec: np.ndarray) -> list[dict]:
        """
        Tính đóng góp từng đặc trưng vào margin (phishing − legitimate):
            contribution_k = (W[1,k] − W[0,k]) * x_k
        Trả về top đặc trưng có |đóng góp| lớn nhất, kèm hướng đẩy.
        """
        diff = self.model.W[1, :-1] - self.model.W[0, :-1]  # (30,)
        contrib = diff * vec.flatten()
        ranked = sorted(
            range(len(FEATURE_ORDER)),
            key=lambda i: abs(contrib[i]),
            reverse=True,
        )
        top = []
        for i in ranked[:5]:
            c = float(contrib[i])
            if abs(c) < 1e-9:
                continue
            top.append({
                "feature": FEATURE_ORDER[i],
                "value": int(vec[0, i]),
                "contribution": round(c, 4),
                "direction": "phishing" if c > 0 else "legitimate",
            })
        return top

    def analyze_url(self, url: str) -> dict:
        """Trích đặc trưng, dự đoán và trả về kết quả dạng dict JSON."""
        res = self.extractor.extract(url)
        vec = np.array([res.features[n] for n in FEATURE_ORDER],
                       dtype=float).reshape(1, -1)

        pred = int(self.model.predict(vec)[0])
        scores = self.model.decision_function(vec)[0]
        prob = softmax(scores)
        margin = float(abs(scores[0] - scores[1]))

        n_total = len(FEATURE_ORDER)
        n_missing = len(res.approximated)
        n_computed = n_total - n_missing

        return {
            "url": res.url,
            "status_code": res.status_code,
            "verdict": CLASS_NAMES[pred],
            "verdict_code": pred,
            "probability": {
                "legitimate": round(float(prob[0]), 6),
                "phishing": round(float(prob[1]), 6),
            },
            "margin": round(margin, 4),
            "features": {n: res.features[n] for n in FEATURE_ORDER},
            "features_count": {
                "total": n_total,
                "computed": n_computed,
                "missing": n_missing,
            },
            "approximated": res.approximated,
            "top_contributors": self._explain(vec),
        }


# Singleton cho web app (nạp model 1 lần lúc khởi động)
_service: DetectService | None = None


def get_service(**kwargs) -> DetectService:
    global _service
    if _service is None:
        _service = DetectService(**kwargs)
    return _service


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "https://duytan.edu.vn/"
    svc = DetectService()
    result = svc.analyze_url(target)
    print(f"URL      : {result['url']}")
    print(f"Verdict  : {result['verdict']}")
    print(f"Prob     : legit={result['probability']['legitimate']:.4f} "
          f"phish={result['probability']['phishing']:.4f}")
    print(f"Margin   : {result['margin']}")
    print(f"Features : {result['features_count']['computed']}/"
          f"{result['features_count']['total']} "
          f"(thiếu {result['features_count']['missing']})")
    print("Top đóng góp:")
    for t in result["top_contributors"]:
        print(f"  {t['feature']:<28} {t['contribution']:+.4f} -> {t['direction']}")