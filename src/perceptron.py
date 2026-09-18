#!/usr/bin/env python3
"""
perceptron.py — Giai đoạn 2 (GĐ2): Cài đặt Perceptron từ đầu (from scratch).

Triển khai đúng thuật toán Perceptron trong giáo trình "Học máy" (TS. Đặng
Việt Hùng, mục 2.2):

  - K bộ trọng số w_j (mỗi lớp một vector).
  - Điểm số của mẫu x với lớp j:  y_j = w_j^T [x ; 1]   (bias ghép vào cuối).
  - Dự đoán lớp = argmax_j y_j.
  - Luật cập nhật khi dự đoán sai (lớp đúng c, lớp dự đoán t):
        w_c <- w_c + alpha * [x ; 1]
        w_t <- w_t - alpha * [x ; 1]
  - Dừng khi hết epoch hoặc không còn cập nhật nào trong một lượt duyệt.

Bộ dữ liệu Phishing là phân loại nhị phân (2 lớp: hợp lệ / lừa đảo) nên K = 2.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Perceptron:
    """
    Perceptron đa lớp dạng argmax theo giáo trình (mục 2.2).

    Attributes:
        n_classes (int): số lớp K.
        alpha (float): tốc độ học (learning rate).
        max_epochs (int): số lượt duyệt dữ liệu tối đa.
        random_state (int): seed cho khởi tạo trọng số.
        W (np.ndarray | None): ma trận trọng số kích thước (K, n_features+1),
            hàng j là w_j, cột cuối là bias. Khởi tạo khi gọi fit.
    """

    n_classes: int = 2
    alpha: float = 0.1
    max_epochs: int = 100
    random_state: int = 42
    W: np.ndarray | None = field(default=None, init=False, repr=False)
    n_features: int = field(default=0, init=False)
    history: list = field(default_factory=list, init=False, repr=False)

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Ghép cột 1 vào cuối mỗi mẫu: [x ; 1]."""
        ones = np.ones((X.shape[0], 1))
        return np.hstack([X, ones])

    def _init_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        # K hàng, mỗi hàng n_features+1 (gồm bias). Khởi tạo nhỏ ngẫu nhiên.
        self.W = rng.normal(scale=0.01, size=(self.n_classes, n_features + 1))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch_callback: Callable[[int, int, "Perceptron"], None] | None = None,
    ) -> "Perceptron":
        """
        Huấn luyện Perceptron.

        Args:
            X: ma trận đặc trưng (n_samples, n_features), giá trị {-1,0,1}.
            y: nhãn (n_samples,), giá trị trong [0, n_classes).
            epoch_callback: hàm được gọi sau mỗi epoch với
                (epoch, số cập nhật, model), dùng để đánh giá checkpoint.

        Returns:
            self.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X phải là ma trận 2 chiều (n_samples, n_features)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("Số nhãn phải bằng số mẫu")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"Nhãn phải nằm trong [0, {self.n_classes})")

        self.history.clear()
        self.n_features = X.shape[1]
        self._init_weights(self.n_features)
        Xb = self._add_bias(X)  # (n, d+1)

        for epoch in range(1, self.max_epochs + 1):
            n_updates = 0
            for xi, yi in zip(Xb, y):
                scores = self.W @ xi          # y_j = w_j^T [x;1] với mọi j
                pred = int(np.argmax(scores))  # lớp dự đoán = argmax
                if pred != yi:
                    # w_c += alpha*[x;1] ; w_t -= alpha*[x;1]
                    self.W[yi] += self.alpha * xi
                    self.W[pred] -= self.alpha * xi
                    n_updates += 1
            self.history.append(n_updates)
            log.debug("epoch %d: %d cập nhật", epoch, n_updates)
            if epoch_callback is not None:
                epoch_callback(epoch, n_updates, self)
            if n_updates == 0:
                log.info("Hội tụ tại epoch %d (không còn cập nhật)", epoch)
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn cho mỗi mẫu: argmax_j w_j^T [x;1]."""
        if self.W is None:
            raise RuntimeError("Chưa huấn luyện — hãy gọi fit() trước.")
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        scores = Xb @ self.W.T          # (n, K)
        return np.argmax(scores, axis=1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Trả về điểm số (margin) cho từng lớp, kích thước (n, K)."""
        if self.W is None:
            raise RuntimeError("Chưa huấn luyện — hãy gọi fit() trước.")
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        return Xb @ self.W.T
