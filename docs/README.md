# Nhom3 — Phát hiện website phishing bằng Perceptron

Đề tài môn Học máy (cao học, ĐH Duy Tân). Triển khai Perceptron **từ đầu**
(theo giáo trình TS. Đặng Việt Hùng, mục 2.2) để phân loại website là
**phishing** hay **legitimate** dựa trên 30 đặc trưng của bộ dữ liệu
Phishing Websites (UCI).

---

## Cấu trúc dự án

```
Nhom3_Phishing_Perceptron/
├── data/
│   ├── Training.arff               # 6157 mẫu (tách từ dataset gốc UCI)
│   ├── Testing.arff                # 4898 mẫu
│   └── Phishing Websites Features.docx  # tài liệu định nghĩa 30 đặc trưng
├── src/
│   ├── preprocess.py               # GĐ1: ARFF -> DataFrame, kiểm tra chất lượng
│   ├── perceptron.py               # GĐ2: Perceptron from scratch (giáo trình 2.2)
│   ├── train_eval.py               # GĐ2: huấn luyện + đánh giá misclassification
│   ├── feature_extractor.py        # GĐ3: trích 30 đặc trưng từ URL thật (+ PhishTank)
│   ├── detect_service.py           # Application Service: điều phối + explainability
│   ├── demo.py                     # CLI mỏng gọi DetectService
│   ├── app.py                      # Web MVP (Flask): /, /api/check, /dashboard
│   ├── templates/                  # index.html, dashboard.html
│   └── static/                     # style.css, app.js
├── output/                         # dữ liệu đã xử lý, trọng số, kết quả đánh giá
└── docs/README.md
```

## Chạy Web MVP (Flask)

```bash
python3 src/app.py
# mở http://127.0.0.1:5000/ — nhập domain để kiểm tra
# GET  /dashboard  — dashboard stub (nội dung bổ sung sau)
```

API:
```bash
curl -X POST http://127.0.0.1:5000/api/check \
     -H "Content-Type: application/json" \
     -d '{"url":"duytan.edu.vn"}'
```

## PhishTank (đặc trưng Statistical_report)

`Statistical_report` tra cứu URL trong danh sách đen PhishTank (API miễn phí,
cần đăng ký `app_key` tại https://www.phishtank.com/developer_info.php).

```bash
export PHISHTANK_API_KEY="<app_key của bạn>"
python3 src/app.py
```

Nếu chưa có key, đặc trưng này trả giá trị trung gian `0` và được đánh dấu
trong danh sách "approximated" — MVP vẫn hoạt động bình thường.

## Cách chạy

```bash
# 1. Tiền xử lý (ARFF -> CSV sạch)
python3 src/preprocess.py

# 2. Huấn luyện + đánh giá Perceptron
python3 src/train_eval.py

# 3. Demo: kiểm tra 1 URL bất kỳ
python3 src/demo.py https://duytan.edu.vn/
```

## Kết quả đánh giá (GĐ2)

Perceptron from scratch, `alpha=0.1`, `max_epochs=100`, seed 42, trên tập test
4898 mẫu:

| Chỉ số | Giá trị |
|---|---|
| Accuracy | 89.06% |
| **Misclassification rate** | **10.94%** |
| Precision (phishing) | 89.37% |
| Recall (phishing) | 91.26% |
| F1 | 90.30% |

Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):

| | legitimate | phishing |
|---|---|---|
| **legitimate** | 1866 | 297 |
| **phishing** | 239 | 2496 |

> Baseline "luôn đoán phishing" chỉ đạt 55.84% accuracy — Perceptron vượt trội.
> Model không hội tụ tuyệt đối (~600 cập nhật/epoch còn lại) vì dữ liệu phishing
> không tuyến tính tách được; trọng số dao động quanh biên quyết định — đúng
> hành vi dự kiến của Perceptron với dữ liệu không separable.

## Ghi chú GĐ3 — trích đặc trưng từ URL thật

- 22/30 đặc trưng được tính xác định từ URL/HTML/DNS/SSL.
- 8 đặc trưng cần dịch vụ ngoài (WHOIS, PageRank, Google Index, trang thống kê)
  trả giá trị trung gian `0` và được đánh dấu trong trường `approximated` để
  minh bạch: `Domain_registeration_length`, `Abnormal_URL`, `age_of_domain`,
  `web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`,
  `Statistical_report`.
- `SSLfinal_State` có thể bị đánh sai khi môi trường dùng proxy chèn chứng chỉ
  không tin cậy (vd mạng công ty) — cần chạy trên mạng thật để chuẩn.