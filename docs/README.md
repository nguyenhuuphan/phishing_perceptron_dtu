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
│   ├── Phishing_Dataset_full.arff  # 11055 mẫu UCI, nguồn duy nhất để chia dữ liệu
│   └── Phishing Websites Features.docx  # tài liệu định nghĩa 30 đặc trưng
├── src/
│   ├── preprocess.py               # GĐ1: kiểm tra + group split train/validation/test
│   ├── perceptron.py               # GĐ2: Perceptron from scratch (giáo trình 2.2)
│   ├── train_eval.py               # GĐ2: huấn luyện + đánh giá misclassification
│   ├── feature_extractor.py        # GĐ3: trích 30 đặc trưng từ URL thật (+ PhishTank)
│   ├── detect_service.py           # Application Service: điều phối + explainability
│   ├── demo.py                     # CLI mỏng gọi DetectService
│   ├── app.py                      # Web MVP (Flask): /, /api/check, /dashboard
│   ├── templates/                  # index.html, dashboard.html
│   └── static/                     # style.css, app.js
├── output/                         # 3 split, manifest, trọng số và kết quả đánh giá
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

Nguồn dữ liệu là toàn bộ 11.055 mẫu trong `Phishing_Dataset_full.arff`. Split
được tạo lại với seed 42 theo tỷ lệ 70/15/15. Mọi hàng có cùng vector 30 đặc
trưng được giữ trong cùng một tập, vì vậy số vector trùng giữa train,
validation và test bằng 0:

| Tập | Số mẫu | Legitimate | Phishing | Vector đặc trưng duy nhất |
|---|---:|---:|---:|---:|
| Train | 7.742 | 4.312 | 3.430 | 4.045 |
| Validation | 1.655 | 922 | 733 | 870 |
| Test | 1.658 | 923 | 735 | 870 |

Dữ liệu có 5.785 vector đặc trưng duy nhất, 5.206 hàng trùng hoàn toàn sau lần
xuất hiện đầu tiên và 64 vector mang nhãn mâu thuẫn. Các trường hợp mâu thuẫn
được giữ nguyên để không tự ý sửa ground truth, nhưng toàn bộ nhóm vẫn chỉ nằm
trong một split.

Perceptron from scratch dùng `alpha=0.1`, `max_epochs=100`, seed 42. Nhãn UCI
được ánh xạ `-1` (phishing) → `1` và `+1` (legitimate) → `0` trong ứng
dụng:

| Chỉ số | Validation | Test |
|---|---:|---:|
| Accuracy | 80,30% | 82,51% |
| **Misclassification rate** | **19,70%** | **17,49%** |
| Precision (phishing) | 69,97% | 72,77% |
| Recall (phishing) | 97,27% | 96,73% |
| F1 | 81,39% | 83,06% |

Ma trận nhầm lẫn trên test (hàng = thực tế, cột = dự đoán):

| | legitimate | phishing |
|---|---:|---:|
| **legitimate** | 657 | 266 |
| **phishing** | 24 | 711 |

Baseline luôn đoán legitimate đạt 55,67% accuracy trên test. Validation hiện
mới được báo cáo độc lập; bước tiếp theo sẽ dùng validation để chọn trạng thái
model/epoch trước khi đánh giá test cuối cùng.

## Ghi chú GĐ3 — trích đặc trưng từ URL thật

- 22/30 đặc trưng được tính xác định từ URL/HTML/DNS/SSL.
- 8 đặc trưng cần dịch vụ ngoài (WHOIS, PageRank, Google Index, trang thống kê)
  trả giá trị trung gian `0` và được đánh dấu trong trường `approximated` để
  minh bạch: `Domain_registeration_length`, `Abnormal_URL`, `age_of_domain`,
  `web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`,
  `Statistical_report`.
- `SSLfinal_State` có thể bị đánh sai khi môi trường dùng proxy chèn chứng chỉ
  không tin cậy (vd mạng công ty) — cần chạy trên mạng thật để chuẩn.
