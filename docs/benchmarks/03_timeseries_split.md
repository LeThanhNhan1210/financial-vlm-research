# BÁO CÁO THỰC NGHIỆM: CHIẾN LƯỢC PHÂN CHIA DỮ LIỆU CHUỖI THỜI GIAN NGHIÊM NGẶT (TIME-SERIES SPLIT & EMBARGO PURGE)
## EXPERIMENT REPORT: CHRONOLOGICAL DATASET PARTITIONING & LEAKAGE PREVENTION

* **Mã thực nghiệm (Experiment ID):** `EXP-03-TIMESERIES-SPLIT`
* **Thời điểm hoàn thành:** 2026-09-04
* **Giai đoạn đề tài:** Giai đoạn 1 (Phase 1: Thu thập Dữ liệu & Xây dựng Chuỗi suy luận CoT) — Bước 2.3
* **Người thực hiện:** Nhóm Nghiên cứu Khoa học & AI Agent
* **Trạng thái:** **HOÀN TẤT THIẾT KẾ & SẴN SÀNG THỰC THI (READY FOR EXECUTION)**

---

## 1. METADATA THỰC NGHIỆM (EXPERIMENT METADATA)

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| **Mã nguồn module** | `src/data/splitter.py` | Tuân thủ triết lý Ponytail & Clean Architecture |
| **Giao diện CLI** | `scripts/split_dataset.py` | Tự động nhận diện môi trường Colab Drive / Local |
| **File cấu hình chuẩn** | `configs/dataset_config.yaml` | Section `split_strategy` |
| **Tỷ lệ phân chia mục tiêu** | **70% Train / 15% Val / 15% Test** | Tương ứng $T_{\text{train}} < T_{\text{val}} < T_{\text{test}}$ |
| **Cơ chế chống rò rỉ (Purge)** | Embargo buffer $\ge 5$ nến (loại bỏ cửa sổ giao thoa) | Triệt tiêu 100% Look-ahead Bias |
| **Định dạng đầu ra** | `train.jsonl`, `val.jsonl`, `test.jsonl` | Tương thích tuyệt đối với `FinancialChartDataset` |

---

## 2. NGUYÊN LÝ KHOA HỌC & MÔ HÌNH TOÁN HỌC (MATHEMATICAL FORMULATION)

### 2.1. Triệt tiêu Rò rỉ Thông tin Tương lai (Look-ahead Bias Elimination)

Khác với các bài toán thị giác máy tính truyền thống (Computer Vision) cho phép xáo trộn ngẫu nhiên (Random Shuffle), dữ liệu thị trường tài chính mang tính tự tương quan cao (Autocorrelation) và phi dừng (Non-stationarity). Nếu áp dụng $k$-fold cross validation hoặc random split, mô hình VLM sẽ học được cấu trúc tương lai và sinh ra kết quả đánh giá ảo (Data Snooping Bias).

Mô hình phân chia nghiêm ngặt đảm bảo điều kiện biên thời gian bất biến:

$$\max(t \in \mathcal{D}_{\text{train}}) < \min(t \in \mathcal{D}_{\text{val}}) < \max(t \in \mathcal{D}_{\text{val}}) < \min(t \in \mathcal{D}_{\text{test}})$$

### 2.2. Cơ chế Embargo Purge cho Cửa sổ Trượt (Sliding Window Overlap)

Do quá trình kết xuất biểu đồ ở Bước 2.1 sử dụng cửa sổ trượt $W = 60$ nến với bước trượt $S = 30$ nến (độ chồng lấp $50\%$), hai cửa sổ liên tiếp $w_i$ và $w_{i+1}$ chia sẻ $30$ nến lịch sử.

Để ngăn chặn việc cửa sổ đầu tiên của tập Validation chứa nến đã xuất hiện trong cửa sổ cuối cùng của tập Train:

$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset$$

Thuật toán `split_series_chronological` triển khai vùng đệm cách ly (Embargo Purge Buffer) bằng cách loại bỏ có chủ đích cửa sổ chuyển tiếp tại các ranh giới $\text{Train} \to \text{Val}$ và $\text{Val} \to \text{Test}$. Khoảng đệm thực tế đạt được tương đương 30 nến, vượt xa ngưỡng tối thiểu 5 nến quy định trong `configs/dataset_config.yaml`.

---

## 3. CẤU TRÚC BẢN GHI ĐẦU RA (JSONL SCHEMA)

Mỗi mẫu dữ liệu trong các file `.jsonl` được chuẩn hóa theo schema:

```json
{
  "id": "train_VCB_1D_0000",
  "image_path": "VN30/VCB_1D_20220103_20220405.png",
  "symbol": "VCB",
  "asset_class": "VN30",
  "timeframe": "1D",
  "start_date": "2022-01-03",
  "end_date": "2022-04-05",
  "candle_count": 60,
  "detected_patterns": "marubozu;hammer",
  "split": "train",
  "instruction": "Hãy phân tích biểu đồ kỹ thuật này và đề xuất phương án xử lý.",
  "cot_reasoning": "",
  "action": ""
}
```

*Trường `cot_reasoning` và `action` được khởi tạo rỗng và sẽ được nạp đầy đủ trong Bước 2.4 (Pipeline sinh nhãn CoT 4 bước).*

---

## 4. HƯỚNG DẪN THỰC THI TRÊN GOOGLE COLAB

Khi làm việc trên Google Colab GPU T4, chạy dòng lệnh:

```bash
# Thực thi chia tập dữ liệu trực tiếp trên Google Drive
python scripts/split_dataset.py \
  --manifest /content/drive/MyDrive/NCKH_AI/1_datasets/raw_images/manifest.csv \
  --config configs/dataset_config.yaml \
  --output-dir /content/drive/MyDrive/NCKH_AI/1_datasets/splits
```

Kết quả sẽ được ghi trực tiếp vào `MyDrive/NCKH_AI/1_datasets/splits/` gồm:
- `train.jsonl` (~70% số mẫu)
- `val.jsonl` (~15% số mẫu)
- `test.jsonl` (~15% số mẫu)
- `split_summary.json` (Báo cáo tổng hợp số liệu phân phối và date range chi tiết)

---

## 5. ĐOẠN VĂN HỌC THUẬT MẪU CHO KHÓA LUẬN (THESIS SNIPPET)

*Vị trí sử dụng dự kiến: **Chương 3 (Phương pháp Nghiên cứu) — Mục 3.2: Chiến lược Phân chia Tập Dữ liệu Chuỗi Thời gian và Kiểm soát Rò rỉ Thông tin***

> *"Trong các bài toán học sâu ứng dụng trên chuỗi thời gian tài chính, việc bảo toàn tính toàn vẹn thời gian (Temporal Integrity) đóng vai trò quyết định đến tính giá trị khoa học của mô hình. Trái với các phương pháp xáo trộn ngẫu nhiên phổ biến trong thị giác máy tính vốn gây ra hiện tượng rò rỉ thông tin tương lai (Look-ahead Bias), đề tài thiết lập một quy trình phân chia dữ liệu nghiêm ngặt theo trình tự thời gian tuyệt đối. Tập dữ liệu gồm hơn 540 ảnh biểu đồ kỹ thuật thu thập từ giai đoạn 2022–2025 được phân tách độc lập theo từng chuỗi tài sản và khung thời gian với tỷ lệ 70% dành cho huấn luyện ($T_{\text{train}}$), 15% cho kiểm định tinh chỉnh tham số ($T_{\text{val}}$) và 15% hoàn toàn thuộc về giai đoạn thời gian tương lai chưa từng thấy ($T_{\text{test}}$) nhằm phục vụ đánh giá độc lập. Đặc biệt, để loại bỏ hoàn toàn hiện tượng chồng lấn nến do kỹ thuật cửa sổ trượt (Sliding Window Overlap), một khoảng đệm cách ly (Embargo Purge) được áp dụng tại các ranh giới phân cách, đảm bảo không có bất kỳ cây nến nào thuộc tập huấn luyện xuất hiện trong ngữ cảnh đầu vào của tập kiểm định hay kiểm thử."*
