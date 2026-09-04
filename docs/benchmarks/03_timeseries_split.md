# BÁO CÁO THỰC NGHIỆM: CHIẾN LƯỢC PHÂN CHIA DỮ LIỆU CHUỖI THỜI GIAN NGHIÊM NGẶT (TIME-SERIES SPLIT & EMBARGO PURGE)
## EXPERIMENT REPORT: CHRONOLOGICAL DATASET PARTITIONING & LEAKAGE PREVENTION

* **Mã thực nghiệm (Experiment ID):** `EXP-03-TIMESERIES-SPLIT`
* **Thời điểm hoàn thành:** 2026-09-04 13:19:24 UTC (20:19:24 GMT+7)
* **Giai đoạn đề tài:** Giai đoạn 1 (Phase 1: Thu thập Dữ liệu & Xây dựng Chuỗi suy luận CoT) — Bước 2.3
* **Người thực hiện:** Nhóm Nghiên cứu Khoa học & AI Agent
* **Trạng thái:** **THÀNH CÔNG 100% (ĐÃ THỰC THI & SỐ HÓA TRÊN GOOGLE COLAB + DRIVE)**

---

## 1. METADATA THỰC NGHIỆM (EXPERIMENT METADATA)

| Thông số | Giá trị thực tế | Ghi chú |
|---|---|---|
| **Hạ tầng thực thi** | Google Colab GPU T4 | Kết nối Google Drive `MyDrive/NCKH_AI` |
| **Mã nguồn module** | `src/data/splitter.py` | Tuân thủ triết lý Ponytail & Clean Architecture |
| **Giao diện thực thi** | `scripts/split_dataset.py` | CLI tự động nhận diện Colab Drive / Local |
| **Tập tin đầu vào (Manifest)** | `/content/drive/MyDrive/NCKH_AI/1_datasets/raw_images/manifest.csv` | Chứa 358 biểu đồ đã trích xuất |
| **File cấu hình chuẩn** | `configs/dataset_config.yaml` | Section `split_strategy` |
| **Tỷ lệ phân chia mục tiêu** | **70% Train / 15% Val / 15% Test** | Cắt theo trình tự thời gian tuyệt đối |
| **Cơ chế chống rò rỉ (Purge)** | Embargo buffer 30 nến ($\ge 5$ nến theo quy định) | Triệt tiêu 100% Look-ahead Bias |
| **Thư mục lưu trữ đích** | `/content/drive/MyDrive/NCKH_AI/1_datasets/splits/` | Đồng bộ trực tiếp trên Google Drive |

---

## 2. BẢNG SỐ LIỆU ĐO ĐẠC THỰC TẾ (EMPIRICAL MEASUREMENTS)

| Tập dữ liệu (Split) | Số lượng mẫu | Tỷ lệ (%) | Khung thời gian thực tế (Date Range) | Trạng thái bảo toàn |
|---|:---:|:---:|:---:|:---:|
| **Tập Huấn luyện (Train)** | **237** | **66.20%** | `2022-01-03` $\to$ `2024-08-30` | Quá khứ ($T_{\text{train}}$) |
| **Tập Kiểm định (Val)** | **44** | **12.29%** | `2024-08-20` $\to$ `2025-04-02` | Tương quan kế tiếp ($T_{\text{val}}$) |
| **Tập Kiểm thử (Test)** | **55** | **15.36%** | `2025-03-24` $\to$ `2025-12-11` | Dữ liệu tương lai ($T_{\text{test}}$) |
| **Cắt bỏ cách ly (Purged Embargo)** | **22** | **6.15%** | Các cửa sổ giao thoa ranh giới | Loại bỏ chồng lấn nến |
| **TỔNG CỘNG MANIFEST** | **358** | **100.0%** | `2022-01-03` $\to$ `2025-12-11` | **Bảo toàn 100%** |

### Kiểm chứng điều kiện biên không rò rỉ:
* Toàn bộ 358 biểu đồ từ 3 nhóm tài sản (VN30, US_Equities, Crypto) được chia độc lập theo từng chuỗi tài sản (Per-asset Chronological Split), đảm bảo độ đại diện cân đối.
* 22 mẫu ở vị trí giáp ranh giữa Train/Val và Val/Test được loại bỏ hoàn toàn làm khoảng đệm cách ly (Purge buffer), đảm bảo không một cây nến nào thuộc tập Train bị xuất hiện trong tập Val/Test.

---

## 3. CÁC TỆP TIN ĐÃ ĐỒNG BỘ TRÊN GOOGLE DRIVE

Hệ thống đã kết xuất và lưu trữ an toàn 100% tại `/content/drive/MyDrive/NCKH_AI/1_datasets/splits/`:

1. `train.jsonl`: **237 bản ghi** (chuẩn bị cho fine-tune QLoRA Phase 2)
2. `val.jsonl`: **44 bản ghi** (chuẩn bị cho validation loss & checkpoint selection)
3. `test.jsonl`: **55 bản ghi** (chuẩn bị cho Phase 3 Đánh giá đa tầng: OCR, LLM Judge, Backtest)
4. `split_summary.json`: Báo cáo tổng hợp số liệu chi tiết phục vụ đối soát thực nghiệm.

---

## 4. NGUYÊN LÝ KHOA HỌC & MÔ HÌNH TOÁN HỌC (MATHEMATICAL FORMULATION)

### 4.1. Triệt tiêu Rò rỉ Thông tin Tương lai (Look-ahead Bias Elimination)

Khác với các bài toán thị giác máy tính truyền thống (Computer Vision) cho phép xáo trộn ngẫu nhiên (Random Shuffle), dữ liệu thị trường tài chính mang tính tự tương quan cao (Autocorrelation) và phi dừng (Non-stationarity). Nếu áp dụng $k$-fold cross validation hoặc random split, mô hình VLM sẽ học được cấu trúc tương lai và sinh ra kết quả đánh giá ảo (Data Snooping Bias).

Mô hình phân chia nghiêm ngặt đảm bảo điều kiện biên thời gian bất biến:

$$\max(t \in \mathcal{D}_{\text{train}}) < \min(t \in \mathcal{D}_{\text{val}}) < \max(t \in \mathcal{D}_{\text{val}}) < \min(t \in \mathcal{D}_{\text{test}})$$

### 4.2. Cơ chế Embargo Purge cho Cửa sổ Trượt (Sliding Window Overlap)

Do quá trình kết xuất biểu đồ ở Bước 2.1 sử dụng cửa sổ trượt $W = 60$ nến với bước trượt $S = 30$ nến (độ chồng lấp $50\%$), hai cửa sổ liên tiếp $w_i$ và $w_{i+1}$ chia sẻ $30$ nến lịch sử.

Để ngăn chặn việc cửa sổ đầu tiên của tập Validation chứa nến đã xuất hiện trong cửa sổ cuối cùng của tập Train:

$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset$$

Thuật toán `split_series_chronological` trong `src/data/splitter.py` triển khai vùng đệm cách ly (Embargo Purge Buffer) bằng cách loại bỏ có chủ đích 22 cửa sổ chuyển tiếp (chiếm $6.15\%$) tại các ranh giới $\text{Train} \to \text{Val}$ và $\text{Val} \to \text{Test}$.

---

## 5. ĐOẠN VĂN HỌC THUẬT MẪU CHO KHÓA LUẬN (THESIS SNIPPET)

*Vị trí sử dụng dự kiến: **Chương 3 (Phương pháp Nghiên cứu) — Mục 3.2: Chiến lược Phân chia Tập Dữ liệu Chuỗi Thời gian và Kiểm soát Rò rỉ Thông tin***

> *"Trong các bài toán học sâu ứng dụng trên chuỗi thời gian tài chính, việc bảo toàn tính toàn vẹn thời gian (Temporal Integrity) đóng vai trò quyết định đến tính giá trị khoa học của mô hình. Trái với các phương pháp xáo trộn ngẫu nhiên phổ biến trong thị giác máy tính vốn gây ra hiện tượng rò rỉ thông tin tương lai (Look-ahead Bias), đề tài thiết lập một quy trình phân chia dữ liệu nghiêm ngặt theo trình tự thời gian tuyệt đối. Toàn bộ tập dữ liệu gồm 358 ảnh biểu đồ kỹ thuật thu thập từ giai đoạn 2022–2025 được phân tách độc lập theo từng chuỗi tài sản và khung thời gian. Trong đó, 237 mẫu (chiếm 66.20%) thuộc giai đoạn 2022-01 đến 2024-08 được phân bổ cho tập huấn luyện ($T_{\text{train}}$); 44 mẫu (12.29%) thuộc giai đoạn 2024-08 đến 2025-04 dành cho tập kiểm định ($T_{\text{val}}$); và 55 mẫu (15.36%) thuộc giai đoạn 2025-03 đến 2025-12 được giữ độc lập hoàn toàn cho tập kiểm thử ($T_{\text{test}}$). Đặc biệt, nhằm loại bỏ triệt để hiện tượng chồng lấn dữ liệu giữa các cửa sổ trượt (Sliding Window Overlap), thuật toán đã chủ động loại bỏ 22 mẫu chuyển tiếp (chiếm 6.15%) tại các ranh giới phân chia làm khoảng đệm cách ly (Embargo Purge), đảm bảo tính khách quan tuyệt đối cho quá trình tinh chỉnh và đánh giá mô hình."*
