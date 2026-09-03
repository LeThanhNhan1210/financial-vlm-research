# QUY CHUẨN THIẾT KẾ CẤU TRÚC LƯU TRỮ DỰ ÁN AI
## HỆ THỐNG PHÂN TÁCH GITHUB & GOOGLE DRIVE CHO ĐỀ TÀI NCKH VLM

*   **Tên đề tài:** Ứng dụng mô hình ngôn ngữ thị giác (VLMs) trong phân tích biểu đồ tài chính và hỗ trợ đề xuất chiến lược
*   **Môi trường thực nghiệm:** Google Colab GPU T4 (16GB VRAM)
*   **Mô hình nền tảng:** Qwen 2.5/3 VL (8B) / Llama 3.2 Vision (11B) + INT4 / QLoRA
*   **Ngày ban hành:** 04/09/2026

---

## 1. TỔNG QUAN NGUYÊN LÝ THIẾT KẾ CỐT LÕI (CORE ARCHITECTURAL PRINCIPLES)

Dự án áp dụng mô hình kiến trúc **"Tam giác phân lập" (Three-pillar AI System)** nhằm tối ưu hóa chi phí phần cứng miễn phí trên đám mây, đảm bảo tính bất biến của dữ liệu và phục vụ nghiệm thu học thuật:

```text
┌────────────────────────────────────────────────────────┐
│               1. GITHUB (Bộ Não Kỹ Thuật)              │
│  • Quản lý logic mã nguồn, thuật toán, cấu hình, docs  │
│  • Nguyên tắc: RẤT NHẸ (Code-only), Kiểm soát phiên bản│
└──────────────────────────┬─────────────────────────────┘
                           │ git clone
                           ▼
┌────────────────────────────────────────────────────────┐
│               2. GOOGLE COLAB (Động Cơ Tính Toán)      │
│  • Cung cấp GPU T4 (16GB VRAM) & RAM tính toán         │
│  • Nguyên tắc: Stateless (Không lưu trữ vĩnh viễn)     │
└──────────────────────────┬─────────────────────────────┘
                           │ drive.mount() (Đọc & Ghi liên tục)
                           ▼
┌────────────────────────────────────────────────────────┐
│               3. GOOGLE DRIVE (Kho Lưu Trữ Bất Biến)   │
│  • Lưu dataset ảnh nặng, checkpoints mô hình, kết quả  │
│  • Nguyên tắc: State Persistence (Bảo toàn trạng thái) │
└────────────────────────────────────────────────────────┘
```

---

## 2. CÁC TIÊU CHUẨN ÁP DỤNG CHO THƯ MỤC GITHUB

Repository GitHub được xây dựng dựa trên **4 tiêu chuẩn công nghiệp và kỹ nghệ phần mềm quốc tế**:

### A. Tiêu chuẩn Cookiecutter Data Science (CCDS)
*   **Nguyên tắc:** Phân tách rạch ròi giữa dữ liệu thô, mã nguồn xử lý, cấu hình và báo cáo. Dữ liệu gốc (`data/raw/`) là bất biến, không bao giờ được sửa đổi trực tiếp bởi code.
*   **Áp dụng:** 
    *   Tách riêng `data/`, `src/`, `configs/`, `notebooks/`, `scripts/`.
    *   `.gitignore` chặn 100% việc vô tình đẩy dữ liệu nặng lên Git.

### B. Tiêu chuẩn Hugging Face & PEFT Best Practices
*   **Nguyên tắc:** Tách rời cấu hình siêu tham số ra khỏi code thực thi; chỉ quản lý adapter weights thay vì lưu cả mô hình nền tảng hàng chục GB.
*   **Áp dụng:**
    *   Toàn bộ tham số huấn luyện (learning rate, batch size, LoRA rank $r$, $\alpha$) được quản lý độc lập tại `configs/training_config.yaml` và `configs/qlora_config.yaml`.
    *   Chỉ theo dõi code gán LoRA adapter trong `src/models/lora_setup.py`.

### C. Nguyên lý Clean Architecture & Phân tách trách nhiệm (Single Responsibility)
*   **Nguyên tắc:** Không bao giờ viết logic xử lý dài hàng trăm dòng trực tiếp trong Jupyter Notebook.
*   **Áp dụng trong `src/`:**
    *   `src/data/`: Chỉ chịu trách nhiệm nạp dữ liệu và tiền xử lý ảnh.
    *   `src/models/`: Chỉ chịu trách nhiệm nạp mô hình INT4 và thiết lập LoRA.
    *   `src/pipeline/`: Module lai chống ảo giác (OCR trích xuất + Vision reasoning).
    *   `src/training/`: Training loop và callback đồng bộ Drive.
    *   `src/evaluation/`: Bộ đo lường 3 tầng (OCR error, LLM-as-a-Judge, Backtest).
    *   File Notebook tại `notebooks/` chỉ đóng vai trò bộ điều khiển (Controller) gọi hàm và vẽ biểu đồ.

### D. Tiêu chuẩn Tài liệu hóa Quyết định Kiến trúc (ADRs) & Conventional Commits
*   **Nguyên tắc:** Mọi thay đổi kiến trúc và bước đi kỹ thuật đều phải có luận cứ khoa học để hội đồng thẩm định đánh giá.
*   **Áp dụng:**
    *   Thư mục `docs/adrs/` lưu trữ các bản ghi quyết định học thuật (như `ADR-0001: Lựa chọn mô hình VLM và INT4`).
    *   Quy chuẩn commit sạch (`feat:`, `fix:`, `exp:`, `data:`, `docs:`) theo dõi tiến độ khoa học.

---

## 3. CÁC TIÊU CHUẨN ÁP DỤNG CHO THƯ MỤC GOOGLE DRIVE

Thư mục gốc lưu trữ thực nghiệm được cố định tại: **`MyDrive/NCKH_AI/`** (đường dẫn truy cập trên Colab: `/content/drive/MyDrive/NCKH_AI/`).
Google Drive đóng vai trò là tầng lưu trữ bền vững (State Persistence Layer), được thiết kế dựa trên **4 cơ sở kỹ thuật**:

### A. Chuẩn Vòng đời Hiện vật MLOps (MLflow / WandB / DVC Artifacts Standard)
Hệ thống lưu trữ trên Drive mô phỏng đúng 3 nhóm hiện vật tiêu chuẩn trong MLOps:
1.  **Input Artifacts (`1_datasets/`):** Lưu trữ toàn bộ ảnh biểu đồ và nhãn CoT phục vụ nạp vào mô hình.
2.  **Model Artifacts (`2_checkpoints/`):** Lưu trữ các file trọng số `adapter_model.safetensors` sau mỗi chu kỳ 50 steps và trọng số tốt nhất (`best_model_adapter/`).
3.  **Metric & Evaluation Artifacts (`3_experiment_outputs/`):** Lưu trữ kết quả dự đoán (`predictions/`), bảng điểm định lượng (`evaluations/`) và hình vẽ (`figures/`).

### B. Tối ưu hóa "Nút thắt cổ chai" I/O của Google Colab (FUSE File System Optimization)
*   **Hiện tượng:** Google Drive mount vào Colab qua giao thức FUSE, có tốc độ đọc ngẫu nhiên (Random Access) rất chậm nếu duyệt qua hàng ngàn file ảnh nhỏ lẻ, gây nghẽn GPU (GPU Starvation).
*   **Giải pháp thiết kế:**
    *   Phân chia `raw_images/` (ảnh gốc) và `processed_images/` (ảnh đã chuẩn hóa $\le 512$px).
    *   Bổ sung thư mục `dataset_archives/` chứa file nén `.zip`: Khi chạy Colab, code chỉ cần copy 1 file `.zip` sang ổ đĩa nội bộ `/content/` rồi giải nén tại chỗ, giúp tăng tốc độ đọc dữ liệu lên **gấp 10 lần**.

### C. Cơ chế Chống mất mát dữ liệu do Timeout (Fault Tolerance & Checkpointing)
*   Google Colab miễn phí thường bị ngắt phiên sau vài giờ hoặc khi mất kết nối mạng.
*   Cấu trúc thư mục `2_checkpoints/qlora_exp.../` kết hợp với `DriveSyncCallback` đảm bảo luôn lưu lại:
    *   Trọng số adapter LoRA (khoảng 60-80 MB/checkpoint).
    *   File `trainer_state.json` lưu chính xác step và learning rate tại thời điểm ngắt kết nối.
    $\rightarrow$ Mở Colab mới chỉ cần bật `resume_from_checkpoint=True` là tiếp tục ngay mà không mất công train lại.

### D. Quy chuẩn Đánh số thứ tự (Numerical Prefixing Pattern)
*   Các thư mục cấp 1 trên Drive được đặt tên có tiền tố: `1_datasets`, `2_checkpoints`, `3_experiment_outputs`, `4_backups`.
*   Giúp giao diện Drive trên trình duyệt web luôn tự động sắp xếp theo đúng **tiến trình thời gian thực hiện nghiên cứu**, không bị xáo trộn lung tung theo bảng chữ cái.

---

## 4. MA TRẬN PHÂN BỔ HIỆN VẬT GIỮA GITHUB VÀ GOOGLE DRIVE

| Hạng mục hiện vật | Vị trí lưu trữ chính | Định dạng file | Lý do kỹ thuật / Học thuật |
| :--- | :---: | :---: | :--- |
| **Mã nguồn, thuật toán** | **GitHub** (`src/`, `scripts/`) | `.py` | Cần kiểm soát phiên bản Git, dung lượng nhẹ. |
| **Siêu tham số, Prompt CoT**| **GitHub** (`configs/`) | `.yaml` | Dễ dàng theo dõi lịch sử thay đổi tham số. |
| **Tài liệu học thuật, ADRs** | **GitHub** (`docs/adrs/`, `skills/`) | `.md` | Dùng làm cơ sở viết báo cáo đề tài. |
| **File phân chia mẫu (Splits)**| **GitHub** (`data/splits/`) | `.jsonl` | File text nhẹ (vài KB), minh chứng tính công bằng của Time-series split. |
| **Ảnh biểu đồ tài chính** | **Google Drive** (`1_datasets/`) | `.png`, `.jpg`, `.zip` | Dung lượng lớn (hàng trăm MB - GB), GitHub cấm lưu. |
| **Checkpoints LoRA** | **Google Drive** (`2_checkpoints/`) | `.safetensors`, `.json` | Cần bảo toàn khi Colab ngắt kết nối đột ngột. |
| **Bảng điểm số, Metrics** | **Google Drive** (`3_experiment_outputs/evaluations/`)| `.csv`, `.json` | Số liệu thô phục vụ vẽ đồ thị cho luận văn. |
| **Hình vẽ Loss Curve, Biểu đồ**| **Google Drive** (`3_experiment_outputs/figures/`)| `.png` (DPI 300) | Ảnh xuất chuẩn chất lượng cao để chèn vào báo cáo Word/LaTeX. |

---

## 5. Ý NGHĨA KHOA HỌC KHI VIẾT BÁO CÁO NGHIỆM THU NCKH

Khi báo cáo trước Hội đồng Khoa học, bạn có thể tự tin trình bày tại **Chương 3 (Phương pháp Nghiên cứu & Thiết kế Thực nghiệm)** như sau:

> *"Để giải quyết bài toán giới hạn phần cứng trên Google Colab T4 mà vẫn đảm bảo tính toàn vẹn dữ liệu, nghiên cứu áp dụng nguyên lý phân tách hiện vật theo tiêu chuẩn MLOps: Hệ thống mã nguồn và cấu hình siêu tham số được kiểm soát phiên bản độc lập trên GitHub theo chuẩn Clean Architecture và Cookiecutter Data Science; trong khi toàn bộ dữ liệu thị giác lớn và các checkpoint tối ưu LoRA được bảo toàn liên tục trên Google Drive thông qua cơ chế tự động đồng bộ Fault-tolerant Checkpointing, giúp triệt tiêu hoàn toàn rủi ro gián đoạn thực nghiệm và rò rỉ dữ liệu."*
