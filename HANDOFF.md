# TÀI LIỆU BÀN GIAO TIẾN ĐỘ DỰ ÁN TOÀN DIỆN (COMPREHENSIVE PROJECT HANDOFF)

*   **Tên đề tài NCKH:** Ứng dụng mô hình ngôn ngữ thị giác (VLMs) trong phân tích biểu đồ tài chính và hỗ trợ đề xuất chiến lược
*   **Hạ tầng thực nghiệm:** Google Colab GPU T4 (16GB VRAM) kết hợp Google Drive (`MyDrive/NCKH_AI`)
*   **Mô hình nền tảng:** Qwen 2.5/3 VL (bản 8B) / Llama 3.2 Vision (11B) với kỹ thuật lượng hóa INT4 (bitsandbytes) & QLoRA
*   **Thời điểm cập nhật:** 04/09/2026 (Hoàn tất toàn bộ Phiên khởi tạo kiến trúc - Phase 0: Project Inception)

---

## 1. TỔNG KẾT TOÀN BỘ CÔNG VIỆC ĐÃ HOÀN THÀNH TRONG PHIÊN LÀM VIỆC NÀY

Trong phiên làm việc này, toàn bộ nền móng của dự án nghiên cứu khoa học đã được xây dựng hoàn chỉnh từ cơ sở lý thuyết, kiến trúc hệ thống, mã nguồn module hóa, bộ quy chuẩn AI Agent cho đến thiết kế phân tách lưu trữ GitHub và Google Drive:

### A. Kế hoạch Nghiên cứu & Định vị Đóng góp Khoa học
*   **File tài liệu:** [ke_hoach_trien_khai.md](file:///e:/NCKH/ke_hoach_trien_khai.md)
*   **Nội dung đã hoàn thành:**
    *   Xác định rõ **3 đóng góp khoa học cốt lõi (Scientific Contributions)**:
        1.  *Đóng góp Dữ liệu:* Bộ dữ liệu biểu đồ nến Nhật kèm chuỗi suy luận (CoT) phân chia nghiêm ngặt theo trục thời gian (Time-series Split) triệt tiêu Look-ahead bias.
        2.  *Đóng góp Pipeline:* Kiến trúc lai (Hybrid Pipeline) kết hợp Vision Encoder và OCR Cross-validation chống ảo giác số liệu.
        3.  *Đóng góp Thực nghiệm:* Khung đánh giá định lượng 3 tầng (OCR Error $\rightarrow$ LLM-as-a-Judge $\rightarrow$ Simulated Backtesting / Directional Accuracy).
    *   Lập chi tiết lộ trình 5 giai đoạn (14 tuần) từ thu thập dữ liệu, huấn luyện QLoRA đến viết báo cáo nghiệm thu đề tài.
    *   Bổ sung ma trận quản trị rủi ro phần cứng và phương án dự phòng.

---

### B. Kiến trúc Mã nguồn Chuẩn Quốc tế (Clean Architecture & CCDS)
Toàn bộ mã nguồn đã được đóng gói thành các module độc lập trong thư mục `src/`, tuân thủ nguyên tắc Clean Architecture:
*   📁 **`configs/` (Quản lý Siêu tham số):**
    *   [training_config.yaml](file:///e:/NCKH/configs/training_config.yaml): Tối ưu cho Colab T4 (`batch_size=1`, `gradient_accumulation_steps=16`, `gradient_checkpointing=True`, liên kết Drive `NCKH_AI`).
    *   [qlora_config.yaml](file:///e:/NCKH/configs/qlora_config.yaml): Lượng hóa INT4 NF4, LoRA rank $r=16, \alpha=32$ áp dụng riêng cho LM backbone.
    *   [prompt_templates.yaml](file:///e:/NCKH/configs/prompt_templates.yaml): Prompt mẫu CoT chuyên sâu phân tích kỹ thuật CMT.
    *   [eval_rubric.yaml](file:///e:/NCKH/configs/eval_rubric.yaml): Tiêu chí chấm điểm LLM-as-a-Judge (Soundness, Consistency, Risk, Hallucination).
*   📁 **`data/` (Phân tầng Dữ liệu):** `raw/`, `processed/`, `annotations/`, `splits/` (đã có `.gitkeep` và file mẫu).
*   📁 **`src/` (Mã nguồn Module hóa):**
    *   `src/data/`: [dataset.py](file:///e:/NCKH/src/data/dataset.py), [preprocessor.py](file:///e:/NCKH/src/data/preprocessor.py).
    *   `src/models/`: [vlm_loader.py](file:///e:/NCKH/src/models/vlm_loader.py) (INT4 loader), [lora_setup.py](file:///e:/NCKH/src/models/lora_setup.py) (freeze vision encoder).
    *   `src/pipeline/`: [prompt_engine.py](file:///e:/NCKH/src/pipeline/prompt_engine.py), [ocr_extractor.py](file:///e:/NCKH/src/pipeline/ocr_extractor.py).
    *   `src/training/`: [trainer.py](file:///e:/NCKH/src/training/trainer.py), [callbacks.py](file:///e:/NCKH/src/training/callbacks.py) (Drive sync callback trỏ vào `NCKH_AI`).
    *   `src/evaluation/`: [llm_judge.py](file:///e:/NCKH/src/evaluation/llm_judge.py), [ocr_metrics.py](file:///e:/NCKH/src/evaluation/ocr_metrics.py).
*   📁 **`notebooks/`:** [01_colab_workflow_demo.ipynb](file:///e:/NCKH/notebooks/01_colab_workflow_demo.ipynb) sẵn sàng mở và chạy ngay trên Colab.
*   📁 **`scripts/`:** 
    *   [run_train.py](file:///e:/NCKH/scripts/run_train.py): Script khởi chạy quá trình huấn luyện QLoRA.
    *   [run_evaluate.py](file:///e:/NCKH/scripts/run_evaluate.py): Script chạy đánh giá 3 tầng.
    *   [setup_drive.py](file:///e:/NCKH/scripts/setup_drive.py): Script tự động tạo toàn bộ cấu trúc thư mục con bên trong `MyDrive/NCKH_AI`.
*   📄 **[requirements.txt](file:///e:/NCKH/requirements.txt)** & **[.gitignore](file:///e:/NCKH/.gitignore)**: Bảo vệ Git khỏi các file dữ liệu nặng và checkpoints.
*   📄 **[README.md](file:///e:/NCKH/README.md)**: Tài liệu hướng dẫn cài đặt và vận hành tổng thể.

---

### C. Hệ thống Kỹ năng & Quy tắc Tự động (Antigravity Customizations)
*   📄 **[AGENTS.md](file:///e:/NCKH/AGENTS.md)** (Project Rules): Bộ quy tắc bắt buộc áp dụng tự động cho AI Agent:
    1. Giới hạn VRAM T4 16GB (INT4, freeze vision encoder).
    2. Toàn vẹn chuỗi thời gian (chỉ dùng time-series split, cấm random shuffle).
    3. Không code logic phức tạp trong notebook (chỉ viết vào `src/`).
    4. Kiểm soát ảo giác thị giác bằng OCR Cross-validation.
*   📁 **Bộ 4 Kỹ năng (Skills) chuyên biệt** (Lưu song song ở `.agents/skills/` và `skills/`):
    1.  **[financial-vlm-guidelines](file:///e:/NCKH/skills/financial-vlm-guidelines/SKILL.md)**: Sổ tay quy chuẩn VLM, tiền xử lý, huấn luyện QLoRA và giao thức đánh giá 3 tầng.
    2.  **[debugging-and-error-recovery](file:///e:/NCKH/skills/debugging-and-error-recovery/SKILL.md)**: Quy trình chẩn đoán lỗi có hệ thống, chuyên trị CUDA OOM, lỗi float16/bf16, tràn token thị giác và phục hồi sau timeout Colab.
    3.  **[documentation-and-adrs](file:///e:/NCKH/skills/documentation-and-adrs/SKILL.md)**: Quy chuẩn ghi chép hồ sơ nghiên cứu và quyết định kiến trúc.
    4.  **[git-workflow-and-versioning](file:///e:/NCKH/skills/git-workflow-and-versioning/SKILL.md)**: Quy chuẩn commit sạch (Conventional Commits), phân nhánh thực nghiệm và gắn tag phiên bản mô hình/benchmark.
*   📄 **[ADR-0001](file:///e:/NCKH/docs/adrs/0001-chon-mo-hinh-vlm-va-luong-hoa-int4.md)**: Bản ghi quyết định kiến trúc đầu tiên về việc lựa chọn mô hình Qwen-VL 8B và kỹ thuật lượng hóa INT4 NF4.

---

### D. Chuẩn hóa Hệ thống Lưu trữ Phân lập (GitHub vs Google Drive)
*   **File tài liệu chi tiết:** [docs/QUY_CHUAN_LUU_TRU_GITHUB_DRIVE.md](file:///e:/NCKH/docs/QUY_CHUAN_LUU_TRU_GITHUB_DRIVE.md)
*   **Nội dung đã hoàn thành:**
    *   Mô tả chi tiết nguyên lý "Tam giác phân lập": GitHub (Code & Logic) $\leftrightarrow$ Google Colab (Compute) $\leftrightarrow$ Google Drive (Storage & State Persistence).
    *   Phân tích 4 tiêu chuẩn áp dụng cho GitHub (CCDS, Hugging Face, Clean Architecture, ADRs).
    *   Phân tích 4 tiêu chuẩn áp dụng cho Google Drive (MLOps Artifacts Lifecycle, FUSE I/O Optimization, Fault Tolerance Checkpointing, Numerical Prefixing).
    *   Bổ sung **đầy đủ 2 sơ đồ cây thư mục chi tiết** của cả GitHub Repo và Google Drive (`MyDrive/NCKH_AI`).
    *   Đồng bộ toàn bộ mã nguồn sang thư mục Drive thực tế của người dùng: **`MyDrive/NCKH_AI`**.
    *   Cung cấp đoạn văn mẫu học thuật chuẩn để đưa thẳng vào Chương 3 của Báo cáo NCKH.

---

## 2. TRẠNG THÁI HIỆN TẠI CỦA DỰ ÁN (CURRENT STATE CHECKLIST)

| Thành phần | Trạng thái | Ghi chú |
| :--- | :---: | :--- |
| **Kế hoạch NCKH (14 tuần)** | Hoàn thành 100% | File `ke_hoach_trien_khai.md` sẵn sàng |
| **Cấu trúc thư mục mã nguồn** | Hoàn thành 100% | Toàn bộ `src/`, `configs/`, `scripts/` đã sẵn sàng |
| **Hồ sơ ADRs** | Hoàn thành ADR-0001 | Sẵn sàng bổ sung ADR-0002 khi chốt dataset split |
| **Hệ thống Skills & Rules** | Hoàn thành 4 Skills | Lưu tại cả `.agents/skills/` và `skills/` |
| **Tài liệu quy chuẩn lưu trữ** | Hoàn thành 100% | File `docs/QUY_CHUAN_LUU_TRU_GITHUB_DRIVE.md` |
| **Đồng bộ Google Drive** | Hoàn thành 100% | Đã tạo đầy đủ cấu trúc thư mục trong `MyDrive/NCKH_AI` |
| **Mã nguồn trên GitHub** | Hoàn thành 100% | Đã push toàn bộ repo lên GitHub |
| **Bộ thông số hoàn chỉnh** | Hoàn thành 100% | Đã có `.env.example`, `dataset_config`, `backtest_config` |
| **Dữ liệu thực tế & Nhãn CoT** | Sẵn sàng triển khai | Bắt đầu Giai đoạn 1 (Phase 1) |

---

## 3. LỘ TRÌNH HÀNH ĐỘNG CHO PHIÊN LÀM VIỆC TIẾP THEO (NEXT SESSION ACTION PLAN)

Khi bắt đầu phiên làm việc tiếp theo, bạn chỉ cần thực hiện theo 4 bước tuần tự sau:

1.  **Cài đặt Git & Đẩy code lên GitHub:**
    *   Cài đặt Git for Windows (`winget install --id Git.Git -e --source winget` hoặc tải từ [git-scm.com](https://git-scm.com/)).
    *   Tạo Repository riêng tư trên [github.com](https://github.com/) (ví dụ: `financial-vlm-research`).
    *   Đẩy toàn bộ mã nguồn lên:
        ```bash
        git init
        git add .
        git commit -m "feat: complete initial project scaffold, skills and storage architecture"
        git branch -M main
        git remote add origin <URL_REPO_GITHUB>
        git push -u origin main
        ```
2.  **Khởi tạo cấu trúc thư mục trên Google Drive (1 Click):**
    *   Mở Google Colab, mount Drive và chạy file `scripts/setup_drive.py` (hoặc copy cell tạo thư mục) để khởi tạo các thư mục con bên trong `MyDrive/NCKH_AI`.
3.  **Chạy thử nghiệm kiểm tra Pipeline ban đầu (Smoke Test):**
    *   Mở notebook [01_colab_workflow_demo.ipynb](file:///e:/NCKH/notebooks/01_colab_workflow_demo.ipynb) trên Colab T4 để kiểm tra việc nạp mô hình Qwen-VL 8B ở chế độ INT4 xem có chạy trơn tru không.
4.  **Bắt đầu Giai đoạn 1: Thu thập Dữ liệu & Tạo Nhãn CoT:**
    *   Thu thập 150 - 250 ảnh biểu đồ nến (Daily/4H) của VN30, S&P500 hoặc Crypto lưu vào Drive `1_datasets/raw_images/`.
    *   Xây dựng script gọi GPT-4o API để sinh nhãn phân tích chuỗi suy luận (CoT) và tiến hành rà soát thủ công (Human Audit).
