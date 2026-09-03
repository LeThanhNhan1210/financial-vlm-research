# TÀI LIỆU BÀN GIAO TIẾN ĐỘ DỰ ÁN (PROJECT HANDOFF)

*   **Tên đề tài:** Ứng dụng mô hình ngôn ngữ thị giác (VLMs) trong phân tích biểu đồ tài chính và hỗ trợ đề xuất chiến lược
*   **Môi trường thực nghiệm mục tiêu:** Google Colab GPU T4 (16GB VRAM)
*   **Mô hình cốt lõi:** Qwen 2.5/3 VL (bản 8B) / Llama 3.2 Vision (11B) với kỹ thuật lượng hóa INT4 & QLoRA
*   **Thời điểm cập nhật:** 04/09/2026 (Phiên làm việc ban đầu - Project Initialization)

---

## 1. TỔNG QUAN CÁC CÔNG VIỆC ĐÃ HOÀN THÀNH TRONG PHIÊN LÀM VIỆC NÀY

Trong phiên làm việc này, toàn bộ nền móng của dự án nghiên cứu khoa học đã được thiết lập đầy đủ từ mặt học thuật, kiến trúc kỹ thuật đến cấu trúc mã nguồn:

### A. Hoàn thiện Kế hoạch Nghiên cứu Khoa học ([ke_hoach_trien_khai.md](file:///e:/NCKH/ke_hoach_trien_khai.md))
*   Xác định rõ **3 đóng góp khoa học chính**:
    1.  Bộ dữ liệu biểu đồ tài chính có chuỗi suy luận (CoT) theo trục thời gian (Time-series split).
    2.  Kiến trúc Pipeline lai (Hybrid Pipeline) kết hợp Vision Encoder và OCR Cross-validation chống ảo giác.
    3.  Quy trình đánh giá đa tầng 3 cấp độ (OCR Error $\rightarrow$ LLM-as-a-Judge $\rightarrow$ Directional Backtest).
*   Lập chi tiết lộ trình 5 giai đoạn (14 tuần) từ thu thập dữ liệu đến viết báo cáo bảo vệ đề tài.
*   Bổ sung ma trận quản trị rủi ro phần cứng và phương án dự phòng.

### B. Khởi tạo Cấu trúc Thư mục Chuẩn Quốc tế (Clean Architecture)
Dự án được phân tách theo tiêu chuẩn *Cookiecutter Data Science* và *Hugging Face Best Practices*:
*   📁 **`configs/`**: Tách rời toàn bộ siêu tham số:
    *   [training_config.yaml](file:///e:/NCKH/configs/training_config.yaml): Cấu hình train tối ưu cho T4 (`batch_size=1`, `gradient_accumulation_steps=16`, `gradient_checkpointing=True`).
    *   [qlora_config.yaml](file:///e:/NCKH/configs/qlora_config.yaml): Lượng hóa INT4 (`nf4`, `double_quant`), LoRA rank $r=16, \alpha=32$ áp dụng riêng cho LM backbone.
    *   [prompt_templates.yaml](file:///e:/NCKH/configs/prompt_templates.yaml): Prompt mẫu CoT chuyên sâu phân tích kỹ thuật CMT.
    *   [eval_rubric.yaml](file:///e:/NCKH/configs/eval_rubric.yaml): Tiêu chí chấm điểm LLM-as-a-Judge (Soundness, Consistency, Risk, Hallucination).
*   📁 **`data/`**: Phân tầng `raw/` (ảnh gốc), `processed/` (ảnh chuẩn hóa $\le 512\times 512$), `annotations/` (nhãn CoT), `splits/` (`train.jsonl`, `val.jsonl`, `test.jsonl`).
*   📁 **`src/`**: Mã nguồn module hóa hoàn chỉnh:
    *   `src/data/`: [dataset.py](file:///e:/NCKH/src/data/dataset.py), [preprocessor.py](file:///e:/NCKH/src/data/preprocessor.py).
    *   `src/models/`: [vlm_loader.py](file:///e:/NCKH/src/models/vlm_loader.py) (INT4 loader), [lora_setup.py](file:///e:/NCKH/src/models/lora_setup.py) (freeze vision encoder).
    *   `src/pipeline/`: [prompt_engine.py](file:///e:/NCKH/src/pipeline/prompt_engine.py), [ocr_extractor.py](file:///e:/NCKH/src/pipeline/ocr_extractor.py).
    *   `src/training/`: [trainer.py](file:///e:/NCKH/src/training/trainer.py), [callbacks.py](file:///e:/NCKH/src/training/callbacks.py) (Drive sync callback).
    *   `src/evaluation/`: [llm_judge.py](file:///e:/NCKH/src/evaluation/llm_judge.py), [ocr_metrics.py](file:///e:/NCKH/src/evaluation/ocr_metrics.py).
*   📁 **`notebooks/`**: [01_colab_workflow_demo.ipynb](file:///e:/NCKH/notebooks/01_colab_workflow_demo.ipynb) sẵn sàng mở và chạy ngay trên Colab.
*   📁 **`scripts/`**: [run_train.py](file:///e:/NCKH/scripts/run_train.py), [run_evaluate.py](file:///e:/NCKH/scripts/run_evaluate.py).
*   📁 **`checkpoints/`** & **`outputs/`**: Nơi lưu trữ adapter và log thực nghiệm (có `.gitkeep`).
*   📄 **[requirements.txt](file:///e:/NCKH/requirements.txt)** & **[.gitignore](file:///e:/NCKH/.gitignore)**: Đảm bảo không đẩy dữ liệu lớn và file rác lên Git.
*   📄 **[README.md](file:///e:/NCKH/README.md)**: Hướng dẫn chi tiết cách kết nối GitHub và chạy trên Colab.
*   📄 **[docs/QUY_CHUAN_LUU_TRU_GITHUB_DRIVE.md](file:///e:/NCKH/docs/QUY_CHUAN_LUU_TRU_GITHUB_DRIVE.md)**: Tài liệu học thuật phân tích toàn diện các tiêu chuẩn CCDS, MLOps, Hugging Face, I/O Optimization áp dụng cho GitHub & Google Drive.


### C. Thiết lập Hệ thống Kỹ năng & Quy tắc Tự động (Antigravity Customizations)
*   📄 **[AGENTS.md](file:///e:/NCKH/AGENTS.md)** (Project Rules): Ràng buộc tự động bắt buộc AI Agent luôn tuân thủ các giới hạn T4, Time-series split và Clean Architecture.
*   📁 **Bộ 4 Kỹ năng (Skills) chuyên biệt:**
    1.  **[financial-vlm-guidelines](file:///e:/NCKH/skills/financial-vlm-guidelines/SKILL.md)**: Sổ tay quy chuẩn VLM, tiền xử lý, huấn luyện QLoRA và giao thức đánh giá 3 tầng.
    2.  **[debugging-and-error-recovery](file:///e:/NCKH/skills/debugging-and-error-recovery/SKILL.md)**: Quy trình chẩn đoán lỗi có hệ thống, chuyên trị CUDA OOM, lỗi float16/bf16, tràn token thị giác và phục hồi sau timeout Colab.
    3.  **[documentation-and-adrs](file:///e:/NCKH/skills/documentation-and-adrs/SKILL.md)**: Quy chuẩn ghi chép hồ sơ nghiên cứu và quyết định kiến trúc (đã tạo sẵn [ADR-0001: Lựa chọn mô hình VLM và INT4](file:///e:/NCKH/docs/adrs/0001-chon-mo-hinh-vlm-va-luong-hoa-int4.md)).
    4.  **[git-workflow-and-versioning](file:///e:/NCKH/skills/git-workflow-and-versioning/SKILL.md)**: Quy chuẩn commit sạch (Conventional Commits), phân nhánh thực nghiệm và gắn tag phiên bản mô hình/benchmark.


---

## 2. TRẠNG THÁI HIỆN TẠI CỦA HỆ THỐNG (CURRENT SYSTEM STATE)

*   **Mã nguồn cục bộ (`e:\NCKH`):** Đã hoàn tất 100% việc chuẩn bị cấu trúc khung và code mẫu.
*   **Git trên máy:** Đang chưa có lệnh `git` trong biến môi trường PATH (cần cài Git for Windows hoặc GitHub Desktop).
*   **Dữ liệu & Drive:** Thư mục đích lưu trữ kết quả và checkpoints đã được cố định là `MyDrive/NCKH_AI` (đã tạo sẵn script tự động khởi tạo tại `scripts/setup_drive.py`).

---

## 3. CÁC BƯỚC HÀNH ĐỘNG TIẾP THEO (NEXT ACTIONABLE STEPS)

Khi tiếp tục dự án ở phiên làm việc kế tiếp, các bước cần thực hiện bao gồm:

1.  **Cài đặt Git & Đẩy mã nguồn lên GitHub:**
    *   Cài đặt Git for Windows (`winget install --id Git.Git -e --source winget` hoặc tải từ trang chủ).
    *   Tạo Repository trên GitHub (khuyến nghị để chế độ Private).
    *   Chạy lệnh đẩy code:
        ```bash
        git init
        git add .
        git commit -m "feat: initialize financial VLM project structure"
        git branch -M main
        git remote add origin <URL_REPO_GITHUB>
        git push -u origin main
        ```
2.  **Khởi tạo Thử nghiệm Baseline trên Google Colab (Giai đoạn 3.1):**
    *   Mở Google Colab, bật GPU T4.
    *   Clone repo về và chạy thử file [01_colab_workflow_demo.ipynb](file:///e:/NCKH/notebooks/01_colab_workflow_demo.ipynb) để kiểm tra việc nạp mô hình Qwen-VL 8B với INT4.
3.  **Thu thập dữ liệu và Gán nhãn (Giai đoạn 1):**
    *   Thu thập 150 - 250 ảnh biểu đồ nến (Daily/4H) của VN30, S&P500 hoặc BTC lưu vào `data/raw/`.
    *   Xây dựng script gọi GPT-4o API để sinh nhãn chuỗi suy luận mẫu (CoT) vào `data/annotations/`.
    *   Chia tập theo trục thời gian vào `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl`.
