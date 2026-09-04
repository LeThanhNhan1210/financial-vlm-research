# TÀI LIỆU BÀN GIAO TIẾN ĐỘ DỰ ÁN TOÀN DIỆN (COMPREHENSIVE PROJECT HANDOFF)

*   **Tên đề tài NCKH:** Ứng dụng mô hình ngôn ngữ thị giác (VLMs) trong phân tích biểu đồ tài chính và hỗ trợ đề xuất chiến lược
*   **Hạ tầng thực nghiệm:** Google Colab GPU T4 (16GB VRAM) kết hợp Google Drive (`MyDrive/NCKH_AI`)
*   **Mô hình nền tảng:** Qwen 2.5-VL (bản 7B/8B Instruct) với lượng hóa INT4 (bitsandbytes NF4) & QLoRA
*   **Thời điểm cập nhật:** 04/09/2026 (Hoàn tất 100% Phase 0: Khởi tạo Kiến trúc & Kiểm chứng Thực nghiệm Phần cứng)

---

## 1. TỔNG KẾT TIẾN ĐỘ ĐÃ HOÀN THÀNH (CẬP NHẬT MỚI NHẤT)

Dự án đã vượt qua toàn bộ các mốc thẩm định ban đầu và chính thức chuyển giao sang **Giai đoạn 1 (Phase 1: Thu thập Dữ liệu & Tạo nhãn CoT)**:

### A. Kiến trúc Nền tảng & Phân tách Lưu trữ (Three-pillar Architecture)
*   **Mã nguồn GitHub:** Tổ chức chuẩn Clean Architecture và Cookiecutter Data Science trong [src/](file:///e:/financial-vlm-research/src) (`data`, `models`, `pipeline`, `training`, `evaluation`).
*   **Google Drive (`MyDrive/NCKH_AI`):** Khởi tạo thành công 100% cấu trúc thư mục lưu trữ (`1_datasets`, `2_checkpoints`, `3_experiment_outputs`, `4_backups`) qua script [setup_drive.py](file:///e:/financial-vlm-research/scripts/setup_drive.py).
*   **Tam giác phân lập:** GitHub (Code) $\leftrightarrow$ Google Colab (Compute) $\leftrightarrow$ Google Drive (Data & Persistence) đã kết nối thông suốt.

### B. Kiểm chứng Thực nghiệm Phần cứng & VRAM trên Colab T4 (ĐÃ ĐẠT 100%)
*   **Báo cáo thực nghiệm chi tiết:** [docs/benchmarks/00_hardware_vram_smoke_test.md](file:///e:/financial-vlm-research/docs/benchmarks/00_hardware_vram_smoke_test.md)
*   **Số liệu đo đạc thực tế:**
    *   Tổng số tham số: **$8,339,756,032$** (~8.34 Tỷ tham số của Qwen2.5-VL).
    *   Số tham số tinh chỉnh (LoRA trainable): **$47,589,376$** (tỉ lệ **$0.5706\%$**).
    *   VRAM thực tế chiếm dụng khi nạp: **$7.74\text{ GB} / 15.00\text{ GB}$** ($51.6\%$).
    *   VRAM dự phòng an toàn (Headroom): **$5.22\text{ GB}$** (đảm bảo an toàn tuyệt đối cho backward pass).
    *   Hiệu quả tối ưu: Tiết kiệm **$53.93\%$ VRAM** so với baseline FP16 ($16.8\text{ GB}$).
    *   Xác nhận: Vision Encoder đã được đóng băng hoàn toàn 100%, chỉ fine-tune Language Model backbone.

### C. Kiểm chứng Suy luận & Phát hiện Khoa học Ban đầu (Zero-shot Inference Test)
*   **Báo cáo thực nghiệm chi tiết:** [docs/benchmarks/01_inference_smoke_test.md](file:///e:/financial-vlm-research/docs/benchmarks/01_inference_smoke_test.md)
*   **Kết quả:** Mô hình sinh văn bản tiếng Việt mượt mà trên GPU T4.
*   **Phát hiện khoa học cốt lõi:** Mô hình nền tảng (Zero-shot) gặp lỗi **ảo giác cấu trúc (Structural Hallucination)** nghiêm trọng khi mô tả nến Búa (nhầm lẫn giữa thân nến và bóng nến dưới). Đây là bằng chứng vàng chứng minh tính cấp thiết của đề tài trong việc xây dựng chuỗi suy luận CoT và fine-tuning chuyên sâu.

### D. Hệ thống Quy chuẩn AI Agent & Kỹ năng Nghiên cứu Khoa học
*   **Bộ 5 Kỹ năng (Skills):**
    1.  [financial-vlm-guidelines](file:///e:/financial-vlm-research/skills/financial-vlm-guidelines/SKILL.md): Quy chuẩn VLM, tiền xử lý, huấn luyện QLoRA và đánh giá 3 tầng.
    2.  [scientific-research-logging](file:///e:/financial-vlm-research/skills/scientific-research-logging/SKILL.md): **(Mới)** Giao thức 5 bước bắt buộc về ghi chép hồ sơ thực nghiệm định lượng, cập nhật ADR và trích xuất đoạn văn học thuật (Thesis Mapping).
    3.  [debugging-and-error-recovery](file:///e:/financial-vlm-research/skills/debugging-and-error-recovery/SKILL.md): Chẩn đoán lỗi VRAM OOM, lượng hóa INT4, thư viện Hugging Face.
    4.  [documentation-and-adrs](file:///e:/financial-vlm-research/skills/documentation-and-adrs/SKILL.md): Quản lý Architecture Decision Records (đã có ADR-0001).
    5.  [git-workflow-and-versioning](file:///e:/financial-vlm-research/skills/git-workflow-and-versioning/SKILL.md): Quy chuẩn Conventional Commits và phân nhánh thực nghiệm.
*   **Quy tắc dự án ([AGENTS.md](file:///e:/financial-vlm-research/AGENTS.md)):** Bổ sung Điều 5 bắt buộc số hóa mọi số liệu thực nghiệm thành file Markdown, không để trôi số liệu trên terminal.
*   **Cẩm nang thu thập dữ liệu ([docs/HUONG_DAN_THU_THAP_DU_LIEU_CHINH_THONG.md](file:///e:/financial-vlm-research/docs/HUONG_DAN_THU_THAP_DU_LIEU_CHINH_THONG.md)):** **(Mới)** Hệ thống hóa các nguồn dữ liệu chính thống (`vnstock3`, `yfinance`, `ccxt`, TradingView) và phương pháp thu thập lai Human-in-the-Loop.

---

## 2. BẢNG TRẠNG THÁI CÁC GIAI ĐOẠN ĐỀ TÀI (PHASE TRACKER)

| Giai đoạn | Nội dung công việc | Trạng thái | Minh chứng học thuật |
| :--- | :--- | :---: | :--- |
| **Phase 0** | Khởi tạo kiến trúc, nạp mô hình INT4, Smoke test VRAM | **HOÀN THÀNH 100%** | `docs/benchmarks/00_hardware_vram_smoke_test.md` |
| **Phase 1** | Thu thập 160-220 ảnh biểu đồ nến & Tạo nhãn CoT (HITL) | **ĐANG TIẾN HÀNH** (Hoàn thành 100% Code Bước 2.1) | `docs/adrs/0002-pipeline-thu-thap-du-lieu-va-ve-bieu-do.md` |
| **Phase 2** | Huấn luyện QLoRA trên Colab T4 & Đồng bộ Checkpoints Drive | Chuẩn bị | `configs/training_config.yaml`, `qlora_config.yaml` |
| **Phase 3** | Đánh giá định lượng 3 tầng (OCR, LLM Judge, Backtest) | Chuẩn bị | `configs/eval_rubric.yaml`, `backtest_config.yaml` |
| **Phase 4** | Nghiệm thu học thuật, xuất báo cáo & slide bảo vệ đề tài | Chuẩn bị | `docs/adrs/`, `docs/benchmarks/`, `ke_hoach_trien_khai.md` |

---

## 3. LỘ TRÌNH HÀNH ĐỘNG TIẾP THEO (NEXT STEPS CHO PHASE 1)

1.  **Đẩy mã nguồn cập nhật lên GitHub từ máy Local:**
    ```bash
    git add .
    git commit -m "feat(data): implement chart generator pipeline and 19 candlestick patterns (ADR-0002)"
    git push origin main
    ```
2.  **Khởi chạy kết xuất ảnh trên Google Colab (Đồng bộ Drive):**
    Trên notebook Google Colab (đã mount Google Drive):
    ```bash
    !pip install -r requirements.txt
    !python scripts/generate_charts.py --config configs/dataset_config.yaml --output-dir /content/drive/MyDrive/NCKH_AI/1_datasets/raw_images/
    ```
3.  **Bước 2.2 (Thủ công / Tác giả rà soát HITL):** Tác giả kiểm tra thư mục Drive, lọc bỏ nến nhiễu và bổ sung các mẫu hình đặc thù từ TradingView nếu cần.
4.  **Bước 2.3 (Khóa tập mẫu):** Triển khai script Time-series Split ($70\%$ Train, $15\%$ Val, $15\%$ Test có Purge 5 bars) dựa trên file `manifest.csv`.
5.  **Bước 2.4 (Gán nhãn CoT):** Khởi chạy pipeline sinh nhãn CoT 4 bước kết hợp rà soát chuyên gia tối thiểu $30\%$.

