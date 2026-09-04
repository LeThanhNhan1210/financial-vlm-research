---
name: scientific-research-logging
description: >-
  Quy chuẩn bắt buộc về ghi chép hồ sơ thực nghiệm định lượng, lưu trữ số liệu đo đạc phần cứng,
  cập nhật ADRs và trích xuất đoạn văn học thuật (Thesis Mapping) cho đề tài NCKH VLM. Kích hoạt kỹ năng này
  sau mỗi lần chạy thử nghiệm (smoke test), nạp mô hình, huấn luyện QLoRA, hoặc đánh giá benchmark.
---

# QUY CHUẨN GHI CHÉP HỒ SƠ THỰC NGHIỆM ĐỀ TÀI NGHIÊN CỨU KHOA HỌC (NCKH)

Kỹ năng này định nghĩa quy trình chuẩn hóa bắt buộc nhằm đảm bảo **tính minh bạch (Transparency), khả năng tái lập (Reproducibility)** và cung cấp trực tiếp tài liệu minh chứng cho Luận văn / Báo cáo đề tài NCKH.

---

## 1. TRIẾT LÝ CỐT LÕI (SCIENTIFIC RIGOR & PAPER TRAIL)

> **Nguyên tắc bất di bất dịch:** *"Không một thực nghiệm nào được coi là hoàn tất nếu số liệu của nó chưa được chuẩn hóa thành văn bản khoa học trong thư mục `docs/`."*

Mỗi khi người dùng chạy thành công một bước thử nghiệm trên Colab/GPU (Smoke test, Training step, Inference benchmark, Evaluation):
1. **Tuyệt đối không bỏ trôi số liệu** trên màn hình terminal hay notebook output.
2. **Ngay lập tức số hóa và lưu trữ** vào thư mục chuẩn hóa tương ứng (`docs/benchmarks/`, `docs/experiments/`, `docs/adrs/`).
3. **Luôn cung cấp kèm Đoạn văn học thuật mẫu** (Thesis Snippet) để người dùng có thể sao chép trực tiếp vào Chương 3 (Phương pháp) hoặc Chương 4 (Thực nghiệm & Thảo luận) của Báo cáo đề tài.

---

## 2. GIAO THỨC GHI CHÉP 5 BƯỚC BẮT BUỘC (THE 5-POINT LOGGING PROTOCOL)

Khi ghi nhận bất kỳ kết quả thực nghiệm nào, AI Agent và nhóm nghiên cứu phải tạo file báo cáo tuân thủ đầy đủ 5 phần sau:

### Điểm 1: Metadata Thực nghiệm (Experiment Metadata)
* **Mã thực nghiệm (ID):** Đặt theo quy ước (ví dụ: `EXP-00-SMOKE-TEST`, `EXP-01-BASELINE-QLORA`, `EXP-02-FULL-COT`).
* **Thời điểm đo:** Ngày tháng năm (`YYYY-MM-DD`).
* **Hạ tầng tính toán:** Loại GPU, VRAM khả dụng (ví dụ: `NVIDIA Tesla T4 16GB VRAM`).
* **Phiên bản phần mềm:** Bản phân phối OS, Python, PyTorch, Transformers, BitsAndBytes, PEFT.
* **Mô hình nền tảng:** Định danh Hugging Face chính xác (ví dụ: `Qwen/Qwen2.5-VL-7B-Instruct`).

### Điểm 2: Bảng Số liệu Đo đạc Thực tế (Empirical Measurements)
Phải có bảng Markdown rõ ràng với các chỉ số đo đạc cụ thể:
* **Quy mô mô hình:** Tổng tham số ($N_{total}$), số tham số LoRA ($N_{trainable}$), tỉ lệ $\%_{trainable}$.
* **Bộ nhớ GPU:** VRAM Allocated (thực dùng), VRAM Reserved (dự phòng), VRAM Headroom (khoảng trống an toàn).
* **Hiệu năng huấn luyện (nếu train):** Batch size, Gradient accumulation steps, Time/step, Throughput (samples/s), Loss ban đầu vs Loss hội tụ.
* **Chỉ số đánh giá (nếu test):** OCR Error (MAE, MAPE), LLM-as-a-Judge Score (Rubric 1-5), Simulated Win-rate %, Max Drawdown.

### Điểm 3: So sánh Định lượng với Baseline (Baseline Comparison)
Phải có công thức hoặc bảng so sánh chứng minh tính vượt trội:
* So sánh với phương án mặc định (Full fine-tune FP16 hoặc Zero-shot VLM).
* Tính toán tỉ lệ cải thiện / tối ưu hóa:
  $$\text{Tỉ lệ tối ưu VRAM} = \frac{\text{VRAM}_{\text{baseline}} - \text{VRAM}_{\text{exp}}}{\text{VRAM}_{\text{baseline}}} \times 100\%$$

### Điểm 4: Hồ sơ Quyết định Kiến trúc & Đánh đổi (ADR Mapping)
* Nếu kết quả đo đạc xác nhận hoặc bác bỏ một giả thuyết kiến trúc, phải cập nhật file ADR liên quan tại `docs/adrs/` (ví dụ: `ADR-0001` về nạp INT4).
* Nêu rõ mặt tích cực và các điểm đánh đổi (trade-offs) về mặt học thuật.

### Điểm 5: Đoạn văn Mẫu Báo cáo Học thuật (Thesis Snippet)
* Soạn sẵn một đoạn văn hoàn chỉnh bằng ngôn phong học thuật của đề tài khoa học.
* Chỉ rõ đoạn văn này sẽ nằm ở mục nào trong đề cương báo cáo (ví dụ: Mục 3.2 hay Mục 4.1).

---

## 3. CẤU TRÚC PHÂN TẦNG LƯU TRỮ HỒ SƠ (`docs/` DIRECTORY STRUCTURE)

Mọi văn bản ghi chép khoa học phải được lưu đúng thư mục quy định:

```text
financial-vlm-research/
└── docs/
    ├── adrs/                                # Bản ghi quyết định kiến trúc học thuật
    │   ├── 0001-chon-mo-hinh-vlm-va-luong-hoa-int4.md
    │   ├── 0002-time-series-split-triet-tieu-lookahead.md
    │   ├── 0003-hybrid-pipeline-ocr-cross-validation.md
    │   └── 0004-khung-danh-gia-dinh-luong-3-tang.md
    │
    ├── benchmarks/                          # Đo đạc phần cứng, VRAM, tốc độ suy luận
    │   ├── 00_hardware_vram_smoke_test.md   # Kết quả đo đạc GPU T4 16GB
    │   └── 01_latency_and_throughput.md     # Đo thời gian suy luận trên mỗi ảnh nến
    │
    ├── experiments/                         # Báo cáo tiến trình huấn luyện các đợt
    │   ├── exp01_baseline_lora/             # Nhật ký thực nghiệm 1: LoRA thô
    │   └── exp02_cot_finetuning/            # Nhật ký thực nghiệm 2: Huấn luyện có CoT
    │
    ├── evaluations/                         # Bảng tổng hợp số liệu nghiệm thu
    │   ├── ocr_cross_validation_metrics.md  # Sai số trích xuất giá
    │   ├── llm_judge_benchmark_table.md     # Điểm thẩm định chuyên gia từ GPT-4o
    │   └── simulated_backtest_results.md    # Kết quả Backtest chiến lược giao dịch
    │
    └── QUY_CHUAN_LUU_TRU_GITHUB_DRIVE.md   # Thiết kế hệ thống Tam giác phân lập
```

---

## 4. TEMPLATE CHUẨN KHI TẠO FILE BÁO CÁO THỰC NGHIỆM MỚI

Khi thực hiện đo đạc mới, dùng template Markdown sau:

```markdown
# BÁO CÁO THỰC NGHIỆM: [TÊN THỰC NGHIỆM]

* **Mã thực nghiệm:** `EXP-XX-[TÊN]`
* **Thời điểm thực hiện:** YYYY-MM-DD
* **Mục tiêu khoa học:** [Mô tả câu hỏi nghiên cứu cần chứng minh]
* **Hạ tầng:** [GPU / RAM / Python version / Library versions]

## 1. Phương pháp & Thiết lập Thực nghiệm
[Mô tả siêu tham số, dữ liệu đầu vào, prompt template sử dụng]

## 2. Kết quả Thực nghiệm Định lượng
| Chỉ số | Baseline | Kết quả Đề tài | Mức độ cải thiện (%) |
| :--- | :---: | :---: | :---: |
| ... | ... | ... | ... |

## 3. Phân tích & Thảo luận Học thuật
[Giải thích tại sao mô hình đạt được số liệu này, các trường hợp ngoại lệ (corner cases)]

## 4. Trích dẫn phục vụ Luận văn / Báo cáo Nghiệm thu (Thesis Snippet)
> *"[Đoạn văn viết bằng phong cách học thuật đưa vào Chương 3 hoặc Chương 4]"*
```

---

## 5. BẢNG KIỂM TRA CHÉO CHO AGENT (SELF-AUDIT CHECKLIST)

Trước khi kết thúc bất kỳ lượt phản hồi nào liên quan đến kết quả thực nghiệm:
- [ ] Số liệu đã được ghi vào một file trong `docs/benchmarks/` hoặc `docs/experiments/` chưa?
- [ ] Bảng số liệu có đầy đủ đơn vị tính và ý nghĩa khoa học không?
- [ ] Đã có công thức định lượng hoặc tỉ lệ cải thiện so với baseline chưa?
- [ ] Đã có đoạn văn mẫu (Thesis Snippet) để người dùng copy vào báo cáo chưa?
- [ ] Đã tạo commit Git với tiền tố `docs(benchmarks):` hoặc `docs(experiments):` chưa?
- [ ] Đã cập nhật file `HANDOFF.md` để người tiếp quản nắm bắt chưa?
