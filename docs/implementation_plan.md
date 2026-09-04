# KẾ HOẠCH KỸ THUẬT PHASE 2: HUẤN LUYỆN QLORA TRÊN GOOGLE COLAB T4 & ĐỒNG BỘ CHECKPOINTS DRIVE

## 1. Bối cảnh và Mục tiêu Nghiên cứu (Context & Objectives)

Dự án đã hoàn tất **Giai đoạn 1 (Phase 1)**:
- Đã có **358 ảnh biểu đồ nến** 512×512 (Dark Mode, 3 lớp tài sản VN30, US_Equities, Crypto).
- Phân chia theo trục thời gian nghiêm ngặt thành **237 Train**, **44 Val**, **55 Test** (và 22 mẫu cách ly purge embargo).
- Tập `train.jsonl` đã được gán nhãn Chuỗi suy luận CoT 4 bước chuẩn CMT (*Xu hướng $\to$ Khối lượng/Chỉ báo $\to$ Rủi ro/Hỗ trợ $\to$ Khuyến nghị*) và vượt qua các chốt chặn kiểm toán toán học Anti-Hallucination.

**Mục tiêu của Phase 2**:
1. Tinh chỉnh mô hình thị giác ngôn ngữ **Qwen2.5-VL-7B-Instruct** bằng kỹ thuật **QLoRA (INT4 NF4)** trên hạ tầng **Google Colab GPU T4 (16GB VRAM)**.
2. Xây dựng **Data Collator chuyên dụng** cho Qwen2.5-VL:
   - Xử lý đồng thời token văn bản và token thị giác (Visual Tokens qua `image_grid_thw` & `pixel_values`).
   - Thiết lập cơ chế **Loss Masking**: Chỉ tính đạo hàm (Cross-Entropy Loss) trên phần câu trả lời của Trợ lý (`cot_reasoning` + `action`), gán nhãn `-100` cho toàn bộ phần Prompt và ảnh đầu vào.
3. Hoàn thiện **DriveSyncCallback**: Tự động sao lưu Adapter Checkpoints sang Google Drive (`MyDrive/NCKH_AI/2_checkpoints/`) sau mỗi 50 steps để triệt tiêu 100% rủi ro mất mát dữ liệu do ngắt kết nối Google Colab.
4. Tối ưu hóa tuyệt đối bộ nhớ VRAM để không bị lỗi **CUDA Out-Of-Memory (OOM)** trên Colab T4:
   - `per_device_train_batch_size = 1`
   - `gradient_accumulation_steps = 16` (effective batch size = 16)
   - `gradient_checkpointing = True`
   - `fp16 = True`
   - Freeze Vision Encoder hoàn toàn (chỉ train 47.58M tham số LoRA trên Language Model backbone).
5. Số hóa nhật ký huấn luyện định lượng (Loss curve, VRAM, throughput) theo chuẩn `scientific-research-logging` vào `docs/benchmarks/05_qlora_training.md`.

---

## 2. Yêu cầu Người dùng Đánh giá (User Review Required)

> [!IMPORTANT]
> **Các tham số huấn luyện cốt lõi trên Colab T4 (16GB VRAM):**
> 1. **Số Epochs:** Đề xuất `num_train_epochs: 3` (với 237 mẫu Train, effective batch size 16 $\rightarrow$ ~15 steps/epoch $\rightarrow$ Tổng ~45 steps huấn luyện, thời gian chạy dự kiến ~25-35 phút trên GPU T4).
> 2. **Tốc độ học (Learning Rate):** $2.0 \times 10^{-4}$ kèm cosine scheduler và 5% warmup (chuẩn mực cho LoRA rank $r=16, \alpha=32$).
> 3. **Cơ chế Validation Loss:** Thực hiện đánh giá trên 44 mẫu `val.jsonl` sau mỗi 20 steps để chọn checkpoint tối ưu nhất.

---

## 3. Kiến trúc Đề xuất & Thay đổi Mã nguồn (Proposed Changes)

Tuân thủ Clean Architecture và triết lý Ponytail:

### Module Huấn luyện & Dữ liệu (`src/training/`, `src/data/`)

#### [MODIFY] [src/data/dataset.py](file:///E:/financial-vlm-research/src/data/dataset.py)
- Nâng cấp `FinancialChartDataset` hỗ trợ chuyển đổi mẫu sang định dạng hội thoại chuẩn của Qwen2.5-VL:
  ```python
  messages = [
      {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": instruction}]},
      {"role": "assistant", "content": cot_reasoning}
  ]
  ```
- Thêm cơ chế tiền xử lý ảnh (resize chuẩn hóa $512 \times 512$).

#### [NEW] [src/training/collator.py](file:///E:/financial-vlm-research/src/training/collator.py)
- Triển khai lớp `QwenVLDataCollator`:
  - Sử dụng `AutoProcessor` của Qwen2.5-VL để tokenize text và trích xuất pixel values từ ảnh.
  - Áp dụng `qwen_vl_utils.process_vision_info`.
  - Triển khai **Loss Masking**: Tìm vị trí token `<|im_start|>assistant` và gán nhãn `labels = -100` cho tất cả token phía trước, đồng thời gán `-100` cho padding tokens.

#### [MODIFY] [src/training/callbacks.py](file:///E:/financial-vlm-research/src/training/callbacks.py)
- Hoàn thiện lớp `DriveSyncCallback`:
  - Kế thừa `TrainerCallback` từ Hugging Face Transformers.
  - Lắng nghe sự kiện `on_save()`: Tự động copy toàn bộ nội dung thư mục checkpoint vừa lưu (`adapter_model.safetensors`, `adapter_config.json`, `trainer_state.json`) sang thư mục đích trên Google Drive (`/content/drive/MyDrive/NCKH_AI/2_checkpoints/`).
  - Xử lý ngoại lệ an toàn (bọc trong `try/except`), ghi log chi tiết, không làm gián đoạn luồng train nếu Drive phản hồi chậm.

#### [NEW] [src/training/trainer.py](file:///E:/financial-vlm-research/src/training/trainer.py)
- Triển khai lớp `VLMQwenTrainer`:
  - Đóng gói toàn bộ quy trình: Nạp mô hình INT4 $\to$ Gắn LoRA adapter $\to$ Khởi tạo Data Collator $\to$ Cấu hình `TrainingArguments` $\to$ Gắn `DriveSyncCallback` $\to$ Chạy `trainer.train()`.
  - Tự động đo đạc và lưu thống kê: Peak VRAM, Training Loss, Validation Loss, Throughput.

#### [MODIFY] [scripts/run_train.py](file:///E:/financial-vlm-research/scripts/run_train.py)
- Hoàn thiện CLI script chạy huấn luyện:
  ```bash
  python scripts/run_train.py \
      --train-file /content/drive/MyDrive/NCKH_AI/1_datasets/splits/train_cot.jsonl \
      --val-file /content/drive/MyDrive/NCKH_AI/1_datasets/splits/val.jsonl \
      --output-dir /content/drive/MyDrive/NCKH_AI/2_checkpoints/qlora_run \
      --epochs 3 \
      --lr 2e-4 \
      --dry-run
  ```
  Hỗ trợ cờ `--dry-run` để kiểm tra toàn bộ pipeline (nạp dataset, collator, forward pass 1 step) mà không tốn thời gian train dài.

---

## 4. Kế hoạch Kiểm chứng Thực nghiệm (Verification Plan)

### 1. Kiểm thử Loss Masking & Tokenization (Unit Tests)
- Viết file test `tests/test_collator.py`:
  - Kiểm tra `QwenVLDataCollator` sinh đúng các tensor: `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `labels`.
  - Xác nhận nhãn `labels` tại các vị trí prompt/ảnh có đúng giá trị `-100`.
  - Xác nhận nhãn `labels` tại phần câu trả lời của assistant khớp với `input_ids`.

### 2. Kiểm thử Sao lưu Google Drive (DriveSyncCallback Test)
- Chạy unit test mô phỏng sự kiện `on_save`:
  - Tạo checkpoint giả lập trong thư mục tạm.
  - Gọi callback và xác nhận file đã được sao lưu sang thư mục đích đầy đủ.

### 3. Kiểm thử Huấn luyện Dry-Run (Forward/Backward Smoke Test)
- Chạy script với `--dry-run --epochs 1`:
  - Xác nhận mô hình INT4 nạp thành công trên GPU T4.
  - Xác nhận 1 bước gradient accumulation chạy trơn tru không OOM.
  - Xác nhận VRAM tiêu thụ thực tế nằm trong ngưỡng an toàn ($\le 12\text{ GB} / 15\text{ GB}$).
