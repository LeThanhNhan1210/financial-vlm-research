---
name: debugging-and-error-recovery
description: >-
  Quy trình chẩn đoán và khắc phục lỗi hệ thống cho mô hình VLM và Google Colab T4.
  Kích hoạt kỹ năng này khi gặp lỗi CUDA Out-Of-Memory (OOM), lỗi nạp trọng số INT4 bitsandbytes,
  lỗi xử lý ảnh token thị giác, NaN loss, timeout hoặc lỗi đồng bộ Google Drive.
---

# DEBUGGING & ERROR RECOVERY CHO DỰ ÁN VLM (GOOGLE COLAB T4)

Quy trình xử lý sự cố có cấu trúc, không đoán mò (No Guesswork), tập trung vào các lỗi đặc thù khi huấn luyện và suy luận mô hình Vision-Language trên hạ tầng Google Colab GPU T4.

---

## 1. NGUYÊN TẮC "DỪNG DÂY CHUYỀN" (STOP-THE-LINE RULE)

Khi gặp bất kỳ lỗi runtime, crash hoặc kết quả bất thường nào:
1. **DỪNG NGAY** việc thay đổi mã nguồn hoặc thử nghiệm tiếp.
2. **LƯU VẾT LỖI:** Chụp lại toàn bộ Stacktrace, thông số `nvidia-smi` và file log gần nhất.
3. **CHẨN ĐOÁN THEO TẦNG:** Xác định lỗi thuộc tầng nào (Data $\rightarrow$ Pipeline/Processor $\rightarrow$ VRAM/CUDA $\rightarrow$ Drive/Network).
4. **SỬA TẬN GỐC (Root Cause):** Khắc phục nguyên nhân gốc rễ thay vì chữa cháy tạm thời.
5. **KIỂM CHỨNG LẠI:** Chạy kiểm thử trên 1 batch mẫu nhỏ trước khi tiếp tục train diện rộng.

---

## 2. BẢN ĐỒ XỬ LÝ CÁC LỖI ĐẶC THÙ (VLM TRIAGE CHECKLIST)

### A. Lỗi CUDA Out-Of-Memory (CUDA OOM)
```text
Xảy ra lỗi OOM (CUDA error: out of memory):
├── Xảy ra lúc Load Model?
│   ├── Kiểm tra bitsandbytes: Đã bật load_in_4bit=True và bnb_4bit_quant_type="nf4" chưa?
│   └── Kiểm tra device_map="auto" hoặc ép tải vào GPU 0: torch.cuda.set_device(0).
├── Xảy ra lúc Nạp Ảnh (Forward Pass)?
│   ├── Giảm độ phân giải ảnh (thumbnail về max 448x448 hoặc 512x512).
│   └── Kiểm tra số lượng Visual Tokens sinh ra cho mỗi ảnh.
└── Xảy ra lúc Huấn luyện (Backward Pass / Optimizer Step)?
    ├── Giảm per_device_train_batch_size = 1.
    ├── Bật gradient_checkpointing = True (tiết kiệm ~40% VRAM).
    ├── Tăng gradient_accumulation_steps (ví dụ từ 8 lên 16 hoặc 32) để giữ effective batch size.
    └── Đóng băng (freeze) toàn bộ Vision Encoder, chỉ train LoRA trên LM Backbone.
```

### B. Lỗi Kiểu Dữ Liệu & Precision (fp16 vs bf16 vs float32)
*   **Hiện tượng:** Loss ra `NaN` hoặc lỗi `RuntimeError: expected scalar type Half but found Float`.
*   **Cách khắc phục:**
    *   Google Colab T4 (kiến trúc Turing) **không hỗ trợ Native BF16**. Luôn dùng `fp16 = True`, `bf16 = False`.
    *   Đảm bảo `bnb_4bit_compute_dtype = torch.float16`.
    *   Thêm `torch.cuda.amp.autocast()` khi chạy suy luận.

### C. Lỗi Tiền xử lý Dữ liệu Ảnh (Image Preprocessing Errors)
*   **Hiện tượng:** `ValueError: cannot identify image file` hoặc `OSError: image file is truncated`.
*   **Quy trình kiểm tra:**
    1.  Luôn chuyển đổi ảnh về RGB: `image = Image.open(path).convert("RGB")` (tránh lỗi 4 kênh RGBA hoặc 1 kênh Grayscale).
    2.  Kiểm tra kích thước ảnh hợp lệ trước khi đưa vào Processor:
        ```python
        if image.width == 0 or image.height == 0:
            raise ValueError(f"Ảnh hỏng: {path}")
        ```

### D. Lỗi Timeout hoặc Đứt kết nối Google Colab
*   **Quy trình khôi phục nhanh:**
    1.  Mở Colab mới, chọn lại GPU T4.
    2.  Mount Google Drive: `drive.mount('/content/drive')`.
    3.  Khôi phục checkpoint gần nhất trong thư mục `checkpoints/` trên Drive bằng tham số `resume_from_checkpoint=True`.
    4.  Đọc file `outputs/logs/` để xác định chính xác epoch/step đã dừng.

---

## 3. LỆNH KIỂM TRA BỘ NHỚ VÀ DEBUG NHANH TRÊN COLAB

Chạy cell sau trên Colab khi nghi ngờ bộ nhớ bị rò rỉ (Memory Leak):
```python
import torch, gc

def debug_vram():
    gc.collect()
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[*] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

debug_vram()
```
