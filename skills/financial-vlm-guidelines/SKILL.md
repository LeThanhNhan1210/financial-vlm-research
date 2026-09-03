---
name: financial-vlm-guidelines
description: >-
  Bộ quy chuẩn, quy tắc và quy trình chuẩn cho đề tài nghiên cứu VLM phân tích biểu đồ tài chính.
  Kích hoạt kỹ năng này khi thực hiện các tác vụ tiền xử lý dữ liệu biểu đồ, thiết kế prompt CoT,
  huấn luyện QLoRA trên Google Colab T4, hoặc triển khai đánh giá đa tầng (OCR, LLM Judge, Backtest).
---

# HƯỚNG DẪN VÀ QUY TẮC PHÁT TRIỂN (FINANCIAL VLM GUIDELINES)

Tài liệu này đóng vai trò là "Sổ tay quy chuẩn" (Runbook & Cheatsheet) cho toàn bộ quá trình nghiên cứu và thực nghiệm của đề tài.

---

## 1. NGUYÊN TẮC QUẢN TRỊ DỮ LIỆU (DATA INTEGRITY)

*   **Quy tắc bất biến về Time-series Split:**
    *   **TUYỆT ĐỐI KHÔNG** sử dụng `train_test_split(..., shuffle=True)` ngẫu nhiên.
    *   Phải chia theo mốc thời gian: $T_{train} < T_{val} < T_{test}$ để triệt tiêu hoàn toàn hiện tượng nhìn trước tương lai (*Look-ahead bias / Data leakage*).
*   **Phân tầng dữ liệu:**
    *   Thư mục `data/raw/`: Chỉ lưu ảnh gốc, không chỉnh sửa, không commit lên Git.
    *   Thư mục `data/processed/`: Ảnh đã qua xử lý chuẩn hóa độ phân giải tối đa 512x512 để tránh bùng nổ token thị giác trên VRAM.
    *   Thư mục `data/annotations/`: File `.jsonl` chứa nhãn do GPT-4o sinh và đã được rà soát (Human Audit).

---

## 2. QUY TẮC BẢO TOÀN PHẦN CỨNG & VRAM (COLAB T4 OPTIMIZATION)

Hạ tầng thực nghiệm là Google Colab GPU T4 (16GB VRAM), bắt buộc tuân thủ:
*   **Lượng hóa INT4 (bitsandbytes):** Luôn nạp mô hình với `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.float16`.
*   **Chiến lược QLoRA:**
    *   Đóng băng (Freeze) toàn bộ Vision Encoder và Projection Layer ban đầu.
    *   Chỉ gắn LoRA adapter lên Language Backbone (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
    *   LoRA rank: $r=16$, $\alpha=32$, dropout $=0.05$.
*   **Tham số Batch & Bộ nhớ:**
    *   `per_device_train_batch_size = 1` (tránh OOM).
    *   `gradient_accumulation_steps = 16` (đảm bảo effective batch size = 16).
    *   Luôn bật `gradient_checkpointing = True`.

---

## 3. QUY TẮC THIẾT KẾ PIPELINE CHỐNG ẢO GIÁC (ANTI-HALLUCINATION)

*   VLM rất dễ bị ảo giác về con số và râu nến mờ.
*   **Quy trình 3 bước bắt buộc trong `src/pipeline/`:**
    1.  *Tiền xử lý:* Tăng tương phản nhẹ để làm rõ bóng nến (wicks) và đường chỉ báo.
    2.  *Trích xuất kép:* Tách riêng phần nhận diện hình thái nến (Vision) và đọc giá trị số trên 2 trục (OCR).
    3.  *Đối chiếu chéo:* So sánh giá khuyến nghị (Entry/SL/TP) với khoảng giá hiển thị trên trục $Y$ để loại trừ các số liệu vô lý trước khi xuất báo cáo.

---

## 4. QUY CHUẨN ĐÁNH GIÁ ĐA TẦNG (EVALUATION PROTOCOL)

*   **Không dùng duy nhất ROUGE/BLEU** vì chỉ phản ánh độ trùng lặp từ ngữ, không đo lường được logic tài chính.
*   **Đánh giá chuẩn 3 tầng:**
    *   *Tầng 1 (Thị giác/Số liệu):* Sai số MAPE/MAE các mốc giá và độ chính xác nhận diện mẫu nến.
    *   *Tầng 2 (Lập luận):* LLM-as-a-Judge (GPT-4o) dựa trên 4 tiêu chí trong [eval_rubric.yaml](file:///e:/NCKH/configs/eval_rubric.yaml) (Soundness, Consistency, Risk, Hallucination).
    *   *Tầng 3 (Thực chiến):* Tỷ lệ bắt đúng chiều xu hướng tiếp theo (Directional Accuracy %) và tỷ lệ thắng mô phỏng (Backtest Win-rate).

---

## 5. QUY TẮC LẬP TRÌNH & ĐỒNG BỘ MÃ NGUỒN

*   **Clean Architecture:**
    *   Code chức năng viết trong `src/`, tuyệt đối không viết hàm dài hàng trăm dòng trong Jupyter Notebook.
    *   Notebook chỉ đóng vai trò Controller (gọi hàm và trực quan hóa kết quả).
*   **Cơ chế lưu trữ:**
    *   Luôn gọi `DriveSyncCallback` trong quá trình train để tự động lưu checkpoint sang Google Drive sau mỗi 50 steps.
