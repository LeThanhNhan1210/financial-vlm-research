# ADR-0001: Lựa Chọn Mô Hình VLM Nền Tảng và Kỹ Thuật Lượng Hóa INT4

* **Trạng thái:** Đã chấp nhận (Accepted)
* **Thời điểm:** 2026-09-04
* **Người quyết định:** Nhóm nghiên cứu NCKH

## 1. Bối cảnh bài toán (Context)
Đề tài nghiên cứu ứng dụng mô hình ngôn ngữ thị giác (VLMs) trong phân tích biểu đồ tài chính.
Môi trường thực nghiệm tính toán chính là **Google Colab GPU T4 miễn phí (16GB VRAM)**.
Thách thức: Các mô hình VLM hiện đại (8B - 11B) khi nạp ở độ chính xác FP16 hoặc BF16 đòi hỏi hơn 16-20 GB VRAM chỉ riêng cho trọng số mô hình, chưa tính đến visual tokens và bộ nhớ optimizer khi huấn luyện, dẫn đến lỗi CUDA Out-Of-Memory (OOM).

## 2. Các phương án được cân nhắc (Options Considered)
* **Phương án A: Sử dụng API thương mại độc quyền (GPT-4o / Claude 3.5 Sonnet):**
  * *Ưu điểm:* Khả năng suy luận thị giác rất mạnh, không cần GPU nội bộ.
  * *Nhược điểm:* Đóng mã nguồn, không thể can thiệp fine-tune trọng số, chi phí API cao, thiếu tính đóng góp công nghệ học thuật tự chủ.
* **Phương án B: Llama 3.2 Vision (11B) với 16-bit LoRA:**
  * *Ưu điểm:* Mô hình mở mạnh mẽ của Meta.
  * *Nhược điểm:* Kích thước 11B quá lớn so với giới hạn 16GB VRAM của T4 khi fine-tune.
* **Phương án C (Được chọn): Họ Qwen 2.5-VL / Qwen 3-VL (8B) với lượng hóa INT4 (bitsandbytes NF4) & QLoRA:**
  * *Ưu điểm:* 
    1. Kiến trúc xử lý ảnh Dynamic Resolution giúp biểu đồ nến mảnh không bị vỡ nét mà vẫn tiết kiệm visual tokens.
    2. Lượng hóa INT4 đưa dung lượng mô hình 8B xuống còn ~5.5 GB VRAM lúc nạp, chừa hơn 10 GB VRAM cho forward/backward pass.
    3. Đóng băng (freeze) Vision Encoder, chỉ fine-tune LoRA adapter trên Language Backbone.

## 3. Quyết định (Decision)
Chọn **Phương án C**: Sử dụng họ mô hình Qwen-VL (bản 8B) kết hợp lượng hóa INT4 NF4 và kỹ thuật tinh chỉnh QLoRA.

## 4. Hệ quả & Đánh đổi (Consequences)
* **Tích cực:** Quá trình huấn luyện diễn ra ổn định trên Colab T4 với `batch_size = 1` và `gradient_accumulation_steps = 16`, triệt tiêu hoàn toàn rủi ro OOM.
* **Đánh đổi:** Card T4 không hỗ trợ native BF16, do đó phải duy trì tính toán ở chế độ FP16 mixed precision và kiểm soát chặt chẽ để tránh hiện tượng tràn số / loss NaN.

## 5. Tích hợp vào Báo cáo Khoa học (Thesis Mapping)
Tài liệu này là cơ sở trực tiếp cho **Mục 2.3 (Mô hình nền tảng)** và **Mục 3.2 (Kỹ thuật lượng hóa và tối ưu bộ nhớ)** trong Báo cáo tổng kết đề tài.
