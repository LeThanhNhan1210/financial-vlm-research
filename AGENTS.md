# QUY TẮC DỰ ÁN (PROJECT RULES FOR AGENT)

Tài liệu này định nghĩa các quy chuẩn bắt buộc mà AI Agent và nhóm phát triển phải tuân thủ khi làm việc trong dự án này:

## 1. Giới hạn phần cứng và Bộ nhớ (Hardware Constraints)
- Môi trường chạy chính: Google Colab T4 GPU (16GB VRAM).
- Luôn sử dụng kỹ thuật lượng hóa INT4 (bitsandbytes) cho mô hình VLM 8B.
- Khi huấn luyện QLoRA: luôn đặt `per_device_train_batch_size = 1`, `gradient_accumulation_steps = 16`, bật `gradient_checkpointing = True`.
- Đóng băng (freeze) Vision Encoder khi fine-tune LoRA, chỉ train Language Model backbone.

## 2. Toàn vẹn dữ liệu chuỗi thời gian (Time-series Integrity)
- Tuyệt đối không dùng split ngẫu nhiên (random shuffle) cho dữ liệu tài chính.
- Phải chia tập dữ liệu nghiêm ngặt theo trục thời gian (Time-series Split: Train < Val < Test) để triệt tiêu hiện tượng Look-ahead bias.

## 3. Kiến trúc mã nguồn (Code Architecture)
- Không viết code logic xử lý phức tạp trực tiếp trong Jupyter Notebook. Tất cả hàm xử lý, nạp dữ liệu, mô hình, đánh giá phải được đóng gói thành module trong thư mục `src/`.
- File notebook chỉ đóng vai trò gọi module và trực quan hóa (EDA/Visualizations).
- Mọi siêu tham số (hyperparameters), đường dẫn, prompt templates phải nằm trong thư mục `configs/`, không hard-code trong mã nguồn Python.

## 4. Kiểm soát ảo giác thị giác (Anti-Hallucination)
- Luôn có cơ chế kiểm tra chéo (Cross-validation) giữa số liệu OCR trích xuất được từ trục tọa độ và kết quả suy luận thị giác từ Vision Encoder.

## 5. Lưu trữ Hồ sơ Thực nghiệm Khoa học (Scientific Logging & Paper Trail)
- Tuyệt đối không để trôi số liệu thực nghiệm (VRAM, loss, throughput, metrics) trên terminal hay notebook.
- Mọi kết quả đo đạc phải được lập tức số hóa và ghi chép thành file Markdown trong thư mục `docs/benchmarks/` hoặc `docs/experiments/` kèm theo bảng định lượng và đoạn văn học thuật mẫu (Thesis Snippet), tuân thủ nghiêm ngặt kỹ năng `scientific-research-logging`.
