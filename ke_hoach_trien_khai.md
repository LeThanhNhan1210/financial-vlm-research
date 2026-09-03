# KẾ HOẠCH TRIỂN KHAI ĐỀ TÀI NGHIÊN CỨU KHOA HỌC
**Tên đề tài/Tài liệu tham chiếu:** Ứng dụng mô hình ngôn ngữ thị giác (VLMs) trong phân tích biểu đồ tài chính và hỗ trợ đề xuất chiến lược

---

## TỔNG QUAN VÀ ĐỊNH VỊ ĐÓNG GÓP KHOA HỌC
Đề tài định hướng giải quyết bài toán chuyển dịch từ **Factoid VQA** (trích xuất thông tin bề mặt) sang **Actionable Reasoning** (lập luận chuỗi suy luận đa bước để đề xuất chiến lược giao dịch/quản trị rủi ro) bằng các mô hình thị giác ngôn ngữ mã nguồn mở (như họ Qwen-VL / Llama-Vision 8B), tối ưu hóa trên hạ tầng tính toán hạn chế (Google Colab T4 GPU 16GB VRAM).

### 3 Đóng góp khoa học trọng tâm của đề án:
1. **Đóng góp Dữ liệu (Dataset & Benchmark):** Xây dựng bộ dữ liệu chuyên biệt gồm biểu đồ kỹ thuật đi kèm chuỗi lập luận (Chain-of-Thought - CoT) tài chính chuẩn mực, phân chia dữ liệu theo trục thời gian (Time-series split) nhằm triệt tiêu thiên kiến tương lai (Look-ahead bias).
2. **Đóng góp Kiến trúc Pipeline (Anti-Hallucination Framework):** Thiết kế pipeline lai (Hybrid Pipeline) kết hợp trích xuất đặc trưng thị giác (Vision Encoder) với kiểm tra chéo số liệu trục tọa độ/ngưỡng giá (OCR Cross-validation) nhằm giảm thiểu ảo giác thị giác.
3. **Đóng góp Thực nghiệm & Đánh giá (Evaluation Methodology):** Đánh giá định lượng toàn diện 3 tầng (OCR Accuracy $\rightarrow$ LLM-as-a-Judge Reasoning $\rightarrow$ Simulated Backtesting / Directional Accuracy) so sánh giữa Zero-shot, Few-shot CoT và QLoRA.

---

## PHẦN I: KẾ HOẠCH TRIỂN KHAI CHI TIẾT THEO GIAI ĐOẠN

### Giai đoạn 1: Xác định bài toán mục tiêu và Thu thập dữ liệu chuẩn hóa (Tuần 1 - 3)
*   **Khoanh vùng phạm vi bài toán mục tiêu:**
    *   *Loại biểu đồ mục tiêu:* Tập trung chuyên sâu vào **Biểu đồ nến Nhật (Candlestick chart)** kết hợp **Khối lượng giao dịch (Volume)** và các chỉ báo kỹ thuật phổ biến (Đường trung bình động SMA/EMA, RSI, MACD).
    *   *Thị trường thực nghiệm:* Dữ liệu thị trường chứng khoán (VN-Index/VN30 hoặc S&P 500) và Crypto (BTC/USDT) ở khung thời gian ngày (Daily) và 4 giờ (4H).
    *   *Định dạng đầu ra:* Báo cáo phân tích kỹ thuật chuẩn cấu trúc: Nhận diện mô hình giá $\rightarrow$ Đánh giá động lượng/xu hướng $\rightarrow$ Cảnh báo rủi ro $\rightarrow$ Đề xuất hành động (Entry, Stop-loss, Take-profit).
*   **Chiến lược xây dựng bộ dữ liệu (Synthetic Data + Human-in-the-loop):**
    *   Thu thập 250 - 300 mẫu ảnh biểu đồ đại diện cho các pha thị trường: Tích lũy (Accumulation), Xu hướng tăng (Uptrend), Xu hướng giảm (Downtrend), Đảo chiều (Reversal).
    *   Sử dụng mô hình thương mại tiên tiến (GPT-4o API) với Prompt có cấu trúc CoT tài chính để sinh nhãn phân tích mẫu tự động (Synthetic Annotation).
    *   *Human Audit:* Tiến hành rà soát, chuẩn hóa thủ công bộ dữ liệu để đảm bảo tính chuẩn xác của các mốc giá và logic giao dịch, hình thành bộ **Golden Dataset**.
*   **Quy tắc phân chia dữ liệu nghiêm ngặt:**
    *   Áp dụng **Time-series Split** theo mốc thời gian (ví dụ: Dữ liệu quá khứ cho Train/Few-shot examples, giai đoạn gần nhất cho Test), tuyệt đối không dùng Random Shuffle để ngăn ngừa hiện tượng Data Leakage / Look-ahead Bias.

---

### Giai đoạn 2: Thiết kế kiến trúc hệ thống và Pipeline kiểm soát ảo giác (Tuần 4 - 5)
*   **Lựa chọn mô hình nền tảng:**
    *   Mô hình mục tiêu: **Qwen 2.5/3 VL (bản 8B)** hoặc **Llama 3.2 Vision (11B)**.
    *   Áp dụng kỹ thuật lượng hóa **INT4 (bitsandbytes)** giúp giảm dung lượng VRAM khi load mô hình xuống mức 5.5 - 6.5 GB, đảm bảo suy luận mượt mà trên Google Colab T4 (16GB VRAM).
*   **Xây dựng Pipeline lai (Hybrid Anti-Hallucination Pipeline):**
    *   *Nhánh 1 - Tiền xử lý ảnh (ROI Processing):* Chuẩn hóa kích thước, tăng cường độ tương phản vùng râu nến và nhãn trục giá/thời gian.
    *   *Nhánh 2 - Trích xuất kép (Dual Feature Extraction):* 
        *   Vision Encoder trích xuất hình thái trực quan tổng thể (Mô hình nến, đường xu hướng).
        *   Module OCR chuyên dụng hoặc OCR tích hợp trích xuất trực tiếp giá trị số trên 2 trục tọa độ (Mốc thời gian & Khung giá cao/thấp).
    *   *Nhánh 3 - Lớp đối chiếu chéo (Cross-Validation Layer):* So khớp ngưỡng giá mô hình đọc được với vùng tọa độ thực để lọc bỏ các suy luận sai lệch về con số trước khi chuyển qua bước phân tích logic.
    *   *Nhánh 4 - Multimodal Reasoning Engine:* Prompting theo cấu trúc Chain-of-Thought (CoT) chuyên ngành tài chính.

---

### Giai đoạn 3: Chiến lược tối ưu hóa 2 bước & Lưu trữ tự động (Tuần 6 - 9)
*   **Bước 3.1: Tối ưu hóa bằng In-Context Learning (Few-shot CoT Prompting):**
    *   Xây dựng hệ thống Prompt Template chuyên biệt chứa 2 - 3 ví dụ mẫu chuẩn (Few-shot Examples) có đầy đủ các bước lập luận logic tài chính.
    *   Đo lường hiệu năng suy luận của mô hình nguyên bản (Zero-shot vs Few-shot CoT) làm đường cơ sở (Baseline).
*   **Bước 3.2: Tinh chỉnh nhẹ có điều kiện bằng QLoRA (Parameter-Efficient Fine-Tuning):**
    *   Nếu Few-shot chưa đạt kỳ vọng về độ ổn định ngữ cảnh, tiến hành Fine-tuning bằng **QLoRA** (lượng hóa 4-bit trong quá trình huấn luyện).
    *   *Cấu hình tối ưu cho Colab T4:*
        *   Đóng băng (Freeze) hoàn toàn Vision Encoder, chỉ cập nhật adapter LoRA trên Language Model Backbone.
        *   `batch_size = 1`, `gradient_accumulation_steps = 16`.
        *   Bật `gradient_checkpointing = True` và sử dụng `bf16`/`fp16` mixed precision.
*   **Thiết lập môi trường & Cơ chế bảo toàn dữ liệu:**
    *   Tự động mount Google Drive ngay đầu notebook.
    *   Thiết lập callback tự động lưu các file log JSON, checkpoint adapter và bảng kết quả suy luận vào Google Drive sau mỗi epoch/chu kỳ 30 phút để phòng ngừa ngắt kết nối đột ngột từ Colab.

---

### Giai đoạn 4: Đánh giá định lượng đa tầng và Kiểm định thực chiến (Tuần 10 - 11)
Xây dựng khung đánh giá toàn diện gồm 3 tầng tiêu chí thay vì chỉ dựa vào các thang đo ngữ pháp truyền thống:

*   **Tầng 1 - Độ chính xác trích xuất thị giác (Visual & OCR Accuracy):**
    *   Sai số tuyệt đối trung bình (MAE / MAPE) giữa giá trị đỉnh/đáy/mốc thời gian mô hình đọc được so với thực tế trên biểu đồ.
    *   Tỷ lệ nhận diện đúng tên các mô hình nến kỹ thuật cơ bản (Pinbar, Doji, Engulfing, Double Top/Bottom).
*   **Tầng 2 - Đánh giá chất lượng lập luận (Reasoning Quality via LLM-as-a-Judge):**
    *   Thay thế các thang đo ngữ pháp từ vựng (ROUGE, BLEU) bằng phương pháp **LLM-as-a-Judge** (sử dụng GPT-4o đánh giá dựa trên bộ Rubric chuẩn):
        *   *Tính logic tài chính (Financial Soundness):* Lập luận có phù hợp với nguyên lý thị trường hay không?
        *   *Tính nhất quán (Internal Consistency):* Nhận định xu hướng có đồng nhất với mức giá cắt lỗ/chốt lời đề xuất không?
        *   *Mức độ kiểm soát rủi ro (Risk Mitigation):* Có đưa ra cảnh báo rủi ro hợp lý khi chỉ báo mâu thuẫn không?
*   **Tầng 3 - Kiểm định tác động thực chiến (Actionable & Directional Metrics):**
    *   *Tỷ lệ dự đoán đúng xu hướng (Directional Accuracy %):* Kiểm tra xem các khuyến nghị (Tăng/Giảm/Đi ngang) có đúng với diễn biến thực tế trong $K$ phiên tiếp theo trên biểu đồ kiểm thử hay không.
    *   *Mô phỏng giao dịch (Simulated Backtest):* Tính toán tỷ lệ thắng giả định (Win Rate) và tỷ lệ Lợi nhuận/Rủi ro (Risk:Reward Ratio) trên tập dữ liệu kiểm thử.

---

### Giai đoạn 5: Viết báo cáo khoa học, Đóng gói và Hoàn thiện (Tuần 12 - 14)
*   **Biên soạn báo cáo tổng kết đề tài NCKH:**
    *   *Chương 1:* Tổng quan tình hình nghiên cứu về VLM và Financial Chart QA trên thế giới.
    *   *Chương 2:* Cơ sở lý thuyết về Vision-Language Models, Lượng hóa INT4/QLoRA và Phân tích kỹ thuật.
    *   *Chương 3:* Đề xuất phương pháp (Kiến trúc Pipeline lai, Quy trình xây dựng dữ liệu CoT, Cơ chế chống ảo giác).
    *   *Chương 4:* Kết quả thực nghiệm và Thảo luận chuyên sâu (So sánh đối chứng Zero-shot vs Few-shot vs QLoRA qua 3 tầng đánh giá).
    *   *Chương 5:* Kết luận, Giới hạn đề tài và Hướng phát triển trong tương lai.
*   **Đóng gói sản phẩm nghiên cứu:**
    *   Xuất bản kho mã nguồn mở (Clean Code) trên GitHub kèm hướng dẫn tái tạo kết quả (Reproducibility Guide) và notebook chạy thử trên Colab bằng 1 cú nhấp chuột.
    *   Biên soạn Slide thuyết trình, video demo trực quan hóa luồng phân tích biểu đồ của mô hình để phục vụ buổi bảo vệ đề tài.

---

## PHẦN II: MA TRẬN THEO DÕI VÀ PHÂN BỔ TÀI NGUYÊN

| Hạng mục | Công cụ / Kỹ thuật chính | Sản phẩm đầu ra cụ thể | Rủi ro & Giải pháp dự phòng |
| :--- | :--- | :--- | :--- |
| **Dữ liệu (Tuần 1-3)** | Python, GPT-4o API, Time-series split | 250+ biểu đồ + bộ nhãn CoT chuẩn hóa | Rủi ro nhãn ảo $\rightarrow$ Human audit rà soát 100% tập Test |
| **Hệ thống (Tuần 4-5)** | Qwen-VL / Llama-Vision, Bitsandbytes INT4 | Pipeline hoàn chỉnh chạy được trên Colab | Rủi ro OOM $\rightarrow$ Tối ưu hóa kích thước ảnh và độ dài context |
| **Thực nghiệm (Tuần 6-9)** | Few-shot CoT, QLoRA, Google Drive API | Checkpoints, Log thực nghiệm, Bảng so sánh | Rủi ro timeout $\rightarrow$ Auto-save checkpoint định kỳ lên Drive |
| **Kiểm định (Tuần 10-11)**| LLM-as-a-Judge, Backtest Script | Báo cáo định lượng 3 tầng (OCR, Logic, PnL) | BLEU/ROUGE không phản ánh đúng $\rightarrow$ Dùng Rubric LLM Judge |
| **Báo cáo (Tuần 12-14)** | LaTeX / Word, GitHub Repo, Slide Demo | Báo cáo NCKH hoàn chỉnh, Slide, Repo mã nguồn | Thiếu tính thuyết phục $\rightarrow$ Bổ sung demo trực quan thời gian thực |
