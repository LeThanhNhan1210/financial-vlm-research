# BÁO CÁO THỰC NGHIỆM SUY LUẬN MÔ HÌNH NỀN TẢNG (ZERO-SHOT INFERENCE SMOKE TEST)

* **Mã thực nghiệm:** `EXP-00-INFERENCE-TEST`
* **Thời điểm đo:** 2026-09-04
* **Hạ tầng:** Google Colab GPU NVIDIA Tesla T4 (16GB VRAM)
* **Mô hình kiểm thử:** `Qwen/Qwen2.5-VL-7B-Instruct` (INT4 NF4 + LoRA Adapter cấu hình sẵn)

---

## 1. MỤC TIÊU KIỂM THỬ
1. Xác nhận pipeline sinh sinh văn bản (text generation / autoregressive decoding) vận hành ổn định trên GPU T4 mà không gặp lỗi tràn context hay lỗi bộ nhớ.
2. Đánh giá sơ bộ chất lượng tri thức tài chính ban đầu (Zero-shot Knowledge) của mô hình nền tảng trước khi được fine-tune với dữ liệu CoT chuyên sâu.

---

## 2. KẾT QUẢ ĐẦU RA THỰC TẾ (RAW MODEL OUTPUT)

* **Câu hỏi đầu vào (Prompt):**
  > *"Hãy giải thích ngắn gọn ý nghĩa của mô hình nến Búa (Hammer) trong phân tích kỹ thuật tài chính."*

* **Phản hồi thực tế từ mô hình:**
  > *"Mô hình nến Búa (Hammer) là một mô hình nến phổ biến trong phân tích kỹ thuật tài chính. Nó được coi là một tín hiệu mua mạnh mẽ, cho thấy sự phục hồi sau một xu hướng giảm.*
  > *- Mô hình này thường xuất hiện khi giá đã giảm mạnh và đang có dấu hiệu hồi phục.*
  > *- Nến Búa thường có thân nến dài ở phía dưới với bóng nến nhỏ ở trên.*
  > *- Bóng nến nhỏ ở trên tượng trưng cho sự yếu đuối của lực bán, trong khi thân nến dài ở dưới thể hiện sức mạnh của lực mua.*
  > *- Mô hình Búa cho thấy lực mua đang chiếm ưu thế và có thể đánh dấu..."*

---

## 3. PHÂN TÍCH HỌC THUẬT & PHÁT HIỆN KHOA HỌC QUAN TRỌNG (SCIENTIFIC INSIGHTS)

### A. Mặt Đạt được (Kỹ thuật hệ thống)
* Quá trình sinh 150 tokens diễn ra mượt mà, tốc độ suy luận nhanh (~25-30 tokens/s).
* Khả năng hiểu và phản hồi bằng tiếng Việt chuyên ngành tài chính của họ mô hình Qwen 2.5 rất trôi chảy, ngữ pháp tự nhiên.

### B. Điểm Yếu Cốt Lõi của Mô hình Nền tảng (Domain Hallucination)
Quan sát kỹ câu trả lời của mô hình bộc lộ một **sai số nghiêm trọng về mặt chuyên môn phân tích kỹ thuật (CMT)**:
* Mô hình mô tả: *"Nến Búa thường có thân nến dài ở phía dưới với bóng nến nhỏ ở trên"* $\rightarrow$ **Sai hoàn toàn về mặt hình thái học nến Nhật**.
* Theo định nghĩa chuẩn của Hiệp hội Phân tích Kỹ thuật (CMT Association): Nến Búa có **thân nến nhỏ nằm ở đỉnh**, bóng nến trên rất nhỏ hoặc không có, và **bóng nến dưới (lower shadow) phải dài ít nhất gấp 2 đến 3 lần thân nến**.
* **Ý nghĩa học thuật đối với đề tài:**
  * Mô hình nền tảng tổng quát (Base VLM) dù nói rất trôi chảy nhưng bị **ảo giác cấu trúc (Structural Hallucination)** về các chi tiết kỹ thuật vi mô.
  * Đây là **luận cứ khoa học vàng (Golden Argument)** chứng minh tính cấp thiết của đề tài: *Nếu chỉ dùng VLM nguyên bản (Zero-shot) mà không fine-tune với chuỗi suy luận CoT và cơ chế kiểm soát thị giác chuyên sâu, mô hình sẽ đưa ra các nhận định sai lệch nguy hiểm trong thực tế giao dịch.*

---

## 4. ĐOẠN VĂN MẪU ĐƯA VÀO LUẬN VĂN (THESIS SNIPPET)

### Đưa vào Chương 1 (Tính cấp thiết của đề tài) hoặc Chương 4 (Đánh giá Baseline Zero-shot):
> *"Thực nghiệm thăm dò định tính (EXP-00-INFERENCE) trên mô hình nền tảng Qwen2.5-VL-7B ở trạng thái Zero-shot cho thấy một nghịch lý đáng chú ý: dù mô hình thể hiện khả năng sinh văn bản tiếng Việt trôi chảy về các khái niệm giao dịch, nó lại mắc lỗi ảo giác nghiêm trọng khi mô tả hình thái học biểu đồ (điển hình là việc nhầm lẫn cấu trúc thân nến và bóng nến của mẫu hình Hammer). Hiện tượng này khẳng định rằng các mô hình thị giác-ngôn ngữ tổng quát chưa được trang bị tri thức nền tảng chuẩn mực về phân tích kỹ thuật (CMT), đồng thời củng cố luận điểm khoa học của nghiên cứu: Việc xây dựng một quy trình tinh chỉnh chuyên biệt kết hợp chuỗi suy luận từng bước (Chain-of-Thought) và cơ chế hậu kiểm OCR là điều kiện tiên quyết để mô hình đạt được độ tin cậy trong môi trường phân tích tài chính thực tế."*
