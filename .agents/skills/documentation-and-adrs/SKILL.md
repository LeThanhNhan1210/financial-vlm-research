---
name: documentation-and-adrs
description: >-
  Quy chuẩn ghi chép hồ sơ nghiên cứu và Quyết định Kiến trúc (ADRs) cho đề tài NCKH.
  Kích hoạt kỹ năng này khi lựa chọn mô hình, thay đổi phương pháp lượng hóa, thiết kế pipeline,
  hoặc chuẩn bị nội dung học thuật cho Báo cáo đề tài (Chương Phương pháp & Thực nghiệm).
---

# QUY TRÌNH GHI CHÉP HỒ SƠ HỌC THUẬT & QUYẾT ĐỊNH KIẾN TRÚC (ADRS)

Kỹ năng này chuẩn hóa cách ghi lại các quyết định kỹ thuật và luận cứ khoa học, đảm bảo tính minh bạch học thuật (Transparency) và cung cấp tài liệu trực tiếp cho việc viết báo cáo nghiệm thu đề tài NCKH.

---

## 1. TẠI SAO CẦN ARCHITECTURE DECISION RECORDS (ADR) TRONG NCKH?

Trong nghiên cứu khoa học, mọi quyết định kỹ thuật đều phải có **lý do học thuật** và **số liệu thực nghiệm chứng minh**:
*   Không được viết: *"Chúng tôi dùng Qwen 8B vì nó tốt."*
*   Phải viết dạng ADR: *"Chúng tôi đánh giá giữa Qwen 2.5-VL 8B và Llama 3.2 Vision 11B; chọn Qwen 8B vì kiến trúc Dynamic Patching giúp tiết kiệm 35% visual tokens trên biểu đồ nến và tương thích tối ưu với bộ nhớ 16GB của Colab T4."*

Mỗi file ADR được lưu tại `docs/adrs/` theo thứ tự: `0001-chon-mo-hinh-vlm.md`, `0002-luong-hoa-int4-qlora.md`, v.v.

---

## 2. TEMPLATE CHUẨN CỦA MỘT BẢN ADR HỌC THUẬT

```markdown
# ADR-[SỐ THỨ TỰ]: [TIÊU ĐỀ QUYẾT ĐỊNH NGẮN GỌN]

* **Trạng thái:** [Đề xuất (Proposed) | Đã chấp nhận (Accepted) | Bị thay thế (Superseded)]
* **Thời điểm:** [YYYY-MM-DD]
* **Người quyết định:** Nhóm nghiên cứu NCKH

## 1. Bối cảnh bài toán (Context)
Mô tả vấn đề đang gặp phải và các ràng buộc phần cứng/học thuật:
- Ví dụ: Hạ tầng chỉ có 1 GPU T4 (16GB VRAM), cần fine-tune mô hình đọc biểu đồ tài chính.

## 2. Các phương án được cân nhắc (Options Considered)
- **Phương án A:** Full Fine-tuning mô hình 8B (Ưu điểm, Nhược điểm, VRAM yêu cầu).
- **Phương án B:** LoRA truyền thống 16-bit.
- **Phương án C (Được chọn):** QLoRA 4-bit (bitsandbytes NF4) kết hợp freeze Vision Encoder.

## 3. Quyết định (Decision)
Chúng tôi quyết định chọn Phương án C vì...

## 4. Hệ quả & Đánh đổi (Consequences)
- **Mặt tích cực:** Tiêu thụ chỉ ~6.2 GB VRAM lúc train, không bị OOM, có thể train với effective batch size = 16.
- **Đánh đổi/Rủi ro:** Cần đảm bảo hàm tính gradient ổn định trong môi trường fp16, không tận dụng được bf16 trên T4.

## 5. Tích hợp vào Báo cáo Khoa học (Thesis Mapping)
- Nội dung này sẽ được sử dụng cho Mục 3.2 (Kỹ thuật tối ưu hóa mô hình) trong Báo cáo tổng kết.
```

---

## 3. CÁC QUYẾT ĐỊNH CỐT LÕI CẦN LẬP ADR NGAY TRONG DỰ ÁN NÀY

1.  **ADR-0001:** Lựa chọn mô hình VLM nền tảng (Qwen-VL vs Llama-Vision).
2.  **ADR-0002:** Phương pháp phân chia dữ liệu Time-series Split (Loại bỏ Look-ahead bias).
3.  **ADR-0003:** Thiết kế Pipeline lai Vision + OCR Cross-validation chống ảo giác.
4.  **ADR-0004:** Chiến lược đánh giá 3 tầng (Thay thế BLEU/ROUGE bằng LLM-as-a-Judge và Backtest).
