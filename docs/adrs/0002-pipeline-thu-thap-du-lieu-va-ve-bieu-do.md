# ADR 0002: Thiết Kế Pipeline Kết Xuất Ảnh Biểu Đồ Nến Tự Động & Danh Mục 19 Mẫu Nến Target

* **Trạng thái:** Được chấp thuận (Accepted)
* **Ngày quyết định:** 2026-09-04
* **Tác giả:** AI Agent & Nhóm Nghiên cứu Khoa học
* **Khu vực áp dụng:** Giai đoạn 1 (Phase 1: Thu thập Dữ liệu & Xây dựng Chuỗi suy luận CoT)

---

## 1. Ngữ Cảnh và Vấn Đề (Context & Problem Statement)

Trong đề tài nghiên cứu ứng dụng mô hình ngôn ngữ thị giác (VLM) cho phân tích biểu đồ tài chính, dữ liệu ảnh biểu đồ là yếu tố quyết định năng lực suy luận của mô hình:
- Nếu chỉ dùng ảnh thu thập thủ công từ internet, tập dữ liệu sẽ thiếu đồng nhất về kích thước, tỷ lệ, font chữ, dính quảng cáo/watermark và tiềm ẩn thiên kiến lựa chọn (Selection Bias).
- Nếu vẽ quá nhiều chỉ báo kỹ thuật (RSI, Bollinger Bands, MACD, SMA đa khung), biểu đồ bị nhiễu thị giác nghiêm trọng, làm suy giảm khả năng nhận diện hình thái nến của Vision Encoder.
- VLM dễ gặp lỗi **ảo giác cấu trúc (Structural Hallucination)** khi đọc thân nến, râu nến nếu không có danh mục mẫu nến chuẩn hóa làm mục tiêu học tập (Ground Truth Target).

## 2. Các Quyết Định Kiến Trúc (Decision Drivers & Outcomes)

### A. Phương pháp Kết xuất Đồ họa Lập trình (Programmatic Rendering)
- Sử dụng thư viện chuyên dụng `mplfinance` kết hợp dữ liệu số OHLCV từ 3 nguồn chính thống: `yfinance` (US Equities), `ccxt` (Crypto), và `vnstock3` (VN30).
- Chuẩn hóa định dạng ảnh: **512×512 pixels**, giao diện nền tối **Dark Mode** (facecolor `#131722`) nhằm tối ưu độ tương phản của thân và bóng nến, đồng thời kiểm soát token thị giác tránh bùng nổ VRAM trên Colab T4.

### B. Tối giản Chỉ báo Thị giác (Visual Minimalism - Tuân thủ Ponytail / YAGNI)
- Tại Phase 1, biểu đồ chỉ hiển thị **nến Nhật OHLCV + 1 đường EMA20 (vàng `#f1c40f`) + Volume panel** bên dưới.
- Tạm lược bỏ SMA50, SMA200, RSI, Bollinger Bands để tập trung tối đa sự chú ý của Vision Encoder vào cấu trúc hình thái 19 mẫu nến.

### C. Mở rộng & Chuẩn hóa 19 Mẫu nến Target (Domain Knowledge)
Tham khảo tổng hợp từ 3 nguồn uy tín ([DNSE](https://www.dnse.com.vn/hoc/cac-mau-nen-dao-chieu-manh), [Binance Academy](https://www.binance.com/vi/academy/articles/a-beginners-guide-to-candlestick-charts), [Pinetree](https://pinetree.vn/post/20210622/bieu-do-nen-nhat-candlestick-charting-cach-doc-phan-tich-mo-hinh-nen-va-y-nghia-cac-loai-nen-trong-phan-tich-ky-thuat-chung-khoan/)), danh mục được phân thành 3 nhóm khoa học:
1. **Nhóm A (Cơ bản & Do dự):** Marubozu, Spinning Top, Doji chuẩn.
2. **Nhóm B (Đảo chiều Tăng):** Dragonfly Doji, Bullish Engulfing, Hammer, Inverted Hammer, Morning Star, Tweezer Bottom, Piercing Pattern, Three White Soldiers.
3. **Nhóm C (Đảo chiều Giảm):** Gravestone Doji, Bearish Engulfing, Hanging Man, Shooting Star, Evening Star, Tweezer Top, Dark Cloud Cover, Three Black Crows.

### D. Cơ chế Sliding Window và Manifest Metadata
- Cửa sổ: 60 nến/ảnh, bước trượt: 30 nến (overlap 50%) tạo ra ~168–224 ảnh đại diện từ các lớp tài sản.
- Tự động sinh `manifest.csv` lưu trữ `start_date`, `end_date`, `candle_count`, `detected_patterns` phục vụ trực tiếp cho bước Time-series Split và gán nhãn CoT tiếp theo.

## 3. Hậu Quả & Đánh Đổi (Consequences & Trade-offs)

* **Tích cực:** 
  - Đảm bảo tính khách quan tuyệt đối (Zero Bias) và khả năng tái lập 100% (Reproducibility).
  - Tương thích hoàn toàn khi chạy trên Colab với đầu ra trực tiếp vào Google Drive (`/content/drive/MyDrive/NCKH_AI/1_datasets/raw_images/`).
* **Đánh đổi:** 
  - Các mẫu nến hiếm (Morning Star, Three White Soldiers...) có thể có tần suất thấp trong dữ liệu tự động $\rightarrow$ Kích hoạt cơ chế Human-in-the-loop (HITL) bổ sung ảnh TradingView ở Bước 2.2.
