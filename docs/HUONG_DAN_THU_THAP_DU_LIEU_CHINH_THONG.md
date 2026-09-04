# HƯỚNG DẪN THU THẬP DỮ LIỆU TÀI CHÍNH CHÍNH THỐNG & PHƯƠNG PHÁP CHUẨN NCKH
## DATA SOURCES AND COLLECTION METHODOLOGIES FOR FINANCIAL VLM RESEARCH

*   **Tên đề tài NCKH:** Ứng dụng mô hình ngôn ngữ thị giác (VLMs) trong phân tích biểu đồ tài chính và hỗ trợ đề xuất chiến lược
*   **Giai đoạn áp dụng:** Giai đoạn 1 (Phase 1: Thu thập Dữ liệu & Xây dựng Chuỗi suy luận CoT)
*   **Tiêu chuẩn học thuật:** IEEE / ACM / MLOps Data Integrity Standards

---

## 1. CÁC NGUỒN CUNG CẤP DỮ LIỆU CHÍNH THỐNG (AUTHORITATIVE DATA SOURCES)

Để một công trình nghiên cứu khoa học có giá trị học thuật và được các hội đồng thẩm định đánh giá cao, dữ liệu đầu vào bắt buộc phải xuất phát từ các kênh chính thống, minh bạch và có khả năng truy xuất nguồn gốc (Traceability):

### A. Thị trường Chứng khoán Việt Nam (VN30 & Cổ phiếu niêm yết)
1.  **Thư viện `vnstock` / `vnstock3` (Khuyên dùng số 1 trong NCKH tại VN):**
    *   *Nguồn gốc:* Thư viện mã nguồn mở uy tín nhất của cộng đồng định lượng tài chính Việt Nam, tự động kết nối trực tiếp đến các cổng dữ liệu của SSI iBoard, TCBS, CafeF, Vietstock.
    *   *Dữ liệu cung cấp:* Lịch sử nến OHLCV (Mở, Cao, Thấp, Đóng, Khối lượng) từ cấp Ngày (1D) đến Phút (1m), dữ liệu đã được điều chỉnh giá (Adjusted Close).
    *   *Cài đặt:* `pip install vnstock3`
2.  **TradingView (Dành cho xuất ảnh biểu đồ trực quan):**
    *   *Nguồn gốc:* Cổng phân tích kỹ thuật chuẩn mực toàn cầu, cấp dữ liệu trực tiếp từ Sở Giao dịch Chứng khoán TP.HCM (HOSE) và Hà Nội (HNX).
    *   *Ưu điểm:* Biểu đồ hiển thị nến sắc nét, đầy đủ các chỉ báo kỹ thuật chuẩn (MA, RSI, MACD, Volume Profile) và hỗ trợ chế độ nền tối (Dark mode) chuẩn mực.
3.  **Dữ liệu Lịch sử từ Vietstock / CafeF / Sở GDCK (HSX):**
    *   Có thể tải trực tiếp file `.csv` lịch sử giao dịch nhiều năm từ trang thống kê của CafeF hoặc Vietstock Finance để lưu trữ thành tập dữ liệu bất biến (Raw Data Benchmark).

### B. Thị trường Chứng khoán Mỹ & Quốc tế (S&P500, ETFs, Tech Stocks)
1.  **Yahoo Finance (`yfinance` API - Tiêu chuẩn Học thuật Quốc tế):**
    *   *Nguồn gốc:* Nguồn dữ liệu miễn phí phổ biến nhất thế giới được trích dẫn trong hàng ngàn bài báo khoa học trên Springer, IEEE và ScienceDirect.
    *   *Dữ liệu:* Lịch sử giá hơn 30 năm của các mã lớn (SPY, AAPL, NVDA, TSLA), tự động xử lý chia tách cổ phiếu và cổ tức.
    *   *Cài đặt:* `pip install yfinance`
2.  **Alpha Vantage (Cổng API Dữ liệu Tài chính Học thuật):**
    *   *Nguồn gốc:* Nhà cung cấp dữ liệu chính thức hỗ trợ các viện nghiên cứu và trường đại học, cung cấp sẵn các chỉ báo kỹ thuật tính toán tự động (RSI, EMA, MACD, Bollinger Bands).
    *   *Website:* `alphavantage.co` (Có gói Free API key 25 lượt gọi/ngày).
3.  **FRED (Federal Reserve Economic Data):**
    *   Dữ liệu kinh tế vĩ mô chính thức của Cục Dự trữ Liên bang Mỹ (hữu ích nếu sau này mở rộng phân tích liên thị trường).

### C. Thị trường Tiền mã hóa (Cryptocurrency)
1.  **Binance Official Public Data API / CCXT Library:**
    *   *Nguồn gốc:* Sàn giao dịch tiền mã hóa có thanh khoản lớn nhất thế giới.
    *   *Thư viện `ccxt`:* Thư viện chuẩn công nghiệp kết nối API hơn 100 sàn tiền số, cho phép tải nến 1D, 4H, 1H không giới hạn lịch sử.
    *   *Cài đặt:* `pip install ccxt`
2.  **CoinGecko / CoinMarketCap API:**
    *   Cung cấp dữ liệu vốn hóa, biến động lịch sử của BTC, ETH, SOL.

---

## 2. CÁC PHƯƠNG THỨC THU THẬP DỮ LIỆU ĐƯỢC TIN CẬY TRONG NCKH

Trong nghiên cứu khoa học chuyên sâu về Computer Vision và AI Tài chính, việc thu thập dữ liệu không chỉ đơn thuần là "tải ảnh trên mạng", mà phải áp dụng các phương thức có cơ sở toán học và kỹ thuật phần mềm chuẩn mực:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        3 PHƯƠNG THỨC THU THẬP DỮ LIỆU ĐỀ TÀI                           │
│                                                                                        │
│   [Phương thức 1: Programmatic Rendering]   [Phương thức 2: TradingView Export]        │
│   • Tải OHLCV dạng số (yfinance, vnstock)  • Chụp/Xuất ảnh trực tiếp từ TradingView   │
│   • Dùng mplfinance vẽ biểu đồ chuẩn 512px • Đảm bảo độ tự nhiên của biểu đồ thực tế   │
│   • Độ sạch 100%, không nhiễu watermark    • Thao tác nhanh cho các mẫu hình kinh điển│
│                                     │                           │                      │
│                                     ▼                           ▼                      │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │        [Phương thức 3: Human-in-the-Loop Curation - CHUẨN MỰC CAO NHẤT]         │   │
│   │   • AI thu thập & sinh nhãn CoT thô (80%)                                      │   │
│   │   • Tác giả rà soát, tinh lọc và thẩm định chuyên sâu (30% - 40%)              │   │
│   │   • Khóa tập dữ liệu theo Time-series Split (Train < Val < Test)               │   │
│   └────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Phương thức 1: Kết xuất Đồ họa Tự động từ Dữ liệu Số (Programmatic Rendering via `mplfinance`)
*   **Cách thức:** 
    1. Dùng code Python gọi `yfinance` hoặc `vnstock` để lấy bảng số liệu nến OHLCV dạng DataFrame.
    2. Dùng thư viện đồ họa tài chính **`mplfinance`** để tự động vẽ biểu đồ nến Nhật kèm khối lượng và các đường chỉ báo kỹ thuật (EMA20, SMA50, RSI).
    3. Tự động xuất ra file ảnh `.png` đúng kích thước **512x512 pixels** nền tối (Dark Theme).
*   **Tại sao NCKH rất tin cậy phương thức này?**
    *   **Tính khách quan tuyệt đối (Zero Bias):** Biểu đồ sinh ra hoàn toàn từ dữ liệu giá số học thực tế, không có can thiệp cảm tính.
    *   **Chuẩn hóa hoàn hảo:** 100% ảnh có cùng tỷ lệ khung hình, cùng font chữ trục tọa độ, cùng bảng màu, không bị dính logo/watermark hay pop-up quảng cáo.
    *   **Tốc độ:** Một đoạn script có thể tạo 500 ảnh biểu đồ chỉ trong vòng 3 phút!

### Phương thức 2: Xuất Ảnh Thực nghiệm từ Nền tảng Phân tích (TradingView Snapshot Export)
*   **Cách thức:** 
    1. Đăng nhập TradingView, thiết lập biểu đồ theo chuẩn của đề tài (Khung 1D hoặc 4H, nến Nhật, chỉ báo EMA20, SMA50, Volume).
    2. Bấm nút **Take a snapshot (Alt + S)** $\rightarrow$ **Save chart image** để lưu ảnh độ phân giải cao.
    3. Đưa ảnh vào thư mục Drive `1_datasets/raw_images/`.
*   **Tại sao phương thức này có giá trị trong NCKH?**
    *   **Tính đại diện thực tế (Real-world Fidelity):** Biểu đồ mang đúng hình hài mà các nhà đầu tư và chuyên viên phân tích tài chính nhìn thấy hàng ngày.
    *   Phù hợp tuyệt vời để thu thập các mẫu hình kinh điển (Vai-Đầu-Vai, Hai Đáy, Nến Búa, Bẫy giá) phục vụ tập kiểm thử (Test Set) nhằm chứng minh năng lực thực chiến của mô hình.

### Phương thức 3: Thu thập Lai có Chuyên gia Thẩm định (Human-in-the-Loop Curation - Đề tài áp dụng)
*   **Cách thức:**
    1. Script tự động tải và vẽ 150 – 250 ảnh biểu đồ đại diện từ VN30, US Equities và Crypto.
    2. Gọi LLM sinh bản nháp chuỗi suy luận (CoT Draft).
    3. **Tác giả trực tiếp rà soát:** Lọc bỏ các ảnh có nến dị thường (do lỗi mất kết nối sàn), bổ sung thêm các trường hợp thị trường đặc thù (Flash crash, tin tức chiến tranh, biến động mạnh), và hiệu chỉnh nhãn CoT chuẩn kiến thức CMT.

---

## 3. BỐN QUY TẮC TOÀN VẸN DỮ LIỆU BẮT BUỘC TRONG NCKH TÀI CHÍNH

Khi thu thập và chuẩn bị dữ liệu, đề tài bắt buộc phải tuân thủ 4 quy tắc sau để bảo vệ tính hợp lệ của kết quả nghiên cứu:

1.  **Quy tắc 1: Phân chia theo Trục Thời gian (Time-series Split - CẤM RANDOM SPLIT):**
    *   $T_{\text{train}} (70\%) < T_{\text{val}} (15\%) < T_{\text{test}} (15\%)$.
    *   Tập Test bắt buộc phải là dữ liệu của tương lai chưa từng xuất hiện trong tập Train. Nếu dùng random split, mô hình sẽ "học vẹt" giá tương lai (Look-ahead bias), kết quả nghiên cứu sẽ bị Hội đồng bác bỏ.
2.  **Quy tắc 2: Điều chỉnh Chia tách và Cổ tức (Dividend & Stock Split Adjusted):**
    *   Đối với cổ phiếu (đặc biệt là VN30), bắt buộc phải dùng chuỗi giá đã điều chỉnh (Adjusted Close). Nếu dùng giá chưa điều chỉnh, các phiên chia cổ tức bằng cổ phiếu sẽ tạo ra khoảng trống giá giảm ảo (Gap Down) cực lớn làm mô hình phân tích sai.
3.  **Quy tắc 3: Khoảng đệm Nến (Purge / Embargo Window):**
    *   Giữa điểm cuối của tập Train và điểm đầu của tập Test phải có khoảng đệm tối thiểu 5 – 10 nến. Điều này ngăn chặn hiện tượng rò rỉ thông tin nến đang hình thành qua biên giới các tập dữ liệu.
4.  **Quy tắc 4: Khống chế Kích thước 512x512 pixels:**
    *   Tránh tải ảnh 2K/4K vào mô hình VLM vì sẽ làm bùng nổ số lượng visual tokens (hơn 1.500 tokens/ảnh), gây tràn 16GB VRAM trên Colab T4. Kích thước 512x512 vừa đủ độ nét cho từng thân và bóng nến mà chỉ tốn khoảng ~256 visual tokens.

---

## 4. BẢNG DANH MỤC TÀI SẢN THU THẬP ĐỀ XUẤT CHO ĐỀ TÀI

Để tập dữ liệu có tính tổng quát hóa (Generalizability) cao, đề tài nên thu thập cân đối trên 3 lớp tài sản:

| Nhóm tài sản | Các mã đại diện | Khung thời gian | Nguồn thu thập khuyến nghị | Số lượng ảnh mẫu |
| :--- | :--- | :---: | :--- | :---: |
| **1. VN30 (VN Equities)** | `VNINDEX`, `VCB`, `FPT`, `HPG`, `MWG`, `VIC` | `1D`, `4H` | `vnstock3` hoặc TradingView | 60 - 80 ảnh |
| **2. US Equities & ETFs** | `SPY` (S&P500), `QQQ` (Nasdaq), `AAPL`, `NVDA`, `TSLA` | `1D`, `4H` | `yfinance` hoặc TradingView | 60 - 80 ảnh |
| **3. Tiền mã hóa (Crypto)** | `BTCUSDT`, `ETHUSDT`, `SOLUSDT` | `1D`, `4H` | `ccxt` (Binance) hoặc TradingView | 40 - 60 ảnh |
| **TỔNG CỘNG** | | | | **160 - 220 ảnh** |
