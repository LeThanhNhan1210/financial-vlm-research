# Phase 1 — Bước 2: Pipeline Kết xuất Tự động Ảnh Biểu đồ Nến

Script Python tự động tải OHLCV → vẽ biểu đồ nến Dark Mode 512×512 → xuất PNG, chạy trên cả máy local và Google Colab.

## Domain Knowledge: 19 Mẫu nến Target cho VLM

> [!IMPORTANT]
> Nguồn tổng hợp:
> - [DNSE — Các mẫu nến đảo chiều mạnh](https://www.dnse.com.vn/hoc/cac-mau-nen-dao-chieu-manh)
> - [Binance Academy — Hướng dẫn đọc biểu đồ nến](https://www.binance.com/vi/academy/articles/a-beginners-guide-to-candlestick-charts)
> - [Pinetree — Biểu đồ nến Nhật Candlestick](https://pinetree.vn/post/20210622/bieu-do-nen-nhat-candlestick-charting-cach-doc-phan-tich-mo-hinh-nen-va-y-nghia-cac-loai-nen-trong-phan-tich-ky-thuat-chung-khoan/)
>
> Pipeline thu thập ảnh phải đảm bảo tập dữ liệu **bao phủ đủ** các mẫu nến mà VLM cần nhận biết. Danh mục gồm **19 mẫu nến** chia thành 3 nhóm:

### Nhóm A: Nến cơ bản & Nến do dự (Fundamental & Indecision — xuất hiện ở mọi vị trí)

| # | Tên tiếng Anh | Tên tiếng Việt | Cấu trúc | Đặc điểm nhận dạng chính |
|---|---|---|---|---|
| 1 | **Marubozu** | Nến Cường lực | 1 nến | Thân rất dài, KHÔNG có bóng (hoặc cực ngắn). Phe mua/bán áp đảo hoàn toàn |
| 2 | **Spinning Top** | Nến Con quay | 1 nến | Thân nhỏ, bóng trên + dưới dài xấp xỉ nhau. Thị trường do dự, chưa rõ xu hướng |
| 3 | **Doji** (chuẩn) | Doji | 1 nến | Thân ≈ 0 (Open ≈ Close), bóng 2 bên cân bằng. Tín hiệu do dự mạnh |

### Nhóm B: Mẫu nến đảo chiều TĂNG (Bullish Reversal — xuất hiện ở đáy xu hướng giảm)

| # | Tên tiếng Anh | Tên tiếng Việt | Cấu trúc | Đặc điểm nhận dạng chính |
|---|---|---|---|---|
| 4 | **Dragonfly Doji** | Doji Chuồn chuồn | 1 nến | Thân gần bằng 0, bóng dưới rất dài, không bóng trên |
| 5 | **Bullish Engulfing** | Nhấn chìm tăng | 2 nến | Nến tăng bao trùm hoàn toàn nến giảm trước đó |
| 6 | **Hammer** | Nến Búa | 1 nến | Thân nhỏ, bóng dưới dài gấp 2-3× thân, không bóng trên |
| 7 | **Inverted Hammer** | Búa ngược | 1 nến | Thân nhỏ, bóng trên dài gấp 2× thân, bóng dưới nhỏ |
| 8 | **Morning Star** | Sao Mai | 3 nến | Nến giảm → Doji/Hammer nhỏ → Nến tăng mạnh |
| 9 | **Tweezer Bottom** | Đáy Nhíp | 2 nến | Nến giảm + nến tăng có giá thấp nhất bằng nhau |
| 10 | **Piercing Pattern** | Đường Nhọn | 2 nến | Nến giảm dài → nến tăng mở dưới đáy nhưng đóng trên 50% nến trước |
| 11 | **Three White Soldiers** | Ba chàng lính trắng | 3 nến | 3 nến tăng liên tiếp, thân dài, mỗi nến mở trong thân nến trước và đóng cao hơn |

### Nhóm C: Mẫu nến đảo chiều GIẢM (Bearish Reversal — xuất hiện ở đỉnh xu hướng tăng)

| # | Tên tiếng Anh | Tên tiếng Việt | Cấu trúc | Đặc điểm nhận dạng chính |
|---|---|---|---|---|
| 12 | **Gravestone Doji** | Doji Bia mộ | 1 nến | Bóng trên dài, thân gần bằng 0, không bóng dưới |
| 13 | **Bearish Engulfing** | Nhấn chìm giảm | 2 nến | Nến giảm bao trùm hoàn toàn nến tăng trước đó |
| 14 | **Hanging Man** | Người Treo cổ | 1 nến | Giống Hammer nhưng ở đỉnh xu hướng tăng |
| 15 | **Shooting Star** | Bắn Sao | 1 nến | Giống Inverted Hammer nhưng ở đỉnh xu hướng tăng |
| 16 | **Evening Star** | Sao Hôm | 3 nến | Nến tăng → Doji/Hammer nhỏ → Nến giảm mạnh |
| 17 | **Tweezer Top** | Đỉnh Nhíp | 2 nến | Nến tăng + nến giảm có giá cao nhất bằng nhau |
| 18 | **Dark Cloud Cover** | Mây đen bao phủ | 2 nến | Nến tăng dài → nến giảm mở trên đỉnh nhưng đóng dưới 50% nến trước (đối xứng Piercing) |
| 19 | **Three Black Crows** | Ba con quạ đen | 3 nến | 3 nến giảm liên tiếp, thân dài, mỗi nến mở trong thân nến trước và đóng thấp hơn (đối xứng Three White Soldiers) |

> [!NOTE]
> **Heikin-Ashi** (Binance Academy): Là biến thể biểu đồ nến dùng công thức trung bình hóa để làm mượt xu hướng. Pipeline hiện tại chỉ vẽ **nến Nhật truyền thống (OHLCV)** — đây là lựa chọn đúng vì VLM cần nhận biết cấu trúc thân/bóng nến thực tế, không phải nến đã biến đổi.

### Ý nghĩa cho Pipeline

- **Sliding window** sẽ quét toàn bộ dải thời gian → tự nhiên bắt được nhiều mẫu nến xuất hiện tại các vùng đảo xu hướng thực tế.
- **Manifest CSV** ghi thêm cột `detected_patterns` (list các mẫu nến phát hiện tự động bằng logic OHLCV đơn giản trên cửa sổ cuối) — phục vụ Phase 2 lọc ảnh theo pattern và cân bằng phân phối.
- **Bước 2.2 (HITL thủ công):** Tác giả rà soát và bổ sung thêm ảnh TradingView cho các mẫu hiếm (Morning Star, Evening Star, Three White Soldiers, Three Black Crows, Dark Cloud Cover) nếu pipeline tự động chưa bắt đủ.

---

## User Review Required

> [!IMPORTANT]
> **Thư viện mới cần thêm vào `requirements.txt`:**
> - `mplfinance` — wrapper trên matplotlib chuyên vẽ nến chuẩn tài chính.
> - `yfinance` — tải OHLCV cho US Equities.
> - `ccxt` — tải OHLCV cho Crypto từ Binance.
> - `vnstock3` — tải OHLCV cho VN30.

> [!WARNING]
> **`vnstock3` có thể bị rate-limit hoặc lỗi kết nối SSI/TCBS** khi chạy trên Colab (IP ngoài VN). Script sẽ `try/except` bỏ qua mã lỗi và log cảnh báo — bạn bổ sung thủ công từ TradingView sau (đúng phương thức HITL).

## Quyết định Thiết kế (Trả lời 3 câu hỏi mở)

| # | Câu hỏi | Quyết định đề xuất | Lý do |
|---|---|---|---|
| 1 | Số nến mỗi ảnh? | **60 nến** (giữ nguyên config) | Đủ ngữ cảnh xu hướng để nhận mẫu 3 nến (Morning/Evening Star) + vùng EMA |
| 2 | Bước trượt sliding window? | **30 nến** (overlap 50%) | ~6-8 ảnh/mã/timeframe → tổng ~168-224 ảnh, đúng mục tiêu 160-220 |
| 3 | Chỉ báo hiển thị? | **Chỉ EMA20 + Volume** | VLM cần tập trung nhận dạng thân nến và bóng nến (đặc điểm phân biệt 13 mẫu). RSI/BB/SMA50 gây nhiễu thị giác, thêm sau ở Phase 2 nếu cần |

## Proposed Changes

### Data Fetching & Chart Rendering Module

Tuân thủ Ponytail: **1 module duy nhất**, không factory, không abstract class. Đọc config từ `configs/dataset_config.yaml`.

#### [NEW] [`chart_generator.py`](file:///e:/financial-vlm-research/src/data/chart_generator.py)

Module chứa:
- `fetch_ohlcv(symbol, source, start, end, timeframe)` → `pd.DataFrame` — routing bằng `if/elif`: `yfinance` cho US, `ccxt` cho Crypto, `vnstock3` cho VN.
- `detect_patterns(df)` → `list[str]` — logic OHLCV đơn giản phát hiện 13 mẫu nến trên cửa sổ cuối (so sánh tỉ lệ thân/bóng, vị trí open/close). Không dùng thư viện ngoài, chỉ `pandas` arithmetic.
- `render_chart(df, symbol, output_path, config)` → lưu PNG 512×512 Dark Mode dùng `mplfinance.plot()` với EMA20 + Volume.
- `generate_all(config_path, output_dir)` → vòng lặp chính: đọc config → từng asset class → từng symbol → sliding window (bước 30 nến) → fetch → detect patterns → render → ghi manifest CSV.

---

### Runner Script

#### [NEW] [`scripts/generate_charts.py`](file:///e:/financial-vlm-research/scripts/generate_charts.py)

Entry point ~10 dòng, chỉ `import` + gọi `generate_all()`. Nhận `--config` và `--output-dir` từ CLI. **Không chứa logic xử lý** (tuân thủ §3 AGENTS.md).

---

### Dependencies

#### [MODIFY] [`requirements.txt`](file:///e:/financial-vlm-research/requirements.txt)

```diff
+# Financial Data Sources & Chart Rendering
+yfinance>=0.2.0
+mplfinance>=0.12.10b0
+ccxt>=4.0.0
+vnstock3>=0.3.0
```

---

### Manifest Output

Script tự động sinh `data/raw/manifest.csv`:

| Cột | Ý nghĩa |
|---|---|
| `image_path` | Đường dẫn tương đối đến ảnh PNG |
| `symbol` | Mã tài sản (VCB, SPY, BTCUSDT...) |
| `asset_class` | VN30 / US_Equities / Crypto |
| `timeframe` | 1D / 4H |
| `start_date` | Ngày bắt đầu cửa sổ nến |
| `end_date` | Ngày kết thúc cửa sổ nến |
| `candle_count` | Số nến thực tế trong ảnh |
| `detected_patterns` | Danh sách mẫu nến phát hiện tự động (VD: `hammer,bullish_engulfing`) |

Phục vụ trực tiếp cho Bước 2.3 (Time-series Split) và Bước 2.4 (CoT labeling).

---

## Verification Plan

### Automated Tests
- `python scripts/generate_charts.py --config configs/dataset_config.yaml --output-dir data/raw/`
- Kiểm tra: ≥1 ảnh PNG 512×512 được tạo cho mỗi asset class.
- Kiểm tra: `data/raw/manifest.csv` tồn tại, đúng schema, cột `detected_patterns` có giá trị.

### Manual Verification
- Mở vài ảnh PNG xác nhận: nền tối, nến rõ ràng, EMA20 vàng, volume panel bên dưới.
- Cross-check: ảnh có `detected_patterns=hammer` → nhìn thấy nến có bóng dưới dài.
- Ghi kết quả vào `docs/benchmarks/02_chart_generation.md` (theo skill `scientific-research-logging`).
