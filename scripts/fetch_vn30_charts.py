"""Script chuyên biệt để tải dữ liệu và kết xuất biểu đồ VN30 trên Google Colab / Local.
Tự động kết nối Yahoo Finance (.VN) và vnstock3, vẽ biểu đồ chuẩn NCKH (512x512 Dark Mode)
và cập nhật trực tiếp vào file manifest.csv trên Google Drive.

Cách dùng trên Colab:
    !python scripts/fetch_vn30_charts.py --output-dir /content/drive/MyDrive/NCKH_AI/1_datasets/raw_images
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.chart_generator import detect_patterns, render_chart

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VN30_Fetcher")

# Danh sách mã VN30 thanh khoản cao nhất
VN30_SYMBOLS = ["VCB", "FPT", "HPG", "MWG", "VIC", "TCB"]


def fetch_vn30_ohlcv(symbol: str, start_date: str = "2022-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """Tải dữ liệu nến Ngày (1D) cho cổ phiếu Việt Nam với cơ chế Fallback tự động."""
    df = pd.DataFrame()

    # Cách 1: Thử lấy qua vnstock3 (nếu chạy local hoặc IP không bị chặn)
    try:
        from vnstock3 import Vnstock
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        raw = stock.quote.history(start=start_date, end=end_date, interval='1d')
        if raw is not None and not raw.empty:
            col_map = {'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
            if 'Date' in raw.columns:
                raw['Date'] = pd.to_datetime(raw['Date'])
                raw = raw.set_index('Date')
            df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            logger.info(f"[{symbol}] Tải thành công {len(df)} phiên qua vnstock3.")
    except Exception as e:
        logger.debug(f"[{symbol}] vnstock3 không khả dụng ({e}), chuyển sang yfinance.")

    # Cách 2: Fallback qua Yahoo Finance (.VN) chuẩn quốc tế (100% hoạt động trên Colab/Cloud)
    if df.empty:
        try:
            import yfinance as yf
            yf_symbol = f"{symbol}.VN"
            raw = yf.download(yf_symbol, start=start_date, end=end_date, interval="1d", progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                logger.info(f"[{symbol}] Tải thành công {len(df)} phiên qua Yahoo Finance ({yf_symbol}).")
        except Exception as e:
            logger.error(f"[{symbol}] Không thể tải dữ liệu: {e}")

    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


def generate_vn30_dataset(
    output_dir: str = "/content/drive/MyDrive/NCKH_AI/1_datasets/raw_images",
    symbols: List[str] = VN30_SYMBOLS,
    window_size: int = 60,
    stride: int = 30
):
    out_base = Path(output_dir)
    vn30_dir = out_base / "VN30"
    vn30_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_base / "manifest.csv"
    existing_manifest = pd.DataFrame()
    if manifest_path.exists():
        existing_manifest = pd.read_csv(manifest_path)
        logger.info(f"Đã nạp file manifest hiện có ({len(existing_manifest)} dòng).")

    new_rows = []
    total_rendered = 0

    print("=" * 70)
    print(f"🚀 BẮT ĐẦU KẾT XUẤT BIỂU ĐỒ NẾN VN30 VÀO: {vn30_dir}")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n[*] Đang xử lý mã: {symbol}...")
        df = fetch_vn30_ohlcv(symbol)
        if df.empty or len(df) < window_size:
            print(f"  ⚠️ Không đủ dữ liệu cho {symbol} ({len(df)} nến). Bỏ qua.")
            continue

        num_windows = (len(df) - window_size) // stride + 1
        symbol_count = 0

        for w in range(num_windows):
            sub_df = df.iloc[w * stride : w * stride + window_size]
            w_start = sub_df.index[0].strftime("%Y%m%d")
            w_end = sub_df.index[-1].strftime("%Y%m%d")

            img_name = f"{symbol}_1D_{w_start}_{w_end}.png"
            img_rel_path = f"VN30/{img_name}"
            img_full_path = vn30_dir / img_name

            detected = detect_patterns(sub_df)
            success = render_chart(sub_df, symbol, img_full_path)

            if success:
                symbol_count += 1
                total_rendered += 1
                new_rows.append({
                    "image_path": img_rel_path,
                    "symbol": symbol,
                    "asset_class": "VN30",
                    "timeframe": "1D",
                    "start_date": sub_df.index[0].strftime("%Y-%m-%d"),
                    "end_date": sub_df.index[-1].strftime("%Y-%m-%d"),
                    "candle_count": len(sub_df),
                    "detected_patterns": ";".join(detected) if detected else "none"
                })

        print(f"  ✔ Đã xuất: {symbol_count} ảnh cho mã {symbol}")

    # Cập nhật tệp manifest.csv
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not existing_manifest.empty:
            # Loại bỏ trùng lặp nếu đã có ảnh cùng đường dẫn
            combined_df = pd.concat([existing_manifest, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["image_path"], keep="last")
        else:
            combined_df = new_df

        combined_df.to_csv(manifest_path, index=False, encoding="utf-8")
        print("\n" + "=" * 70)
        print(f"🎉 HOÀN THÀNH: Đã xuất thêm {total_rendered} ảnh VN30 vào {vn30_dir}!")
        print(f"📁 Tổng số lượng ảnh ghi nhận trong manifest.csv: {len(combined_df)} ảnh.")
        print(f"📄 File manifest cập nhật tại: {manifest_path}")
        print("=" * 70)
    else:
        print("\n❌ Không có ảnh nào được xuất thêm.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VN30 Candlestick Chart Generator for Colab")
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/NCKH_AI/1_datasets/raw_images")
    args = parser.parse_args()
    generate_vn30_dataset(output_dir=args.output_dir)
