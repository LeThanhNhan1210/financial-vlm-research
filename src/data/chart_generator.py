"""Chart data fetching, pattern detection, and candlestick chart rendering module.

Complies with AGENTS.md and ponytail principles (minimalist, stdlib-first, robust).
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbol: str,
    asset_class: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1D"
) -> pd.DataFrame:
    """Fetch OHLCV data for given symbol and asset class.

    Returns DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    and DatetimeIndex. Returns empty DataFrame on failure.
    """
    df = pd.DataFrame()
    try:
        if asset_class == "US_Equities":
            import yfinance as yf
            interval = "1d" if timeframe.upper() == "1D" else "1h"
            raw = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        elif asset_class == "Crypto":
            import ccxt
            exchange = ccxt.binance({'enableRateLimit': True})
            tf = "1d" if timeframe.upper() == "1D" else "4h"
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=1000)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('Date').drop(columns=['timestamp'])
                end_ts = pd.Timestamp(end_date)
                df = df[df.index <= end_ts]

        elif asset_class == "VN30":
            try:
                from vnstock3 import Vnstock
                stock = Vnstock().stock(symbol=symbol, source='VCI')
                raw = stock.quote.history(start=start_date, end=end_date, interval=timeframe.lower())
                if raw is not None and not raw.empty:
                    col_map = {'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
                    if 'Date' in raw.columns:
                        raw['Date'] = pd.to_datetime(raw['Date'])
                        raw = raw.set_index('Date')
                    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception as vn_err:
                logger.warning(f"vnstock3 error for {symbol}: {vn_err}. (Manual TradingView HITL fallback enabled)")

    except Exception as e:
        logger.error(f"Failed to fetch {symbol} ({asset_class}): {e}")

    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


def detect_patterns(df: pd.DataFrame) -> List[str]:
    """Detect presence of 19 target candlestick patterns in the latest candles of window.

    Patterns:
      Nhóm A: marubozu, spinning_top, doji
      Nhóm B: dragonfly_doji, bullish_engulfing, hammer, inverted_hammer,
              morning_star, tweezer_bottom, piercing_pattern, three_white_soldiers
      Nhóm C: gravestone_doji, bearish_engulfing, hanging_man, shooting_star,
              evening_star, tweezer_top, dark_cloud_cover, three_black_crows
    """
    if len(df) < 5:
        return []

    patterns = []
    c = df.iloc[-1]
    p = df.iloc[-2]
    p2 = df.iloc[-3] if len(df) >= 3 else None

    # Helper metrics for current candle c
    rng = c['High'] - c['Low']
    if rng <= 0:
        return []
    body = abs(c['Close'] - c['Open'])
    upper_shadow = c['High'] - max(c['Open'], c['Close'])
    lower_shadow = min(c['Open'], c['Close']) - c['Low']
    is_bull = c['Close'] >= c['Open']

    # Prior candle metrics
    p_rng = p['High'] - p['Low']
    p_body = abs(p['Close'] - p['Open'])
    p_is_bull = p['Close'] >= p['Open']

    # Trend context (short-term 5-candle slope)
    prior_trend = df['Close'].iloc[-6:-1].diff().sum() if len(df) >= 6 else 0
    in_downtrend = prior_trend < 0
    in_uptrend = prior_trend > 0

    # 1. Nhóm A: Nến cơ bản & Do dự
    if body >= 0.85 * rng and (upper_shadow + lower_shadow) <= 0.15 * rng:
        patterns.append("marubozu")

    if body <= 0.3 * rng and upper_shadow >= 0.25 * rng and lower_shadow >= 0.25 * rng:
        if abs(upper_shadow - lower_shadow) / rng <= 0.3:
            patterns.append("spinning_top")

    if body <= 0.1 * rng and upper_shadow >= 0.1 * rng and lower_shadow >= 0.1 * rng:
        patterns.append("doji")

    # 2. Nhóm B: Đảo chiều TĂNG (Bullish Reversal)
    if body <= 0.1 * rng and lower_shadow >= 0.6 * rng and upper_shadow <= 0.1 * rng:
        patterns.append("dragonfly_doji")

    if not p_is_bull and is_bull and c['Open'] <= p['Close'] and c['Close'] >= p['Open'] and body > p_body:
        patterns.append("bullish_engulfing")

    if lower_shadow >= 2.0 * body and upper_shadow <= 0.15 * rng and in_downtrend:
        patterns.append("hammer")

    if upper_shadow >= 2.0 * body and lower_shadow <= 0.15 * rng and in_downtrend:
        patterns.append("inverted_hammer")

    if (not p_is_bull) and is_bull and p_body >= 0.4 * p_rng:
        if c['Open'] < p['Low'] and c['Close'] > (p['Open'] + p['Close']) / 2:
            patterns.append("piercing_pattern")

    if p_rng > 0 and abs(c['Low'] - p['Low']) / p_rng <= 0.05 and not p_is_bull and is_bull:
        patterns.append("tweezer_bottom")

    if p2 is not None:
        p2_body = abs(p2['Close'] - p2['Open'])
        p2_rng = p2['High'] - p2['Low']
        # Morning Star: Bearish long -> Small star -> Bullish strong
        if (not p2['Close'] >= p2['Open']) and p2_body >= 0.4 * p2_rng:
            if p_body <= 0.3 * p_rng and is_bull and c['Close'] > (p2['Open'] + p2['Close']) / 2:
                patterns.append("morning_star")

        # Three White Soldiers: 3 consecutive bullish long candles
        c1, c2, c3 = p2, p, c
        if (c1['Close'] > c1['Open']) and (c2['Close'] > c2['Open']) and (c3['Close'] > c3['Open']):
            if c1['Close'] < c2['Close'] < c3['Close'] and c2['Open'] > c1['Open'] and c3['Open'] > c2['Open']:
                patterns.append("three_white_soldiers")

    # 3. Nhóm C: Đảo chiều GIẢM (Bearish Reversal)
    if body <= 0.1 * rng and upper_shadow >= 0.6 * rng and lower_shadow <= 0.1 * rng:
        patterns.append("gravestone_doji")

    if p_is_bull and (not is_bull) and c['Open'] >= p['Close'] and c['Close'] <= p['Open'] and body > p_body:
        patterns.append("bearish_engulfing")

    if lower_shadow >= 2.0 * body and upper_shadow <= 0.15 * rng and in_uptrend:
        patterns.append("hanging_man")

    if upper_shadow >= 2.0 * body and lower_shadow <= 0.15 * rng and in_uptrend:
        patterns.append("shooting_star")

    if p_is_bull and (not is_bull) and p_body >= 0.4 * p_rng:
        if c['Open'] > p['High'] and c['Close'] < (p['Open'] + p['Close']) / 2:
            patterns.append("dark_cloud_cover")

    if p_rng > 0 and abs(c['High'] - p['High']) / p_rng <= 0.05 and p_is_bull and not is_bull:
        patterns.append("tweezer_top")

    if p2 is not None:
        p2_body = abs(p2['Close'] - p2['Open'])
        p2_rng = p2['High'] - p2['Low']
        # Evening Star: Bullish long -> Small star -> Bearish strong
        if (p2['Close'] >= p2['Open']) and p2_body >= 0.4 * p2_rng:
            if p_body <= 0.3 * p_rng and (not is_bull) and c['Close'] < (p2['Open'] + p2['Close']) / 2:
                patterns.append("evening_star")

        # Three Black Crows: 3 consecutive bearish long candles
        c1, c2, c3 = p2, p, c
        if (c1['Close'] < c1['Open']) and (c2['Close'] < c2['Open']) and (c3['Close'] < c3['Open']):
            if c1['Close'] > c2['Close'] > c3['Close'] and c2['Open'] < c1['Open'] and c3['Open'] < c2['Open']:
                patterns.append("three_black_crows")

    return patterns


def render_chart(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path,
    image_size: Tuple[int, int] = (512, 512)
) -> bool:
    """Render candlestick chart with EMA20 and Volume panel in Dark Mode 512x512."""
    import mplfinance as mpf
    from PIL import Image

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Custom TradingView Dark Style
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge={'up': '#26a69a', 'down': '#ef5350'},
            wick={'up': '#26a69a', 'down': '#ef5350'},
            volume={'up': '#1e5e54', 'down': '#782d2b'}
        )
        style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=mc,
            facecolor='#131722',
            edgecolor='#2a2e39',
            figcolor='#131722',
            gridcolor='#1e222d',
            gridstyle='--'
        )

        fig, axlist = mpf.plot(
            df,
            type='candle',
            mav=(20,),
            volume=True,
            style=style,
            returnfig=True,
            figsize=(5.12, 5.12)
        )
        # Style EMA line to bright yellow
        if len(axlist) > 0 and len(axlist[0].lines) > 0:
            for line in axlist[0].lines:
                line.set_color('#f1c40f')
                line.set_linewidth(1.2)

        fig.savefig(str(output_path), dpi=100, facecolor='#131722')
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Ensure exact 512x512 pixel resolution
        with Image.open(output_path) as im:
            if im.size != image_size:
                im = im.resize(image_size, Image.Resampling.LANCZOS)
                im.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Error rendering chart {output_path}: {e}")
        return False


def generate_all(config_path: str, output_dir: str) -> str:
    """Main pipeline execution: fetch, slice sliding windows, detect patterns, render charts, save manifest."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    candlestick_count = cfg.get("chart_rendering", {}).get("candlestick_count", 60)
    stride = 30  # 50% overlap per implementation plan
    start_date = cfg.get("collection", {}).get("date_range", {}).get("start_date", "2022-01-01")
    end_date = cfg.get("collection", {}).get("date_range", {}).get("end_date", "2025-12-31")

    for asset_group in cfg.get("collection", {}).get("asset_classes", []):
        asset_class = asset_group["name"]
        symbols = asset_group.get("symbols", [])
        timeframes = asset_group.get("timeframes", ["1D"])

        for symbol in symbols:
            for tf in timeframes:
                logger.info(f"Processing {asset_class} | {symbol} | {tf}...")
                df = fetch_ohlcv(symbol, asset_class, start_date, end_date, tf)
                if df.empty or len(df) < candlestick_count:
                    logger.warning(f"Insufficient data for {symbol} ({len(df)} rows). Skipping.")
                    continue

                # Sliding window
                num_windows = (len(df) - candlestick_count) // stride + 1
                for w in range(num_windows):
                    sub_df = df.iloc[w * stride : w * stride + candlestick_count]
                    w_start = sub_df.index[0].strftime("%Y%m%d")
                    w_end = sub_df.index[-1].strftime("%Y%m%d")

                    detected = detect_patterns(sub_df)
                    img_name = f"{symbol}_{tf}_{w_start}_{w_end}.png"
                    img_rel_path = f"{asset_class}/{img_name}"
                    img_full_path = out_base / asset_class / img_name

                    success = render_chart(sub_df, symbol, img_full_path)
                    if success:
                        manifest_rows.append({
                            "image_path": img_rel_path,
                            "symbol": symbol,
                            "asset_class": asset_class,
                            "timeframe": tf,
                            "start_date": sub_df.index[0].strftime("%Y-%m-%d"),
                            "end_date": sub_df.index[-1].strftime("%Y-%m-%d"),
                            "candle_count": len(sub_df),
                            "detected_patterns": ";".join(detected) if detected else "none"
                        })

    manifest_path = out_base / "manifest.csv"
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
    logger.info(f"Generation completed: {len(manifest_rows)} charts rendered. Manifest saved to {manifest_path}")
    return str(manifest_path)
