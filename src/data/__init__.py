"""Data loading and preprocessing utilities."""
from .dataset import FinancialChartDataset
from .preprocessor import ChartPreprocessor

try:
    from .chart_generator import fetch_ohlcv, detect_patterns, render_chart, generate_all
except ImportError:
    fetch_ohlcv = detect_patterns = render_chart = generate_all = None

try:
    from .splitter import split_manifest, split_series_chronological
except ImportError:
    split_manifest = split_series_chronological = None

__all__ = [
    "FinancialChartDataset",
    "ChartPreprocessor",
    "fetch_ohlcv",
    "detect_patterns",
    "render_chart",
    "generate_all",
    "split_manifest",
    "split_series_chronological",
]



