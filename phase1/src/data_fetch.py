"""
行情数据获取（yfinance），对应 design_doc.md 第5.1节。
本地缓存成 parquet，避免每次重跑都重新打 API（yfinance 免费源不稳定，
design_doc.md 第9节已提示过这个风险，缓存也是缓解手段之一）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICE_CACHE = DATA_DIR / "prices.parquet"


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    force_refresh: bool = False,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """下载多只股票的日线 OHLCV，返回长表：columns=[date, ticker, open, high, low, close, adj_close, volume]

    cache_path: 默认写 phase1 自己的缓存文件；phase2 等其他调用方可以传入
    自己的缓存路径，避免互相覆盖对方缓存的股票池（比如 phase2 需要额外拉
    SOXX/SMH 这类 phase1 股票池里没有的 ticker）。
    """
    cache_path = cache_path or PRICE_CACHE
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        cached_tickers = set(cached["ticker"].unique())
        if set(tickers).issubset(cached_tickers):
            return cached[cached["ticker"].isin(tickers)].copy()

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    frames = []
    for t in tickers:
        try:
            sub = raw[t].copy()
        except KeyError:
            print(f"[data_fetch] WARNING: no data returned for {t}, skipping")
            continue
        sub = sub.rename(
            columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
            }
        )
        sub["ticker"] = t
        sub = sub.reset_index().rename(columns={"Date": "date"})
        frames.append(sub[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])

    long_df = pd.concat(frames, ignore_index=True).dropna(subset=["adj_close"])
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df.to_parquet(cache_path, index=False)
    return long_df
