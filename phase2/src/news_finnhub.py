"""
可选的新闻热度数据源（design_doc.md 5.7节 Tier 3：Finnhub 免费层）。

MVP阶段（design_doc.md 5.7节"MVP数据源决定"）默认不接付费源，Finnhub
免费层本身也需要个人注册一个免费 API Key（https://finnhub.io/register），
这一步不能替用户完成，所以这里做成可插拔：

- 设置了环境变量 FINNHUB_API_KEY 就会启用，新闻提及量趋势会加入主题
  热度指数（见 theme_index.py）。
- 没设置就跳过，主题热度指数退化成只用"L1核心股动量 + ETF成交量异动"
  两个纯行情数据源，仍然能跑，只是少一路信号。

这正是 design_doc.md 5.7节承诺的"新闻数据接入层做成可插拔，以后无缝升级"。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
NEWS_CACHE = Path(__file__).resolve().parent.parent / "data" / "news_counts.parquet"


def api_key_available() -> bool:
    return bool(os.environ.get("FINNHUB_API_KEY"))


def fetch_daily_news_count(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """返回某只股票每天的新闻条数，用作"热度趋势"的原始输入。
    Finnhub 免费层限流较严（约60次/分钟），调用间隔要留够。
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return None

    r = requests.get(
        f"{FINNHUB_BASE}/company-news",
        params={"symbol": ticker, "from": start, "to": end, "token": api_key},
        timeout=15,
    )
    time.sleep(1.1)  # 免费层限流保守处理
    if r.status_code != 200:
        print(f"[news_finnhub] WARNING: {ticker} 请求失败 status={r.status_code}")
        return None

    articles = r.json()
    if not articles:
        return pd.DataFrame(columns=["date", "news_count"])

    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.normalize()
    daily_counts = df.groupby("date").size().rename("news_count").reset_index()
    return daily_counts


def fetch_news_for_tickers(
    tickers: list[str], start: str, end: str, cache_path: Path | None = None, force_refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """批量拉取一组股票的每日新闻条数，本地缓存成 parquet（长表：
    ticker, date, news_count），避免每次重跑都重新打 Finnhub 免费层
    （限流约60次/分钟），和 phase1 data_fetch.py 的缓存风格保持一致。

    只在这里限定 tickers（调用方传 L1_TICKERS，见 pipeline.py）是有意的：
    新闻热度信号如果拉的是被测下游股票自己的新闻量，会和 circularity
    问题（12节问题11）犯同一种错误——用一只股票自己的信号去预测它自己的
    收益。L1（NVDA/AMD/AVGO/TSM）是"AI叙事"的公认源头，不在下游候选池
    `DOWNSTREAM_TICKERS` 里，拉它们的新闻量和拉它们的价格动量
    （l1_momentum_component）道理一致，都是测"故事源头有多热"，不是
    测被测股票自己。
    """
    cache_path = cache_path or NEWS_CACHE
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        cached_tickers = set(cached["ticker"].unique())
        if set(tickers).issubset(cached_tickers):
            return {
                t: cached[cached["ticker"] == t][["date", "news_count"]].reset_index(drop=True)
                for t in tickers
            }

    frames = []
    result = {}
    for t in tickers:
        daily = fetch_daily_news_count(t, start, end)
        if daily is None:
            print(f"[news_finnhub] WARNING: {t} 未取到新闻数据，跳过")
            continue
        result[t] = daily
        tagged = daily.copy()
        tagged["ticker"] = t
        frames.append(tagged)

    if frames:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(cache_path, index=False)

    return result
