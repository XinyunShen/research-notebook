"""
Google Trends 搜索热度，用来替换 SOXX/SMH 成交量代理，修复
design_doc.md 12节问题11指出的 circularity：ETF成交量本身和部分被测
下游股票（AMAT/LRCX/KLAC/MU 等）重叠，用它构造"关注度"信号再拿去测
这些股票和信号的"共振"，存在自我印证风险。Google Trends 搜索关键词
（见下方 KEYWORDS）不含任何被测公司名/股票代码，是一个干净、不重叠的
关注度代理（design_doc.md 5.7节 Tier 4）。

**用的是非官方 pytrends 库**（Google没有官方公开的Trends API），有两个
比 Finnhub 更麻烦的已知限制：

1. **限流更严、更容易被临时封禁**——这里做了保守的请求间隔
   （REQUEST_SLEEP_SECONDS）和本地缓存（拿到一次就存 parquet），失败了
   就跳过那一段/那一整个关键词，不重试到死。

2. **历史区间越长，分辨率越粗**（这是Google Trends的官方规则，不是这份
   代码的bug，下面数字是本模块开发时实测验证过的）：
   - 请求窗口 <= 约9个月：每日分辨率
   - 请求窗口 9个月 ~ 约5.2年：每周分辨率
   - 请求窗口 > 约5.2年：每月分辨率（对我们要覆盖的2016-2026十年历史
     完全不够用，63/126日滚动窗口需要的粒度撑不起来）

   所以要拿到覆盖十年、可用的数据，只能把历史拆成多段 < 5.2年的窗口
   （每段能拿到周频），段与段之间留一段重叠期，用重叠期的均值比例把
   后一段缩放到和前一段同一个刻度上再拼接（Google Trends 每次查询都是
   独立按0-100归一化的，不同次查询的100不是同一个基准，直接拼接会在
   接缝处产生虚假跳变，必须用重叠区间做缩放）。拼出来的还是**周频**，
   reindex到交易日日历时用前向填充（ffill）近似日频——这是明确的已知
   简化，不是真日频数据，配合我们120+日的滚动窗口不算太粗糙。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

TRENDS_CACHE = Path(__file__).resolve().parent.parent / "data" / "google_trends.parquet"

# 第一版关键词是通用词（"semiconductor stocks"/"AI stocks"），换成
# 叙事专属词（"AI data center"等3个）后IC从0.02回升到0.08（design_doc.md
# 12节问题16）。这一版是**第二次扩充**，起因不是"上次结果不好看再试试
# 别的"——上次结果已经不错——而是用户指出3.2.1节产业链实际覆盖的层级
# 比原来3个关键词覆盖的窄：L2（设备/材料）完全没有对应词，L3只覆盖了
# "液冷"没覆盖同样真实的"人造钻石散热"这条2026年的新兴趋势（用WebSearch
# 核实过：AI芯片功耗超2000W、热流密度超1000W/cm²逼近传统铜散热极限，
# CVD人造钻石导热率是铜的5倍，2026年已进入8寸晶圆量产阶段，市场规模从
# 2亿美元冲向4.3亿美元(2035)，是真实、可查证的产业趋势，不是编的）。
#
# 新增4个、保留2个（"AI data center"这个太笼统、和新增的关键词功能重叠，
# 这版拿掉了）：
# - L2（设备/材料，ASML/AMAT/LRCX/KLAC/MU/AMKR）："semiconductor
#   equipment"（设备产能扩张）、"HBM memory"（2026年公认的AI供应链
#   核心瓶颈）、"chip packaging"（先进封装，TSMC/SK海力士都在砸钱扩产）
# - L3（材料/散热，VRT/MOD/NVT/ETN/ENTG/MKSI/LIN/APD）："data center
#   cooling"（保留）、"diamond heat spreader"（人造钻石散热，新增）
# - L4（电力，GEV/ETN/PWR/CEG/VST/BE）："data center power"（保留）、
#   "AI power demand"（新增）
#
# **这依然是基于叙事逻辑一次性扩充的，不是试了很多组合挑IC最好看的那组**
# ——扩充理由是"覆盖面不完整"，不是"数字不好看"，扩完之后不管结果如何都
# 如实记录，这一轮之后不再继续增删关键词去追更好看的数字（design_doc.md
# 2.4节过拟合警告）。
#
# 依然特意不含任何L1/下游候选股的公司名或股票代码，避免重蹈ETF成交量的
# circularity覆辙（design_doc.md 12节问题11）。
KEYWORDS = [
    "semiconductor equipment", "HBM memory", "chip packaging",
    "data center cooling", "diamond heat spreader",
    "data center power", "AI power demand",
]

CHUNK_DAYS = 1600  # ~4.4年，留出安全边际，确保每段查询都落在"周频"区间（<~5.2年阈值）
OVERLAP_DAYS = 90
REQUEST_SLEEP_SECONDS = 5  # pytrends比Finnhub更容易被限流/封禁，间隔留大一点
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 30  # 实测429后短暂重试没用，pytrends社区经验是要等半分钟以上


def _date_chunks(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), end_ts)
        chunks.append((chunk_start, chunk_end))
        if chunk_end >= end_ts:
            break
        chunk_start = chunk_end - pd.Timedelta(days=OVERLAP_DAYS)
    return chunks


def _fetch_chunk(pytrends: TrendReq, keyword: str, chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> pd.Series:
    """限流(429)是pytrends的常态，不是偶发异常，所以这里做了指数退避重试
    （最多 MAX_RETRIES 次），而不是失败一次就放弃这一段历史。"""
    timeframe = f"{chunk_start.date()} {chunk_end.date()}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pytrends.build_payload([keyword], timeframe=timeframe)
            df = pytrends.interest_over_time()
            time.sleep(REQUEST_SLEEP_SECONDS)
            if df.empty or keyword not in df.columns:
                return pd.Series(dtype=float)
            return df[keyword].astype(float)
        except Exception as e:  # pytrends对429/网络错误的封装不统一，宁可粗一点捕获也不让整条流水线崩掉
            is_last = attempt == MAX_RETRIES
            print(f"[google_trends] WARNING: 拉取 '{keyword}' {timeframe} 第{attempt}次尝试失败: {e}"
                  f"{'，放弃这一段' if is_last else f'，{RETRY_BACKOFF_SECONDS}秒后重试'}")
            if is_last:
                return pd.Series(dtype=float)
            time.sleep(RETRY_BACKOFF_SECONDS)
    return pd.Series(dtype=float)


def _stitch_chunks(pytrends: TrendReq, keyword: str, chunks: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    """按**从近到远**的顺序处理（chunks本身是从远到近生成的，这里reversed），
    让最新、最相关的一段先锚定拼接的刻度基准。这样如果中途被限流放弃后面的
    段，丢的是更早的历史，不是我们最在乎的近期数据（pytrends限流后经常就
    没法继续了，见 _fetch_chunk 的重试逻辑）。"""
    stitched: pd.Series | None = None
    for i, (chunk_start, chunk_end) in enumerate(reversed(chunks)):
        series = _fetch_chunk(pytrends, keyword, chunk_start, chunk_end)
        if series.empty:
            print(f"[google_trends] WARNING: '{keyword}' 第{i}段({chunk_start.date()}~{chunk_end.date()})无数据，跳过")
            continue

        if stitched is None:
            stitched = series
            continue

        overlap_idx = stitched.index.intersection(series.index)
        if len(overlap_idx) == 0:
            print(f"[google_trends] WARNING: '{keyword}' 第{i}段和前一段没有重叠日期，无法缩放拼接，丢弃这段")
            continue

        prev_mean = stitched.loc[overlap_idx].mean()
        new_mean = series.loc[overlap_idx].mean()
        if new_mean == 0:
            print(f"[google_trends] WARNING: '{keyword}' 第{i}段重叠区间均值为0，无法算缩放比例，丢弃这段")
            continue

        scale = prev_mean / new_mean
        series_scaled = series * scale
        new_part = series_scaled[~series_scaled.index.isin(stitched.index)]
        stitched = pd.concat([stitched, new_part]).sort_index()

    return stitched if stitched is not None else pd.Series(dtype=float)


def fetch_trends_series(keyword: str, start: str, end: str, force_refresh: bool = False,
                         cache_path: Path | None = None) -> pd.Series:
    """返回某个关键词的搜索热度（周频，index=date, 0-100，跨多段拼接缩放过）。
    整段拉取/拼接失败时返回空 Series，调用方据此判断要不要跳过Google Trends
    这一路信号、退回旧的ETF成交量信号（见 theme_index.py / pipeline.py）。
    """
    cache_path = cache_path or TRENDS_CACHE
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        if keyword in cached.columns:
            return cached[keyword].dropna()

    pytrends = TrendReq(hl="en-US", tz=360)
    chunks = _date_chunks(pd.Timestamp(start), pd.Timestamp(end))
    stitched = _stitch_chunks(pytrends, keyword, chunks)
    if stitched.empty:
        return stitched

    stitched = stitched.rename(keyword)
    _save_to_cache(keyword, stitched, cache_path)
    return stitched


def _save_to_cache(keyword: str, series: pd.Series, cache_path: Path) -> None:
    """**易踩的坑**：`existing[keyword] = series` 这种写法是按 `existing`
    当前的索引对齐赋值，不会把 `series` 里索引不存在的日期并进来——如果
    `existing` 是之前一次被限流截断的旧缓存（索引更短），新一次成功拿到
    的完整历史会被静默截断回旧索引的范围。这里用 `join(how="outer")`
    显式取索引并集，避免这个问题（本模块开发时实测踩到过这个坑：第一次
    因429只拿到2016-2024，之后重跑拿到完整2016-2026，却被这种写法悄悄
    截断回2016-2024，靠人工核对histogram/日期范围才发现）。
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series = series.rename(keyword)
    if cache_path.exists():
        existing = pd.read_parquet(cache_path).drop(columns=[keyword], errors="ignore")
        combined = existing.join(series, how="outer")
    else:
        combined = series.to_frame()
    combined = combined.sort_index()
    combined.to_parquet(cache_path)


def build_combined_trends_series(keywords: list[str], start: str, end: str) -> pd.Series | None:
    """等权合成多个关键词的搜索热度（周频，未reindex到交易日，见
    theme_index.google_trends_component 做reindex+ffill+滚动z-score）。
    所有关键词都拉取失败时返回 None，调用方据此整体跳过Google Trends信号。
    """
    series_list = []
    for kw in keywords:
        s = fetch_trends_series(kw, start, end)
        if s.empty:
            print(f"[google_trends] '{kw}' 没有取到任何数据，跳过这个关键词")
            continue
        series_list.append(s)

    if not series_list:
        return None
    combined = pd.concat(series_list, axis=1)
    return combined.mean(axis=1, skipna=True)
