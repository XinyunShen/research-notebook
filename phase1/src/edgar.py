"""
SEC EDGAR 基本面数据获取，严格保留 `filed`（申报日期）字段用于事后做
point-in-time 对齐 —— 这是 design_doc.md 第2.5节"前视偏差"要求的核心工程手段：
任何一天的因子计算，只能用"那一天已经公开申报"的财务数据，不能用之后才
披露/修正的数字。

数据源：data.sec.gov 官方 XBRL API，完全免费，见 design_doc.md 第5.2节。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "edgar_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SEC 要求请求头带可识别的联系方式，见 https://www.sec.gov/os/webmaster-faq#developers
HEADERS = {"User-Agent": "Personal quant research project alyssa.xinyun@gmail.com"}

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_TICKER_MAP_CACHE = CACHE_DIR / "company_tickers.json"

_RATE_LIMIT_SECONDS = 0.15  # SEC 限制 <=10 req/s，留足余量


def _sleep():
    time.sleep(_RATE_LIMIT_SECONDS)


def load_ticker_to_cik() -> dict[str, str]:
    """下载/读取官方 ticker -> CIK 映射表，本地缓存。"""
    if _TICKER_MAP_CACHE.exists():
        raw = json.loads(_TICKER_MAP_CACHE.read_text())
    else:
        r = requests.get(_TICKER_MAP_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        raw = r.json()
        _TICKER_MAP_CACHE.write_text(json.dumps(raw))
        _sleep()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}


def fetch_company_facts(cik: str) -> dict | None:
    """拉取单家公司全部 XBRL facts（us-gaap + dei），本地缓存 JSON。"""
    cache_file = CACHE_DIR / f"{cik}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    _sleep()
    if r.status_code != 200:
        return None
    data = r.json()
    cache_file.write_text(json.dumps(data))
    return data


def _extract_instant(facts: dict, tag: str) -> pd.DataFrame:
    """提取某个时点型（instant）字段，比如股东权益 StockholdersEquity。
    返回列：end（财报截止日）、val、filed（申报日，用于as-of对齐）。
    """
    try:
        entries = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return pd.DataFrame(columns=["end", "val", "filed", "form"])
    df = pd.DataFrame(entries)
    df = df[df["form"].isin(["10-Q", "10-K"])]
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df[["end", "val", "filed", "form"]].sort_values("filed")


def _extract_single_quarter_duration(facts: dict, tag: str) -> pd.DataFrame:
    """提取区间型（duration）字段的"单季度"数值，比如净利润 NetIncomeLoss。

    XBRL 里 10-Q 同时会报"当季"和"年初至今累计"两种区间，必须按区间长度
    （约80-100天=单季度）过滤，否则会把累计值错当成单季度值，算出来的
    TTM（滚动12个月）净利润会严重失真。
    """
    try:
        entries = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "form"])
    df = pd.DataFrame(entries)
    df = df[df["form"].isin(["10-Q", "10-K"])]
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    days = (df["end"] - df["start"]).dt.days
    is_single_quarter = days.between(75, 100)
    is_annual = days.between(350, 380)
    df = df[is_single_quarter | is_annual].copy()
    return df[["start", "end", "val", "filed", "form"]].sort_values("filed")


MIN_PLAUSIBLE_SHARES_OUTSTANDING = 10_000  # 见下方注释，过滤"壳公司占位股数"这类脏数据


def _extract_shares_outstanding(facts: dict) -> pd.DataFrame:
    """**踩过的坑**：新设立/刚完成分拆重组的公司，第一次XBRL申报时
    EntityCommonStockSharesOutstanding 有时会是一个占位数字（比如
    DXC Technology 2017年从CSC/HPE分拆重组后的第一份申报，股数报的是
    "100"，真实股数要等几个月后正式股权发行完成才更新到2.8亿）。这个
    脏数据会被 merge_asof 一直carry forward到下一次申报之前，导致
    market_cap = 100 * 股价 ≈ 0，book_to_market 算出天文数字（实测
    出现过28742倍这种荒谬值，还会因为cross-sectional winsorize用的是
    同一天的分位数边界，连累同一天其他正常股票的B/M也被错误地钳到这个
    脏数据设的上限）。真实上市公司不可能只有几万股以下的流通股（哪怕是
    伯克希尔A类股每股70万美元、股数最少的极端案例也有60万股以上），
    所以直接用一个远低于任何真实案例的下限（1万股）过滤掉这类占位值，
    当缺失处理，让 merge_asof 往前找上一个真实值（或者在没有更早真实值
    时保持缺失，而不是硬用一个荒谬值算出来的比率）。
    """
    try:
        entries = facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]
    except KeyError:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df = pd.DataFrame(entries)
    if df.empty:
        return df
    df = df[df["val"] >= MIN_PLAUSIBLE_SHARES_OUTSTANDING]
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df[["end", "val", "filed"]].sort_values("filed")


def build_fundamentals_panel(ticker: str, cik: str) -> pd.DataFrame | None:
    """为单只股票构建"逐次披露事件"的基本面面板，每一行代表一次新的
    财报数据点，附带 filed 日期，供 factors.py 用 merge_asof 做
    point-in-time 对齐。

    输出列：filed, stockholders_equity, net_income_ttm, shares_outstanding
    """
    facts = fetch_company_facts(cik)
    if facts is None:
        return None

    se = _extract_instant(facts, "StockholdersEquity")
    ni = _extract_single_quarter_duration(facts, "NetIncomeLoss")
    so = _extract_shares_outstanding(facts)

    if se.empty or ni.empty:
        return None

    # TTM 净利润 = 最近4个"单季度"数值之和（按 end 日期排序后滚动求和）
    ni_q = ni.drop_duplicates(subset=["end"], keep="last").sort_values("end")
    ni_q["net_income_ttm"] = ni_q["val"].rolling(4, min_periods=4).sum()
    ni_q = ni_q.dropna(subset=["net_income_ttm"])[["filed", "end", "net_income_ttm"]]

    se_q = se.drop_duplicates(subset=["end"], keep="last")[["filed", "end", "val"]].rename(
        columns={"val": "stockholders_equity"}
    )

    panel = pd.merge_asof(
        se_q.sort_values("filed"),
        ni_q.sort_values("filed")[["filed", "net_income_ttm"]],
        on="filed",
        direction="backward",
    )

    if not so.empty:
        so_q = so.drop_duplicates(subset=["filed"])[["filed", "val"]].rename(
            columns={"val": "shares_outstanding"}
        )
        panel = pd.merge_asof(
            panel.sort_values("filed"), so_q.sort_values("filed"), on="filed", direction="backward"
        )
    else:
        panel["shares_outstanding"] = float("nan")

    # 股东权益为负/为零时 ROE 没有经济意义（常见于大量股票回购的科技公司），
    # 学术上（Fama-French）标准做法是直接剔除负账面权益公司，而不是硬算出
    # 一个畸形的极端比率——否则会产生 inf/-inf，污染整个因子分布。
    # 用 np.nan（而不是 pd.NA）填充，保证列是纯 float64，兼容 numpy/alphalens
    # 的 isfinite 等运算；pd.NA 会把列变成 object dtype 导致下游报错。
    positive_equity = panel["stockholders_equity"] > 0
    panel["roe_ttm"] = float("nan")
    panel.loc[positive_equity, "roe_ttm"] = (
        panel.loc[positive_equity, "net_income_ttm"] / panel.loc[positive_equity, "stockholders_equity"]
    )
    panel["roe_ttm"] = panel["roe_ttm"].astype("float64")
    return panel.sort_values("filed").reset_index(drop=True)
