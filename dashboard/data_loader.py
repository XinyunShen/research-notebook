"""
数据读取层，给 dashboard 各页面共用。只读本地已经跑出来的 parquet/csv/json
文件（phase1/phase2 的 output、data 目录），不会主动去联网重新拉数据——
这个平台是"看数据"的工具，不是"拉数据"的工具，两者职责分开。

用 st.cache_data 缓存读取结果，避免每次换页/换筛选条件都重新读一遍磁盘，
文件改了（比如重新跑了一遍 pipeline）后手动点 Streamlit 右上角的
"Rerun"/刷新页面会重新读取最新内容。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT / "phase1"
PHASE2 = ROOT / "phase2"


@st.cache_data
def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PHASE1 / "data" / "prices.parquet")


@st.cache_data
def load_factor_panel() -> pd.DataFrame:
    df = pd.read_parquet(PHASE1 / "output" / "factor_panel.parquet")
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    return df


@st.cache_data
def load_universe_meta() -> pd.DataFrame:
    """tech_base_pool.csv：symbol/name/sector/market_cap/as_of，没跑过
    build_universe.py 时文件不存在，返回空表，页面要能处理这种情况。"""
    path = PHASE1 / "config" / "tech_base_pool.csv"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name", "sector", "market_cap", "as_of"])
    return pd.read_csv(path)


@st.cache_data
def load_phase1_ic_summary(factor_name: str) -> pd.DataFrame | None:
    path = PHASE1 / "output" / f"{factor_name}_ic_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_phase1_quantile_returns(factor_name: str) -> pd.DataFrame | None:
    path = PHASE1 / "output" / f"{factor_name}_quantile_returns.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def list_phase1_factors() -> list[str]:
    """从 output/ 目录里实际存在的 *_ic_summary.csv 反推有哪些因子跑过，
    而不是写死一份因子名单——insider_net_buy_pct 这类新因子跑完pipeline
    后不用改dashboard代码就能自动出现。"""
    out_dir = PHASE1 / "output"
    if not out_dir.exists():
        return []
    return sorted(p.stem.replace("_ic_summary", "") for p in out_dir.glob("*_ic_summary.csv"))


@st.cache_data
def load_phase2_theme_heat_index(tag: str) -> pd.DataFrame | None:
    path = PHASE2 / "output" / f"theme_heat_index_{tag}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data
def load_phase2_thematic_beta(tag: str) -> pd.DataFrame | None:
    path = PHASE2 / "output" / f"thematic_beta_{tag}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    return df


@st.cache_data
def load_phase2_ic_summary(tag: str) -> pd.DataFrame | None:
    path = PHASE2 / "output" / f"thematic_beta_{tag}_ic_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def list_phase2_tags() -> list[str]:
    out_dir = PHASE2 / "output"
    if not out_dir.exists():
        return []
    return sorted(p.stem.replace("theme_heat_index_", "") for p in out_dir.glob("theme_heat_index_*.csv"))


@st.cache_data
def load_edgar_cik_map() -> dict[str, str]:
    path = PHASE1 / "data" / "edgar_cache" / "company_tickers.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}


@st.cache_data
def load_edgar_raw_facts(cik: str) -> dict | None:
    path = PHASE1 / "data" / "edgar_cache" / f"{cik}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


COMMON_XBRL_TAGS = [
    "StockholdersEquity",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "Revenues",
    "CashAndCashEquivalentsAtCarryingValue",
]
