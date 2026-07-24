from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from data_loader import COMMON_XBRL_TAGS, load_edgar_cik_map, load_edgar_raw_facts, load_prices

st.set_page_config(page_title="SEC原始数据", layout="wide")
st.title("SEC EDGAR 原始披露数据")
st.caption("这是 factors.py 算 ROE/账面市值比之前的最原始输入——一家公司每次财报申报的"
           "逐条披露记录，包含 filed（申报日）字段，这是避免前视偏差的关键（design_doc.md 2.5节）。")

prices = load_prices()
tickers = sorted(prices["ticker"].unique())
ticker = st.selectbox("选股票", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)

cik_map = load_edgar_cik_map()
cik = cik_map.get(ticker.upper())

if cik is None:
    st.warning("本地还没有 SEC ticker->CIK 映射缓存，先跑一遍 phase1 的 pipeline 触发缓存下载。")
    st.stop()

facts = load_edgar_raw_facts(cik)
if facts is None:
    st.warning(f"{ticker}（CIK {cik}）没有本地缓存的 XBRL 数据——常见原因是外国私人发行人"
               "（走20-F年报，不按10-Q/10-K节奏报送），见 phase1/README.md 已知限制#1。")
    st.stop()

st.write(f"**公司**: {facts.get('entityName')}　**CIK**: {cik}")

available_tags = sorted(facts.get("facts", {}).get("us-gaap", {}).keys())
default_tag = next((t for t in COMMON_XBRL_TAGS if t in available_tags), available_tags[0] if available_tags else None)
tag = st.selectbox(f"选一个 XBRL 字段（us-gaap，共 {len(available_tags)} 个可选）",
                    available_tags, index=available_tags.index(default_tag) if default_tag else 0)

try:
    entries = facts["facts"]["us-gaap"][tag]["units"]
    unit = next(iter(entries.keys()))
    rows = entries[unit]
except (KeyError, StopIteration):
    rows = []

if not rows:
    st.info("这个字段没有数据。")
else:
    df = pd.DataFrame(rows)
    show_cols = [c for c in ["start", "end", "val", "form", "fy", "fp", "filed", "frame"] if c in df.columns]
    df = df[show_cols].sort_values("filed" if "filed" in df.columns else show_cols[0])
    st.caption(f"单位: {unit}　共 {len(df)} 条披露记录（同一个 `end` 期间可能被多次 filed，"
               "比如年报里会重新披露上年同期数字用于对比，这是正常现象）")
    st.dataframe(df, width='stretch', height=500)
