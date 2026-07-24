"""
本地数据浏览平台首页，对应你想"轻松阅读拉下来的数据"这个需求。

启动方法：
    cd stock/dashboard
    ../phase1/.venv/bin/streamlit run app.py

浏览器会自动打开 http://localhost:8501，左侧栏能切换到其他页面
（价格与因子、IC与分层回测结果、Phase2主题热度、SEC原始数据）。
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_factor_panel, load_prices, load_universe_meta

st.set_page_config(page_title="量化研究数据平台", layout="wide")

st.title("量化研究数据平台")
st.caption("只读本地已经跑出来的数据，不会联网重新拉取——想刷新数据请先在终端重跑对应的 pipeline")

prices = load_prices()
factor_panel = load_factor_panel()
universe_meta = load_universe_meta()

col1, col2, col3, col4 = st.columns(4)
col1.metric("股票数（价格数据覆盖）", f"{prices['ticker'].nunique()}")
col2.metric("价格数据行数", f"{len(prices):,}")
col3.metric("数据起止", f"{prices['date'].min().date()} ~ {prices['date'].max().date()}")
col4.metric("因子面板行数", f"{len(factor_panel):,}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("科技基础池行业分布（tech_base_pool.csv）")
    if universe_meta.empty:
        st.info("还没跑过 `python -m config.build_universe`，没有行业分布数据可看。")
    else:
        sector_counts = universe_meta["sector"].value_counts().reset_index()
        sector_counts.columns = ["sector", "count"]
        fig = px.bar(sector_counts, x="sector", y="count")
        st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("市值分布（tech_base_pool.csv，对数坐标）")
    if universe_meta.empty:
        st.info("还没跑过 `python -m config.build_universe`，没有市值数据可看。")
    else:
        fig = px.histogram(universe_meta, x="market_cap", log_x=True, nbins=40)
        st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("各因子的非空覆盖率（因子面板里有多少行不是缺失值）")
coverage_cols = [c for c in factor_panel.columns if c not in ("date", "ticker")]
coverage = pd.DataFrame({
    "factor": coverage_cols,
    "non_null_pct": [factor_panel[c].notna().mean() * 100 for c in coverage_cols],
})
st.dataframe(coverage.style.format({"non_null_pct": "{:.1f}%"}), width='stretch')

st.info("左侧栏切换页面：**价格与因子**看单只股票的走势和横截面排名，"
        "**IC与分层回测**看因子有没有预测力，**Phase2主题热度**看产业链主题贝塔，"
        "**SEC原始数据**看基本面数字最原始的披露记录长什么样。")
