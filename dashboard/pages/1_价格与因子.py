from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_loader import load_factor_panel, load_prices

st.set_page_config(page_title="价格与因子", layout="wide")
st.title("价格与因子")

prices = load_prices()
factor_panel = load_factor_panel()
factor_cols = [c for c in factor_panel.columns if c not in ("date", "ticker")]

tab_single, tab_cross = st.tabs(["单只股票走势", "某一天的横截面排名"])

with tab_single:
    tickers = sorted(prices["ticker"].unique())
    ticker = st.selectbox("选股票", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)

    px_t = prices[prices["ticker"] == ticker].sort_values("date")
    factor_t = factor_panel[factor_panel["ticker"] == ticker].sort_values("date")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
        vertical_spacing=0.05, subplot_titles=(f"{ticker} 收盘价（复权）", "成交量"),
    )
    fig.add_trace(go.Scatter(x=px_t["date"], y=px_t["adj_close"], name="adj_close"), row=1, col=1)
    fig.add_trace(go.Bar(x=px_t["date"], y=px_t["volume"], name="volume"), row=2, col=1)
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, width='stretch')

    st.subheader("因子时间序列")
    selected_factors = st.multiselect("选因子（可多选，注意不同因子量纲不同，建议一次看1-2个）",
                                       factor_cols, default=factor_cols[:1])
    if selected_factors:
        fig2 = go.Figure()
        for f in selected_factors:
            fig2.add_trace(go.Scatter(x=factor_t["date"], y=factor_t[f], name=f))
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, width='stretch')

    with st.expander("看原始数据表（该股票最近200行）"):
        merged = px_t.merge(factor_t, on=["date", "ticker"], how="left")
        st.dataframe(merged.tail(200).sort_values("date", ascending=False), width='stretch')

with tab_cross:
    st.caption("挑一天，看当天所有股票的因子值排名——因子模型本质是横截面排序，这个视图比"
               "看单只股票的时间序列更能体现\"排序\"这件事在做什么。")
    dates = sorted(factor_panel["date"].unique())
    date_sel = st.select_slider("选日期", options=dates, value=dates[-1],
                                 format_func=lambda d: str(d)[:10])
    factor_sel = st.selectbox("按哪个因子排序", factor_cols)

    snapshot = factor_panel[factor_panel["date"] == date_sel][["ticker"] + factor_cols].dropna(subset=[factor_sel])
    snapshot = snapshot.sort_values(factor_sel, ascending=False).reset_index(drop=True)
    snapshot.index = snapshot.index + 1
    st.dataframe(snapshot, width='stretch', height=600)
