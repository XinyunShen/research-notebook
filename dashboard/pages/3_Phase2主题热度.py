from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from data_loader import list_phase2_tags, load_phase2_theme_heat_index, load_phase2_thematic_beta

st.set_page_config(page_title="Phase2 主题热度", layout="wide")
st.title("Phase 2：主题热度指数 与 主题贝塔")

tags = list_phase2_tags()
if not tags:
    st.warning("phase2/output/ 目录里还没有结果，先跑一遍 `cd phase2 && python -m src.pipeline`。")
    st.stop()

st.subheader("主题热度指数（theme_heat_index）三个版本对比")
fig = go.Figure()
for tag in tags:
    df = load_phase2_theme_heat_index(tag)
    if df is not None and "theme_heat_index" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["theme_heat_index"], name=tag, mode="lines"))
fig.update_layout(height=400, legend_title="版本")
st.plotly_chart(fig, width='stretch')
st.caption("raw/excess 用SOXX/SMH成交量构造，trends用Google Trends替换——三条线走势像不像，"
           "本身不能说明信号好坏，要看下面的贝塔和上一页的IC。")

st.divider()
st.subheader("下游候选股的主题贝塔（thematic_beta）")
tag_sel = st.selectbox("选版本", tags, index=tags.index("trends") if "trends" in tags else 0)
beta_df = load_phase2_thematic_beta(tag_sel)

if beta_df is None or beta_df.empty:
    st.info(f"没有 {tag_sel} 版本的贝塔数据。")
else:
    tickers = sorted(beta_df["ticker"].unique())
    ticker_sel = st.selectbox("选下游候选股", tickers)
    beta_t = beta_df[beta_df["ticker"] == ticker_sel].sort_values("date")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=beta_t["date"], y=beta_t["thematic_beta"], mode="lines"))
    fig2.update_layout(height=350, title=f"{ticker_sel} 对主题热度指数的滚动126日贝塔（{tag_sel}版本）")
    st.plotly_chart(fig2, width='stretch')

    with st.expander("看所有下游股票最新一天的贝塔排名"):
        latest_date = beta_df["date"].max()
        latest = beta_df[beta_df["date"] == latest_date].sort_values("thematic_beta", ascending=False)
        st.dataframe(latest.reset_index(drop=True), width='stretch')
