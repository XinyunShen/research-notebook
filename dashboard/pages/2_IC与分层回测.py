from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    list_phase1_factors,
    list_phase2_tags,
    load_phase1_ic_summary,
    load_phase1_quantile_returns,
    load_phase2_ic_summary,
)

st.set_page_config(page_title="IC与分层回测", layout="wide")
st.title("IC 与分层回测结果")
st.caption("对应 design_doc.md 第8节第一步/第二步：IC看因子本身有没有预测力，分层回测看"
           "因子分数和未来收益是不是单调相关。数字来自最近一次跑 pipeline 落盘的 csv，不是实时算的。")

tab_p1, tab_p2 = st.tabs(["Phase 1 因子", "Phase 2 主题贝塔（raw/excess/trends对比）"])

with tab_p1:
    factors_available = list_phase1_factors()
    if not factors_available:
        st.warning("output/ 目录里还没有 *_ic_summary.csv，先跑一遍 `python -m src.pipeline`。")
    else:
        factor_name = st.selectbox("选因子", factors_available)
        ic_summary = load_phase1_ic_summary(factor_name)
        quantile_returns = load_phase1_quantile_returns(factor_name)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("IC 均值 / IC_IR")
            st.dataframe(ic_summary, width='stretch')
            if ic_summary is not None:
                fig = px.bar(ic_summary.reset_index(), x="index", y=["mean_ic", "ic_ir"], barmode="group")
                fig.update_layout(xaxis_title="持有期", legend_title="")
                st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("分层平均收益（1=分数最低组）")
            st.dataframe(quantile_returns, width='stretch')
            if quantile_returns is not None:
                qr = quantile_returns.reset_index().melt(id_vars=quantile_returns.index.name or "index",
                                                           var_name="period", value_name="avg_return")
                qr = qr.rename(columns={qr.columns[0]: "quantile"})
                fig2 = px.bar(qr, x="quantile", y="avg_return", color="period", barmode="group")
                st.plotly_chart(fig2, width='stretch')
                st.caption("越干净的因子，这张图应该长得像一道从左到右往上走的楼梯（单调递增）。")

with tab_p2:
    tags = list_phase2_tags()
    if not tags:
        st.warning("phase2/output/ 目录里还没有结果，先跑一遍 `cd phase2 && python -m src.pipeline`。")
    else:
        st.caption("三条对比线：raw=都没修方法论缺陷，excess=剔除了行业共振，"
                   "trends=进一步用Google Trends修了circularity。63日IC从raw/excess的~0.09"
                   "掉到trends的~0.02，是这个项目目前最重要的一次发现（design_doc.md 12节问题12）。")
        rows = []
        for tag in tags:
            s = load_phase2_ic_summary(tag)
            if s is None:
                continue
            s = s.reset_index().rename(columns={s.index.name or "index": "period"})
            s["version"] = tag
            rows.append(s)
        if rows:
            combined = pd.concat(rows, ignore_index=True)
            fig3 = px.bar(combined, x="period", y="mean_ic", color="version", barmode="group",
                          title="63日/21日/5日 Rank IC 对比")
            st.plotly_chart(fig3, width='stretch')
            fig4 = px.bar(combined, x="period", y="ic_ir", color="version", barmode="group",
                          title="IC_IR（IC均值/IC标准差，越高越稳定）对比")
            st.plotly_chart(fig4, width='stretch')
            st.dataframe(combined, width='stretch')
