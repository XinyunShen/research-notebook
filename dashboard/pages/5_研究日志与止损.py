"""
每日云端routine（stock-strategy-daily-research）产出的研究日志 + 持仓止损监控，
读的是 research_log/（跟 phase1/phase2 的 output/ 不一样，这个目录是被追踪进git的，
routine每天在 github.com/XinyunShen/research-notebook 上提交，本地要先"拉取最新"
才能看到当天的）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from data_loader import (
    ROOT,
    list_research_log_dates,
    load_price_watch_state,
    load_research_log_entry,
    load_research_log_index,
)

st.set_page_config(page_title="研究日志与止损", layout="wide")
st.title("研究日志与止损监控")
st.caption("内容来自每天自动跑的云端routine（stock-strategy-daily-research），"
           "只做研究记录和止损提醒，不会自动下单——真要操作请回到对话里跟Claude确认。")

if st.button("拉取最新（git pull）"):
    result = subprocess.run(["git", "pull"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        st.success(f"拉取成功：{result.stdout.strip() or '已是最新'}")
        st.cache_data.clear()
    else:
        st.error(f"拉取失败：{result.stderr.strip()}")

st.divider()

st.subheader("持仓止损状态")
state = load_price_watch_state()
if state is None:
    st.info("还没有 `research_log/price_watch_state.json`——routine第一次运行后才会出现"
            "（下次运行时间见 claude.ai/code/routines），先点上面「拉取最新」试试。")
else:
    if isinstance(state, dict) and all(isinstance(v, dict) for v in state.values()):
        rows = []
        for ticker, info in state.items():
            row = {"ticker": ticker}
            row.update(info)
            rows.append(row)
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), width='stretch')
    else:
        st.json(state)
    st.caption("这是routine上次运行时算出来的参考值，不是Robinhood的实时权威数据——"
               "真要决定卖不卖，回到对话里让Claude现查一遍Robinhood当前报价。")

st.divider()

st.subheader("研究日志索引（INDEX.md，最新在最上面）")
index_content = load_research_log_index()
if index_content is None:
    st.info("还没有 `research_log/INDEX.md`，routine第一次运行后才会出现。")
else:
    st.markdown(index_content)

st.divider()

st.subheader("查看某一天的完整记录")
dates = list_research_log_dates()
if not dates:
    st.info("还没有任何一天的研究日志。")
else:
    picked = st.selectbox("选日期", dates)
    entry = load_research_log_entry(picked)
    if entry:
        st.markdown(entry)
