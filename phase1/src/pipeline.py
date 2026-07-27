"""
Phase 1 主入口：跑通 数据 -> 因子 -> IC检验/分层回测 这条完整流水线。
对应 design_doc.md 第10节 Phase 1 范围。

每个因子除了跑全池IC检验，还会跑：
- 按市值分段的IC检验（大盘 vs 小盘，见 validate.run_ic_by_market_cap_segment）
- Fama-MacBeth横截面回归（因子 + log市值 + 因子×log市值交互项，见
  validate.run_fama_macbeth）——用一个回归模型同时估计因子的平均边际
  效应、以及这个效应是否随市值连续变化，比手动分两组更精确（比如ROE
  分段用20亿门槛测出"大盘转正"，回归里精确算出真实交叉点在约260亿美元，
  说明20亿门槛下的"大盘组"其实还混着很多效应仍为负的中型股）。

用法：
    cd phase1 && source .venv/bin/activate
    python -m src.pipeline                  # 全量4因子（含Form4资金流，较慢）
    python -m src.pipeline --no-insider      # 跳过Form4资金流因子（调试用，快很多）
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.universe import UNIVERSE  # noqa: E402
from src import data_fetch, factors, validate  # noqa: E402

START_DATE = "2016-01-01"  # 约10年历史，对应 design_doc.md 第8节"第零步"最低建议


def main():
    include_insider = "--no-insider" not in sys.argv

    print(f"[pipeline] 股票池共 {len(UNIVERSE)} 只: {UNIVERSE}")
    print(f"[pipeline] 拉取行情数据 {START_DATE} 至今 ...")
    prices = data_fetch.fetch_prices(UNIVERSE, start=START_DATE, end=str(date.today()))
    print(f"[pipeline] 行情数据行数: {len(prices)}, 覆盖股票数: {prices['ticker'].nunique()}")

    factor_desc = "动量 / ROE质量 / 账面市值比价值" + (" / Form4内部人资金流" if include_insider else "")
    print(f"[pipeline] 计算因子（{factor_desc}），并做 point-in-time 对齐 ...")
    if include_insider:
        print("[pipeline] Form4资金流因子逐笔filing拉XML，全量股票池可能要几十分钟到一个多小时，"
              "每只股票拉完会落盘缓存（data/insider_cache/），中断重跑不会白拉。")
    factor_panel = factors.build_factor_panel(prices, UNIVERSE, start=START_DATE, include_insider=include_insider)

    print("[pipeline] 应用流动性过滤（剔除挂单没有真实成交的股票-日，见 validate.liquidity_mask 注释） ...")
    mask = validate.liquidity_mask(prices).set_index(["date", "ticker"])["liquid"]
    illiquid_pct = (~mask.reindex(factor_panel.index).fillna(False)).mean() * 100
    print(f"[pipeline] 因子面板里 {illiquid_pct:.1f}% 的行因流动性不足被置空")
    factor_panel = factor_panel.where(mask.reindex(factor_panel.index).fillna(False), other=float("nan"))

    factor_panel.to_parquet(Path(__file__).resolve().parent.parent / "output" / "factor_panel.parquet")
    print(f"[pipeline] 因子面板 shape: {factor_panel.shape}")
    print(factor_panel.describe())

    price_panel = validate.prices_wide(prices)

    market_cap_series = factor_panel["market_cap"]

    results = {}
    factor_names = ["momentum", "roe_ttm", "book_to_market", "fcf_yield", "margin_trend"] + (
        ["insider_net_buy_pct"] if include_insider else []
    )
    for factor_name in factor_names:
        series = factor_panel[factor_name].dropna()
        if series.empty:
            print(f"[pipeline] 因子 {factor_name} 无有效数据，跳过（常见原因见 factors.py 注释）")
            continue
        results[factor_name] = validate.run_ic_and_quantile_analysis(series, price_panel, factor_name)

        print(f"[pipeline] {factor_name} 按市值分段检验（大盘 vs 小盘，见 design_doc.md 12节问题15发现，"
              "同一因子在不同市值段可能方向相反）...")
        results[f"{factor_name}_by_cap"] = validate.run_ic_by_market_cap_segment(
            series, market_cap_series, price_panel, factor_name
        )

        validate.run_fama_macbeth(factor_panel, price_panel, factor_name, factor_name)

    print("\n[pipeline] 完成。结果已保存到 output/ 目录（*_ic_summary.csv, *_quantile_returns.csv）")
    print(f"[pipeline] 当前股票池 {len(UNIVERSE)} 只，已达到 design_doc.md 第8节'第零步'建议的")
    print("           300-500只门槛（此前是244只），统计意义比之前更可信，但仍只覆盖")
    print("           2016年至今，还没经历完整的多轮牛熊周期检验，结论依然只能是阶段性参考。")
    return results


if __name__ == "__main__":
    main()
