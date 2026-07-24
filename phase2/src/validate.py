"""
和 phase1/src/validate.py 逻辑一致（IC检验 + 分层回测，见 design_doc.md
第8节），独立一份只是为了让 phase2 的输出落到 phase2/output/，不和
phase1 的输出混在一起。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens import performance as perf

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prices_wide(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def volumes_wide(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values="volume").sort_index()


def run_ic_and_quantile_analysis(
    factor_series: pd.Series,
    price_panel: pd.DataFrame,
    factor_name: str,
    periods: tuple[int, ...] = (5, 21, 63),
    quantiles: int = 4,  # 下游候选股只有约20只，5组会太稀疏，用4组
) -> dict:
    factor_series = factor_series.dropna()
    if factor_series.empty:
        print(f"[validate] {factor_name}: 因子全是空值，跳过")
        return {}

    factor_data = get_clean_factor_and_forward_returns(
        factor=factor_series,
        prices=price_panel,
        periods=periods,
        quantiles=quantiles,
        max_loss=0.7,  # 候选股数量少+滚动窗口热身期长，进一步放宽
    )

    ic = perf.factor_information_coefficient(factor_data)
    ic_summary = ic.mean().to_frame("mean_ic").join(ic.std().to_frame("std_ic"))
    ic_summary["ic_ir"] = ic_summary["mean_ic"] / ic_summary["std_ic"]

    mean_ret_by_q, std_err_by_q = perf.mean_return_by_quantile(factor_data)

    print(f"\n===== 因子: {factor_name} =====")
    print("-- Rank IC 均值 / IC信息比率（IC_IR）--")
    print(ic_summary.round(4))
    print("-- 分层平均收益（按 quantile，1=分数最低组，N=分数最高组）--")
    print(mean_ret_by_q.round(5))

    ic_summary.to_csv(OUTPUT_DIR / f"{factor_name}_ic_summary.csv")
    mean_ret_by_q.to_csv(OUTPUT_DIR / f"{factor_name}_quantile_returns.csv")

    return {"ic_summary": ic_summary, "mean_return_by_quantile": mean_ret_by_q, "factor_data": factor_data}
