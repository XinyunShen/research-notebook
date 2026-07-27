"""
验证层：对应 design_doc.md 第8节"第一步：单因子IC检验"和"第二步：分层回测"。

用 Alphalens 做：
1. IC 分析（Rank IC 均值、IC信息比率）——判断因子本身有没有预测力。
2. 分层回测（Quantile Backtest）——判断因子分数是否和未来收益单调相关。

明确不做的事（Phase 1 范围声明，避免过度承诺）：
- 不做 Deflated Sharpe Ratio（第8节第五步）、不算交易成本（第六步）。
  这些留到 Phase 3（design_doc.md 第10节路线图）。
- walk-forward（第三步）、样本外冻结测试（第四步）、非重叠采样/Newey-West
  调整IC_IR、regime分段IC_IR、起始日敏感性——原本也计划留到Phase 3，但
  2026-07-24已经开始拿真钱在Robinhood小规模实盘，提前补做了，见
  `robustness.py` 和 `README.md`"稳健性检验"一节，不在这个文件里。
- 当前股票池只有约50只，第8节"第零步"已经说明这远低于统计意义上
  可信的规模（建议300-500只起步），这里的 IC 数字只能用来验证代码
  流水线本身跑得对不对，不能当成"这个因子真的有效"的证据。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens import performance as perf
from linearmodels.panel import FamaMacBeth

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prices_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """Alphalens 需要 index=date, columns=ticker 的收盘价宽表。"""
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


MIN_DOLLAR_VOLUME = 1_000_000  # 美元/天，可交易流动性的保守下限


def liquidity_mask(prices: pd.DataFrame, window: int = 21, min_dollar_volume: float = MIN_DOLLAR_VOLUME) -> pd.DataFrame:
    """返回长表：columns=[date, ticker, liquid]。

    **排查Phase 1精修扩池后发现的真实问题**（design_doc.md 12节问题12）：
    用"当前市值"筛出的科技基础池，回溯拉全部历史价格时，有些股票早年
    是完全不同性质的公司（壳公司/仙股，后来改名/重组成了今天这家公司），
    历史那段价格数据是真实的行情，但不代表"可交易"——实测案例：某只
    现在市值18亿美元、够格进入股票池的股票，2018年那段历史成交额中位数
    是 0（多数交易日根本没有真实成交，价格只是挂单报价没有成交量），
    126日动量算出18899%这种荒谬值，还拖累了同期分层回测的单调性。

    **不能用股价高低做门槛**：同一批数据里 NVDA 2016年（複权调整后）
    股价也低至0.82美元，但当天成交额中位数2.86亿美元，是真实流动的
    交易——股价便宜可能只是因为后续多次拆股，不代表不能交易。能区分
    "真实但便宜"和"挂单没成交"这两种情况的是**成交额**，不是股价本身。

    门槛选1百万美元/天（过去21个交易日成交额中位数），是一个保守、
    明显低于"真实机构可交易"标准的下限，只用来剔除挂单空转的极端情况，
    不是想做一个严格的流动性因子。
    """
    df = prices[["date", "ticker", "adj_close", "volume"]].copy()
    df["dollar_vol"] = df["adj_close"] * df["volume"]
    df = df.sort_values(["ticker", "date"])
    df["trailing_dollar_vol"] = df.groupby("ticker")["dollar_vol"].transform(
        lambda s: s.rolling(window, min_periods=max(1, window // 2)).median()
    )
    df["liquid"] = df["trailing_dollar_vol"] >= min_dollar_volume
    return df[["date", "ticker", "liquid"]]


def run_ic_and_quantile_analysis(
    factor_series: pd.Series,
    price_panel: pd.DataFrame,
    factor_name: str,
    periods: tuple[int, ...] = (5, 21, 63),
    quantiles: int = 5,
) -> dict:
    """factor_series: MultiIndex (date, ticker) -> 因子值
    price_panel: index=date, columns=ticker 的收盘价宽表
    periods: 未来收益的观察窗口（交易日），比如5天/1个月/1个季度
    """
    factor_series = factor_series.dropna()
    if factor_series.empty:
        print(f"[validate] {factor_name}: 因子全是空值，跳过")
        return {}

    factor_data = get_clean_factor_and_forward_returns(
        factor=factor_series,
        prices=price_panel,
        periods=periods,
        quantiles=quantiles,
        max_loss=0.6,  # 小样本+数据缺口较多，放宽丢弃比例上限，Phase1可接受
    )

    ic = perf.factor_information_coefficient(factor_data)
    ic_summary = ic.mean().to_frame("mean_ic").join(ic.std().to_frame("std_ic"))
    ic_summary["ic_ir"] = ic_summary["mean_ic"] / ic_summary["std_ic"]

    mean_ret_by_q, std_err_by_q = perf.mean_return_by_quantile(factor_data)

    print(f"\n===== 因子: {factor_name} =====")
    print("-- Rank IC 均值 / IC信息比率（IC_IR）--")
    print(ic_summary.round(4))
    print("-- 分层平均收益（按 quantile，1=分数最低组，5=分数最高组）--")
    print(mean_ret_by_q.round(5))

    ic_summary.to_csv(OUTPUT_DIR / f"{factor_name}_ic_summary.csv")
    mean_ret_by_q.to_csv(OUTPUT_DIR / f"{factor_name}_quantile_returns.csv")

    monotonic = {}
    for period_col in mean_ret_by_q.columns:
        vals = mean_ret_by_q[period_col].values
        monotonic[period_col] = bool(pd.Series(vals).is_monotonic_increasing)
    print(f"-- 分层收益是否单调递增（越高分组收益越高）: {monotonic} --")

    return {
        "ic_summary": ic_summary,
        "mean_return_by_quantile": mean_ret_by_q,
        "monotonic": monotonic,
        "factor_data": factor_data,
    }


LARGE_CAP_THRESHOLD = 2_000_000_000  # 呼应科技基础池最初的20亿门槛，见 factors.py market_cap 注释


def run_ic_by_market_cap_segment(
    factor_series: pd.Series,
    market_cap_series: pd.Series,
    price_panel: pd.DataFrame,
    factor_name: str,
    threshold: float = LARGE_CAP_THRESHOLD,
) -> dict:
    """把因子按**当天真实市值**（不是"今天"的市值套用到整段历史）分成
    大盘（>=threshold）/小盘（<threshold）两组，分别跑IC检验。

    **为什么要有这个**（design_doc.md 12节问题15）：排查book_to_market
    扩池到305只后IC从0.008掉到接近0的原因时发现，新增的61只小盘股
    （9亿-20亿市值）63日IC是**-0.0186**（方向反了），老范围的约244只
    63日IC是**0.0062**（接近旧结果、方向正确）——同一个因子在大盘/小盘
    股里可能方向完全相反，混在一起算会互相抵消，把两段都算出来才看得
    清楚。这次发现之后，市值分段就从"一次性诊断"变成标准流程的一部分，
    对所有因子都跑，不只是book_to_market。

    market_cap_series: MultiIndex (date, ticker) -> 市值，来自
    factors.py 里用 shares_outstanding × 当天股价算出的point-in-time
    市值（不是快照）。外国私人发行人没有XBRL数据、market_cap全程缺失
    的股票会被两个分段都排除（不是bug，是数据覆盖限制，和ROE/B-M因子
    本身缺数据的股票范围一致）。
    """
    aligned_cap = market_cap_series.reindex(factor_series.index)
    results = {}
    for label, mask in [("large_cap", aligned_cap >= threshold), ("small_cap", aligned_cap < threshold)]:
        sub = factor_series[mask.fillna(False)]
        n_tickers = sub.index.get_level_values("ticker").nunique() if not sub.empty else 0
        print(f"\n[validate] --- {factor_name} 分市值段: {label}"
              f"（阈值{threshold:,.0f}美元，{n_tickers}只股票，{len(sub)}条观测）---")
        results[label] = run_ic_and_quantile_analysis(sub, price_panel, f"{factor_name}_{label}")
    return results


def compute_forward_returns(price_panel: pd.DataFrame, period: int) -> pd.DataFrame:
    """price_panel: index=date, columns=ticker 的收盘价宽表。返回同形状的
    DataFrame，每个格子是"从这天开始持有period个交易日"的未来收益率——
    这是 Fama-MacBeth 回归要预测的目标变量（y），不是特征，用未来价格
    算目标变量不构成前视偏差（这正是回测要预测的东西）。"""
    return price_panel.shift(-period) / price_panel - 1


def run_fama_macbeth(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    factor_col: str,
    factor_label: str,
    period: int = 63,
    interact_with_log_size: bool = True,
) -> None:
    """Fama-MacBeth 横截面回归（design_doc.md 12节：市值分段发现ROE在
    大盘/小盘方向相反之后的自然下一步）。跟 `run_ic_by_market_cap_segment`
    手动拆两组分别算IC不同，这里用**一个回归模型**同时估计：
    - 因子本身的平均边际效应（控制了市值之后）
    - 因子效应是否**随市值连续变化**（因子×log市值 交互项，如果显著，
      说明"这个因子在大盘/小盘方向不同"不是拆两组时才出现的巧合，而是
      一个连续、系统性的规律）

    做法：每一天做一次横截面OLS回归（未来收益 ~ 因子 + log市值 +
    因子×log市值），日复一日得到一串系数，取这串系数的均值/标准误
    （标准误来自这串系数本身的时间序列波动，不是单次回归的标准误——
    这是Fama-MacBeth和普通pooled OLS的本质区别，天然处理了"同一天多只
    股票的残差互相关"这个问题，比对完整panel直接跑一次OLS更可信）。
    用 `linearmodels.panel.FamaMacBeth`，不自己手写循环，避免重新发明
    一个可能有微妙bug的统计实现。

    **point-in-time说明**：市值和因子值都是当天的值，未来收益是当天往后
    period个交易日，和IC检验的时间对齐方式完全一致，不引入新的前视偏差。
    """
    fwd_ret = compute_forward_returns(price_panel, period)
    fwd_long = fwd_ret.stack().rename("fwd_ret")
    fwd_long.index.names = ["date", "ticker"]

    df = factor_panel[[factor_col, "market_cap"]].copy()
    df = df.join(fwd_long, how="inner")
    df = df.dropna(subset=[factor_col, "market_cap", "fwd_ret"])
    df = df[df["market_cap"] > 0]
    df["log_market_cap"] = np.log(df["market_cap"])

    formula = f"fwd_ret ~ 1 + {factor_col} + log_market_cap"
    if interact_with_log_size:
        df["interaction"] = df[factor_col] * df["log_market_cap"]
        formula += " + interaction"

    # linearmodels 要求 MultiIndex 是 (entity, time)，我们面板是 (date, ticker)，
    # 顺序要倒过来。
    panel_df = df.reset_index().set_index(["ticker", "date"])[
        [c for c in ["fwd_ret", factor_col, "log_market_cap", "interaction"] if c in df.columns]
    ]

    n_dates = panel_df.index.get_level_values("date").nunique()
    n_tickers = panel_df.index.get_level_values("ticker").nunique()
    print(f"\n[validate] ===== Fama-MacBeth 回归: {factor_label} (period={period}日) =====")
    print(f"[validate] 样本: {n_tickers}只股票 × {n_dates}个交易日，共{len(panel_df)}条观测")

    model = FamaMacBeth.from_formula(formula, panel_df)
    res = model.fit()
    print(res.summary.tables[1])

    if interact_with_log_size:
        interaction_coef = res.params.get("interaction")
        interaction_p = res.pvalues.get("interaction")
        if interaction_coef is not None:
            sig = "显著" if interaction_p < 0.05 else "不显著"
            direction = "随市值增大而增强" if interaction_coef > 0 else "随市值增大而减弱/反转"
            print(f"[validate] 交互项 {factor_col}×log市值 系数={interaction_coef:.4f}, "
                  f"p值={interaction_p:.4f}（{sig}）——因子效应{direction}")
