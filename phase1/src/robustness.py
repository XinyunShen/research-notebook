"""
稳健性检验层：补齐 design_doc.md 第8节第三步（走步前进验证）、第四步
（样本外冻结测试），并回应2026-07-24讨论里提出的两个方法论问题：

1. **重叠窗口造成的自相关伪影**：momentum用126日滚动窗口、IC按每个
   交易日算一次，今天和昨天的动量分数125/126的输入完全重复，未来收益
   的滚动窗口同样重叠。`ic_ir = mean_ic/std_ic` 隐含"逐日IC观测值互相
   独立"这个假设，重叠窗口下明显不成立——std_ic算出来的到底是"信号真
   实不稳定"还是"重叠采样的人为噪音"，validate.py原来的实现没有区分。
   这里给两种独立于这个假设的估计方式：`ic_ir_nonoverlapping`（稀疏
   采样，简单但扔数据）、`ic_ir_newey_west`（HAC调整标准误，保留全部
   观测但需要多一层统计假设）。

2. **单一路径/单一起始日的偏差**：如果换个起始日期、调仓时点跟着平移，
   算出来的回测统计量会不会明显不同？如果会，说明现在报告的数字本身
   就是有偏的估计，不能只看一条路径。见 `start_date_sensitivity`。

3. **Regime混杂**：2016-2026横跨AI牛市、2022加息熊市、2020疫情熔断等
   好几种完全不同的市场环境，混在一起测一个IC_IR，逐日IC肯定跟着regime
   剧烈摆动——这是不同regime本身造成的低IC_IR，不代表信号是假的，但也
   不该被平均数字掩盖。复用 backtest.build_regime_filter（QQQ 200日均
   线，业界标准参数，不是调出来的）做risk-on/risk-off分段，phase2的
   "叙事前/后"分段检验已经验证过这个做法能揪出被混杂掩盖的差异。
   见 `ic_ir_by_regime`。

**这里明确不做的事**：Deflated Sharpe Ratio（design_doc.md第8节第五步）、
真实交易成本（第六步）——这两步依赖前面几步先把"IC_IR低到底是信号真弱
还是计算伪影"这个问题弄清楚，属于下一阶段的工作，不在本文件范围内。

**走步前进验证在这里的诚实说明**：Phase 1的因子目前没有任何"调参"步骤
（动量用126日窗口是design_doc.md 3.1节按学术惯例定的固定值，没有网格
搜索选出来），design_doc.md第三步说的"用前3年训练/调参"这一层在当前
代码里没有对应物——所以 `walk_forward_ic` 实质上测的是"把IC_IR拆成多
个滚动的、互不重叠的样本外时间窗口分别算，结果是否稳定"，跟"参数在训
练窗口拟合、之后从未见过的窗口测试"这个更严格的定义不完全一样，只是
在没有可调参数的前提下能做到的最接近的版本，这个区别在结果里要如实说明。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens import performance as perf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from . import backtest

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _daily_ic(factor_series: pd.Series, price_panel: pd.DataFrame, period: int) -> pd.Series:
    """内部工具：算出逐日IC序列（复用alphalens，跟validate.py里的实现
    保持一致，避免两套代码算出不同数字）。"""
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor_series.dropna(),
        prices=price_panel,
        periods=(period,),
        quantiles=5,
        max_loss=0.6,
    )
    return perf.factor_information_coefficient(factor_data)[f"{period}D"].dropna()


def ic_ir_nonoverlapping(
    factor_series: pd.Series,
    price_panel: pd.DataFrame,
    period: int,
    factor_name: str,
    sample_offset: int = 0,
) -> dict:
    """把逐日IC按period个交易日做非重叠采样再算IC_IR——采样间隔正好等于
    forward return的窗口长度，采出来的观测之间不再共享任何未来收益的
    重叠区间，`std_ic`才是对"信号本身有多不稳定"干净的估计，不掺重叠
    采样的伪影。代价：只用了1/period的数据，n_obs降到十分之一量级，
    标准误变大是这个方法本身的代价，不是新发现的问题。

    sample_offset：从第几个交易日开始采样（0..period-1），配合多个
    offset跑一遍能看出"非重叠采样对齐点本身"引入了多少随机性
    ——这正是 start_date_sensitivity 想测的同一类问题在IC层面的版本。
    """
    ic_daily = _daily_ic(factor_series, price_panel, period)
    ic_sparse = ic_daily.iloc[sample_offset::period]
    n = len(ic_sparse)
    mean_ic = ic_sparse.mean()
    std_ic = ic_sparse.std()
    ic_ir = mean_ic / std_ic if std_ic else float("nan")
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if n > 1 and std_ic else float("nan")
    result = {
        "factor": factor_name, "period": period, "sample_offset": sample_offset,
        "n_obs": n, "mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "t_stat": t_stat,
    }
    print(f"[robustness] {factor_name} {period}D 非重叠IC_IR（offset={sample_offset}）: "
          f"n={n}, mean_ic={mean_ic:.4f}, ic_ir={ic_ir:.4f}, t={t_stat:.2f}")
    return result


def ic_ir_newey_west(
    factor_series: pd.Series,
    price_panel: pd.DataFrame,
    period: int,
    factor_name: str,
    max_lag: int | None = None,
) -> dict:
    """备选做法：不做稀疏采样（会扔掉大部分数据），改用Newey-West HAC
    调整逐日IC均值的标准误，同样解决"重叠窗口下逐日IC不独立"这个问题，
    但保留全部观测（信息效率更高）。max_lag默认取period-1，正好对应
    重叠窗口理论上的自相关截断长度（今天和period-1天前的动量分数还
    共享输入价格，再往前就不共享了），不是随意调出来的参数。

    只对均值做HAC回归（y ~ 常数项），t_stat_nw是"这个因子的平均IC是否
    显著不为0"的干净检验；ic_ir仍然按naive std_ic算（跟validate.py里
    的定义保持一致，方便直接对比两种方法差多少），但报告里应该以
    t_stat_nw为准，不是ic_ir本身。
    """
    ic_daily = _daily_ic(factor_series, price_panel, period)
    max_lag = max_lag if max_lag is not None else period - 1
    y = ic_daily.values
    x = add_constant(np.ones(len(y)))
    model = OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})
    mean_ic = float(model.params[0])
    se_nw = float(model.bse[0])
    t_stat_nw = float(model.tvalues[0])
    naive_std = ic_daily.std()
    naive_ir = mean_ic / naive_std if naive_std else float("nan")
    result = {
        "factor": factor_name, "period": period, "max_lag": max_lag, "n_obs": len(y),
        "mean_ic": mean_ic, "naive_std_ic": naive_std, "naive_ic_ir": naive_ir,
        "newey_west_se": se_nw, "t_stat_nw": t_stat_nw,
    }
    print(f"[robustness] {factor_name} {period}D Newey-West（max_lag={max_lag}）: "
          f"mean_ic={mean_ic:.4f}, naive_ic_ir={naive_ir:.4f}, t_stat_nw={t_stat_nw:.2f}")
    return result


def ic_ir_by_regime(
    factor_series: pd.Series,
    price_panel: pd.DataFrame,
    index_prices: pd.Series,
    period: int,
    factor_name: str,
    window: int = 200,
) -> dict:
    """按QQQ 200日均线regime（risk-on/risk-off）拆分逐日IC分别算IC_IR
    ——不是按固定日期分段（那是phase2"叙事前/后"用的方法，AI叙事有明确
    起点2022-11可以按日期切；这里没有类似的"信号起点"，用regime本身
    做切分更合理），能看出低IC_IR是不是被某个特定市场环境（比如2022
    加息熊市）拖累，还是所有regime下都同样弱。
    """
    ic_daily = _daily_ic(factor_series, price_panel, period)
    regime = backtest.build_regime_filter(index_prices, window=window)
    regime_aligned = regime.reindex(ic_daily.index).fillna(False)

    results = {}
    for label, mask in [("risk_on", regime_aligned), ("risk_off", ~regime_aligned)]:
        sub = ic_daily[mask]
        n = len(sub)
        mean_ic = sub.mean() if n else float("nan")
        std_ic = sub.std() if n else float("nan")
        ic_ir = mean_ic / std_ic if std_ic else float("nan")
        results[label] = {"n_obs": n, "mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir}
        print(f"[robustness] {factor_name} {period}D regime={label}: "
              f"n={n}, mean_ic={mean_ic:.4f}, ic_ir={ic_ir:.4f}")

    df = pd.DataFrame(results).T
    df.to_csv(OUTPUT_DIR / f"{factor_name}_{period}d_ic_by_regime.csv")
    return results


def walk_forward_ic(
    factor_series: pd.Series,
    price_panel: pd.DataFrame,
    period: int,
    factor_name: str,
    test_months: int = 6,
    step_months: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """把逐日IC切成连续、不重叠的test_months个月一段，每段单独算IC_IR，
    再把所有段的IC拼起来算一个"walk-forward整体IC_IR"。design_doc.md
    第三步说的"训练窗口"这里不存在（本文件顶部docstring已经说明原因），
    所以这个函数测的是"IC_IR在不同、从未混在一起算的时间段里是否稳定"，
    不是"用训练窗口选出的参数在测试窗口还有效吗"——两者不是同一件事，
    这里给的是前者。

    step_months默认等于test_months（完全不重叠、连续滚动），design_doc.md
    建议至少6-8个独立窗口，26年历史、6个月一段能切出40+段，远超这个门槛。
    """
    step_months = step_months or test_months
    ic_daily = _daily_ic(factor_series, price_panel, period)
    dates = ic_daily.index

    start, end = dates.min(), dates.max()
    windows = []
    cur = start
    while cur < end:
        window_end = cur + pd.DateOffset(months=test_months)
        windows.append((cur, min(window_end, end)))
        cur = cur + pd.DateOffset(months=step_months)

    rows = []
    for i, (ws, we) in enumerate(windows):
        sub = ic_daily[(dates >= ws) & (dates < we)]
        n = len(sub)
        if n < 5:  # 太短的尾段（比如历史刚好切不整）没有统计意义，跳过
            continue
        mean_ic, std_ic = sub.mean(), sub.std()
        ic_ir = mean_ic / std_ic if std_ic else float("nan")
        rows.append({
            "window": i, "test_start": ws, "test_end": we, "n_obs": n,
            "mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir,
        })

    windows_df = pd.DataFrame(rows)
    windows_df.to_csv(OUTPUT_DIR / f"{factor_name}_{period}d_walk_forward_windows.csv", index=False)

    # 整体：把所有窗口的逐日IC重新拼起来算一次（等价于全样本IC_IR，
    # 但保留了"这是K个独立窗口拼出来的"这个可追溯性），另外单独报告
    # "窗口间IC_IR的IC_IR"——即窗口均值的标准差相对窗口均值的均值，
    # 衡量的是"结论本身随时间点漂不漂"，跟单窗口内部的噪音是两件事。
    overall_mean = windows_df["mean_ic"].mul(windows_df["n_obs"]).sum() / windows_df["n_obs"].sum()
    pooled_std = ic_daily.std()
    overall_ic_ir = overall_mean / pooled_std if pooled_std else float("nan")
    window_mean_std = windows_df["mean_ic"].std()
    pct_positive_windows = (windows_df["mean_ic"] > 0).mean()

    summary = {
        "factor": factor_name, "period": period, "n_windows": len(windows_df),
        "overall_mean_ic": overall_mean, "overall_ic_ir": overall_ic_ir,
        "window_mean_ic_std": window_mean_std, "pct_windows_positive_mean_ic": pct_positive_windows,
    }
    print(f"\n[robustness] {factor_name} {period}D 走步前进（{len(windows_df)}个{test_months}月窗口）：")
    print(windows_df.round(4).to_string(index=False))
    print(f"[robustness] 汇总: overall_mean_ic={overall_mean:.4f}, overall_ic_ir={overall_ic_ir:.4f}, "
          f"窗口间mean_ic的std={window_mean_std:.4f}, "
          f"{pct_positive_windows*100:.0f}%的窗口mean_ic为正")
    return windows_df, summary


def start_date_sensitivity(
    prices_wide: pd.DataFrame,
    factor_panel: pd.DataFrame,
    factor_col: str,
    base_start: str,
    n_offsets: int = 21,
    **backtest_kwargs,
) -> pd.DataFrame:
    """把 backtest.simulate_portfolio 的start_date沿交易日依次平移
    0..n_offsets-1天重跑n_offsets次（rebalance_freq等其他参数不变），
    只是"从哪天开始数第一次再平仓"这一个变量在变——如果CAGR/Sharpe/
    最大回撤随这个跟策略信号毫无关系的选择大幅摆动，说明原来报告的
    单一路径数字本身就是有偏估计，不能直接采信。

    n_offsets=21：默认rebalance_freq是21个交易日，平移满一整个再平衡
    周期就足够覆盖"月初调仓"和"月中调仓"之间的所有可能对齐方式，
    再往后平移会开始重复前面已经测过的相位。
    """
    trading_days = prices_wide.index[prices_wide.index >= pd.Timestamp(base_start)]
    rows = []
    for offset in range(min(n_offsets, len(trading_days))):
        start_date = trading_days[offset]
        result = backtest.simulate_portfolio(
            prices_wide, factor_panel, factor_col,
            start_date=str(start_date.date()), **backtest_kwargs,
        )
        s = result.summary()
        rows.append({"offset_days": offset, "start_date": start_date, **s})

    df = pd.DataFrame(rows)
    numeric_cols = ["total_return", "cagr", "max_drawdown", "sharpe", "win_rate"]
    stats = df[numeric_cols].agg(["mean", "std", "min", "max"])
    df.to_csv(OUTPUT_DIR / f"{factor_col}_start_date_sensitivity.csv", index=False)

    print(f"\n[robustness] {factor_col} 起始日敏感性（{len(df)}个偏移，base_start={base_start}）：")
    print(df.round(4).to_string(index=False))
    print("-- 跨偏移的统计量分布（如果std相对mean很大，说明单一路径的数字不可信）--")
    print(stats.round(4))
    return df


def out_of_sample_freeze(
    factor_series: pd.Series,
    price_panel: pd.DataFrame,
    period: int,
    factor_name: str,
    freeze_start: str,
) -> dict:
    """把freeze_start之后的数据当成从未用于任何调参/挑选决策的样本外
    冻结区间，只跑一次、单独报告IC。

    **诚实的限制说明，不能跳过**：动量因子是纯价格构造，freeze_start
    之前已经在多次交互式分析里被反复算过IC、看过分层回测——如果
    freeze_start设在那些分析已经覆盖到的日期之前，这就不是design_doc.md
    第四步说的"从没在优化过程中用过的时间段"，只是"最近一段"，性质上
    退化成普通的时间分段检验，不能当成"样本外冻结测试"的证据来用。
    这个函数只提供计算机制，freeze_start选多晚才算"真冻结"要看清楚
    实际的调参/分析历史，不是这个函数能自动判断的。
    """
    ic_daily = _daily_ic(factor_series, price_panel, period)
    freeze_ts = pd.Timestamp(freeze_start)
    is_ic = ic_daily[ic_daily.index < freeze_ts]
    oos_ic = ic_daily[ic_daily.index >= freeze_ts]

    def _summ(s):
        mean_ic, std_ic = s.mean(), s.std()
        return {"n_obs": len(s), "mean_ic": mean_ic, "std_ic": std_ic,
                "ic_ir": mean_ic / std_ic if std_ic else float("nan")}

    result = {"factor": factor_name, "period": period, "freeze_start": freeze_start,
              "in_sample": _summ(is_ic), "frozen_oos": _summ(oos_ic)}
    print(f"\n[robustness] {factor_name} {period}D 样本外冻结检验（分界{freeze_start}）：")
    print(f"  样本内（<{freeze_start}）: {result['in_sample']}")
    print(f"  冻结样本外（>={freeze_start}）: {result['frozen_oos']}")
    print("  ⚠️  是否真正'冻结'取决于freeze_start之前的分析历史有没有覆盖到"
          "这段数据，见函数docstring")
    return result
