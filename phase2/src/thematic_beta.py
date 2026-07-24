"""
主题贝塔（Thematic Beta）计算，对应 design_doc.md 3.2节 + 8节"验证机制B"步骤2。

做法：对每只候选股票，用滚动窗口把它的日收益率对"主题热度指数"的日度
变化做回归，取斜率(beta)。beta越高，说明这只股票的涨跌和主题热度的
共振越强——这是我们把"证明因果"换成"证明共振强度"的具体实现
（design_doc.md 3.2节已经说明这个思路转换）。

用滚动协方差/方差算 beta（等价于滚动OLS单变量回归的斜率，但快得多）：
    beta_t = Cov(stock_return, index_change)_{t-window+1:t} / Var(index_change)_{t-window+1:t}

**point-in-time说明**：滚动窗口在t天结束时只用到了 t 及之前的数据，
天然满足"不能用未来数据"的要求，不需要额外做 as-of 对齐（这点和
Phase 1 的基本面因子不同，基本面因子是"离散事件+申报延迟"，这里是
"连续滚动窗口"，两种不同的point-in-time处理方式，design_doc.md
8节验证机制B部分提到过这个区别）。

**已知简化**：这里用"主题热度指数的日度变化"近似"主题指数的收益率"，
不是一个真正意义上可交易的指数收益率，是v1的近似做法，第10节路线图
如果继续深化可以再精修（比如换成真正可交易的复合ETF组合收益率）。
"""
from __future__ import annotations

import pandas as pd

BETA_WINDOW = 126  # 约6个月，和其他因子的63/126日尺度保持一致


def compute_thematic_beta(
    prices_wide: pd.DataFrame,
    theme_heat_index: pd.Series,
    tickers: list[str],
    window: int = BETA_WINDOW,
    returns_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """返回长表：columns=[date, ticker, thematic_beta]

    returns_wide: 传入"股票收益-科技板块基准收益"的超额收益宽表（见
    benchmark.py），就是用超额收益算贝塔，剔除掉普通行业共振；不传就退回
    用 prices_wide 现算的原始收益（会有README里说的行业共振混淆问题，
    仅用于新旧对比）。
    """
    index_change = theme_heat_index.diff()
    returns = returns_wide if returns_wide is not None else prices_wide.pct_change()

    rows = []
    for t in tickers:
        if t not in returns.columns:
            continue
        stock_ret = returns[t]
        cov = stock_ret.rolling(window, min_periods=window // 2).cov(index_change)
        var = index_change.rolling(window, min_periods=window // 2).var()
        beta = cov / var
        rows.append(pd.DataFrame({"date": beta.index, "ticker": t, "thematic_beta": beta.values}))

    return pd.concat(rows, ignore_index=True)
