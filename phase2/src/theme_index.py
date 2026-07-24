"""
构造"主题热度指数"，对应 design_doc.md 3.2节 机制B 的验证方法和 8节
"验证机制B"步骤1。

三路信号（新闻这一路可选，见 news_finnhub.py）：
1. L1核心股动量：NVDA/AMD/AVGO/TSM 等龙头股的滚动收益，代表"故事的
   源头有多热"。
2. 第二路信号有两个版本，二选一：
   - **旧版（有circularity问题）**：主题ETF成交量异动，SOXX/SMH 近期
     成交量相对自身历史的偏离程度，作为"资金关注度"的免费代理指标
     （design_doc.md 5.7节 Tier 4）。**已知问题**：SOXX/SMH本身持有
     AMAT/LRCX/KLAC/MU 等好几只被测下游股票，用它构造关注度信号再去测
     这些股票和信号的"共振"，存在自我印证风险（12节问题11）。这版本
     保留是为了能对比修复前后差异。
   - **新版（修复circularity）**：Google Trends搜索热度（见
     google_trends.py），关键词和任何被测股票都不重叠，是干净的替代
     信号，见 `google_trends_component`。
3. （可选）新闻提及量趋势：需要 FINNHUB_API_KEY，见 news_finnhub.py。

**关键的point-in-time细节**：三路信号标准化成 z-score 时，用的是"截至
当天为止"的滚动窗口均值/标准差（rolling, not全样本），不能用整个历史
（含未来）的均值方差去标准化今天的值——用全样本统计量做标准化本身就是
一种隐蔽的前视偏差，design_doc.md 2.5节强调的对齐原则在这里同样适用。
"""
from __future__ import annotations

import pandas as pd

ZSCORE_WINDOW = 252  # 用过去约1年的滚动统计量做标准化，避免用到未来数据


def _rolling_zscore(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    roll_mean = s.rolling(window, min_periods=window // 4).mean()
    roll_std = s.rolling(window, min_periods=window // 4).std()
    return (s - roll_mean) / roll_std


def l1_momentum_component(prices_wide: pd.DataFrame, l1_tickers: list[str], lookback: int = 21) -> pd.Series:
    """原始版本：直接用L1核心股的价格动量。**已知问题**：这没有剔除"整个
    科技板块都在涨"的普通行业/大盘共振，所以L1动量高的时候，可能只是
    大盘在涨，不一定是"AI叙事"特有的热度。建议优先用下面的
    `l1_excess_momentum_component`（用超额收益构造），这个函数保留是为了
    能对比"剔除行业共振前/后"的差异，见 phase2/README.md。"""
    available = [t for t in l1_tickers if t in prices_wide.columns]
    ret = prices_wide[available].pct_change(lookback)
    equal_weighted = ret.mean(axis=1)
    return _rolling_zscore(equal_weighted).rename("l1_momentum_z")


def l1_excess_momentum_component(
    excess_returns_wide: pd.DataFrame, l1_tickers: list[str], lookback: int = 21
) -> pd.Series:
    """用"L1核心股收益 - 泛科技板块基准收益"的超额收益构造动量分量，
    剔除掉"整个科技板块同涨同跌"这层普通共振，剩下的更接近"AI叙事"
    本身带来的额外热度。用滚动窗口内超额收益的加总近似"超额动量"——
    这里只是给排序打分用，Rank IC只关心排名先后，不要求精确复利计算，
    用求和代替复利是可接受的简化。"""
    available = [t for t in l1_tickers if t in excess_returns_wide.columns]
    rolling_excess = excess_returns_wide[available].rolling(lookback).sum()
    equal_weighted = rolling_excess.mean(axis=1)
    return _rolling_zscore(equal_weighted).rename("l1_excess_momentum_z")


def etf_relvol_component(volumes_wide: pd.DataFrame, etf_tickers: list[str]) -> pd.Series:
    available = [t for t in etf_tickers if t in volumes_wide.columns]
    relvol_cols = {}
    for t in available:
        short_avg = volumes_wide[t].rolling(21).mean()
        long_avg = volumes_wide[t].rolling(252, min_periods=63).mean()
        relvol_cols[t] = short_avg / long_avg - 1
    relvol_df = pd.DataFrame(relvol_cols)
    combined = relvol_df.mean(axis=1)
    return _rolling_zscore(combined).rename("etf_relvol_z")


def google_trends_component(trends_series: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """trends_series: Google Trends搜索热度（周频，见 google_trends.py 的拼接
    缩放逻辑），index是不规则的周频日期，值是0-100。reindex到交易日日历时
    用前向填充（ffill）——Google Trends这段历史区间只给得到周频，这是已知
    简化，不是bug（见 google_trends.py 模块docstring）。

    修复 design_doc.md 12节问题11指出的circularity：SOXX/SMH成交量本身和
    部分被测下游股票重叠，Google Trends搜索关键词（KEYWORDS见
    google_trends.py）不含任何被测公司名，是干净、不重叠的替代信号。
    """
    combined_idx = trading_days.union(trends_series.index)
    daily = trends_series.reindex(combined_idx).sort_index().ffill()
    daily = daily.reindex(trading_days)
    return _rolling_zscore(daily).rename("google_trends_z")


def news_heat_component(news_by_ticker: dict[str, pd.DataFrame], trading_days: pd.DatetimeIndex) -> pd.Series | None:
    """news_by_ticker: {ticker: DataFrame(date, news_count)}，来自 news_finnhub.py。
    没有任何数据时返回 None（调用方据此跳过这一路信号）。

    **实测发现的真限制**（design_doc.md 12节问题9实测结果）：Finnhub免费层
    `company-news` 接口不提供深历史，实测对NVDA查2020年1月整月返回0条，
    查2016-2026整个10年区间也只返回249条（且是最近的249条，不是按请求的
    from/to均匀覆盖），也就是免费层实际只能拿到最近几周的新闻。这导致
    这里算出来的原始trend序列只有最近一小段非NaN，经过下面
    `_rolling_zscore`（min_periods=63）过滤后，在覆盖多年的历史回测窗口里
    几乎全是NaN——**不是bug，是免费数据源的真实覆盖范围限制**，见
    pipeline.py 里对这种情况的显式告警。
    """
    if not news_by_ticker:
        return None

    series_list = []
    for ticker, df in news_by_ticker.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")["news_count"].reindex(trading_days).fillna(0)
        short_avg = s.rolling(7).mean()
        long_avg = s.rolling(30, min_periods=10).mean()
        trend = (short_avg - long_avg) / long_avg.replace(0, pd.NA)
        series_list.append(trend)

    if not series_list:
        return None
    combined = pd.concat(series_list, axis=1).mean(axis=1)
    return _rolling_zscore(combined).rename("news_heat_z")


def build_theme_heat_index(
    prices_wide: pd.DataFrame,
    volumes_wide: pd.DataFrame,
    l1_tickers: list[str],
    etf_tickers: list[str],
    news_by_ticker: dict[str, pd.DataFrame] | None = None,
    excess_returns_wide: pd.DataFrame | None = None,
    google_trends_series: pd.Series | None = None,
) -> pd.DataFrame:
    """返回 DataFrame：index=date，columns=[各分量z-score..., theme_heat_index]
    theme_heat_index = 可用分量的等权平均（哪路信号缺失就用剩下的，缺失
    路数在输出里可见，不隐藏）。

    excess_returns_wide: 如果传入了"股票收益-科技板块基准收益"的超额收益
    宽表（见 benchmark.py），L1动量分量会用剔除了普通行业共振的超额动量
    （l1_excess_momentum_component）；不传就退回原始版本（会被README和
    12节问题8指出的"未剔除行业共振"问题影响，仅用于新旧对比）。

    google_trends_series: 如果传入了Google Trends搜索热度（见
    google_trends.py），会用它替换掉有circularity问题的ETF成交量分量
    （12节问题11）；不传就退回旧的 etf_relvol_component，仅用于新旧对比。
    """
    if excess_returns_wide is not None:
        l1_component = l1_excess_momentum_component(excess_returns_wide, l1_tickers)
    else:
        l1_component = l1_momentum_component(prices_wide, l1_tickers)

    components = {"l1_momentum_z": l1_component}
    if google_trends_series is not None:
        components["google_trends_z"] = google_trends_component(google_trends_series, prices_wide.index)
    else:
        components["etf_relvol_z"] = etf_relvol_component(volumes_wide, etf_tickers)

    news_component = news_heat_component(news_by_ticker or {}, prices_wide.index)
    if news_component is not None:
        components["news_heat_z"] = news_component
    else:
        print("[theme_index] 未设置 FINNHUB_API_KEY，主题热度指数只用 L1动量 + ETF成交量异动两路信号")

    df = pd.DataFrame(components)
    df["theme_heat_index"] = df.mean(axis=1, skipna=True)
    return df
