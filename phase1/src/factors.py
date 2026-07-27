"""
因子计算，对应 design_doc.md 3.1节 + 2.5节（前视偏差对齐）。

实现4个因子，对应 design_doc.md 第4节推荐的"估值+质量+动量+资金流"组合
（design_doc.md 12节问题12：Phase 1精修补齐了资金流这一路）：
- 动量 Momentum：过去126个交易日收益率，纯价格数据，没有前视偏差问题。
- 质量 Quality：ROE（TTM净利润 / 最新股东权益），来自 SEC EDGAR，
  用 filed 日期做 point-in-time 对齐。
- 价值 Value：账面市值比 Book-to-Market（股东权益 / 市值），同样用
  filed 日期对齐；用 B/M 而不是简单 PE，是因为 B/M 是学术上更标准、
  对亏损公司更稳健的价值因子构造方式（Fama-French 经典做法）。
- 资金流 Fund Flow：内部人净买入占流通股比例，来自 SEC EDGAR Form 4，
  见 insider.py。

**2026-07-24新增2个（design_doc.md 12节：单独测超大盘ROE发现信号偏弱后，
回应"真实投资者买入前还会看哪些数据"这个问题，找跟现有因子重叠度更低
的角度，而不是继续在ROE本身上打转）**：
- fcf_yield：TTM自由现金流（经营性现金流-资本开支）/ 市值，比B/M更贴近
  "真金白银"的价值信号，对会计政策（折旧/摊销假设）比净利润更不敏感。
- margin_trend：TTM营业利润率相对一年前同口径TTM营业利润率的变化，跟
  ROE是不同的角度——ROE看"现在赚不赚钱"，这个看"利润率在变好还是变差"。

关键工程细节（避免前视偏差）：用 pd.merge_asof(..., direction="backward")
把"某一天"匹配到"当天为止最新已公开申报"的财务数据，绝不会用到未来才
披露的数字。
"""
from __future__ import annotations

import time

import pandas as pd

from . import edgar, insider


def compute_momentum(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """过去 lookback 个交易日的收益率，按 ticker 分组计算。"""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["momentum"] = df.groupby("ticker")["adj_close"].transform(
        lambda s: s / s.shift(lookback) - 1
    )
    return df[["date", "ticker", "momentum"]]


def compute_fundamental_factors(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """对每只股票拉取 SEC EDGAR 基本面面板，再用 merge_asof 对齐到每个交易日，
    算出 ROE（质量因子）和 Book-to-Market（价值因子）。

    跳过规则：外国私人发行人（比如 TSM、ASML）大多用 20-F 年报，不走
    10-Q/10-K 的 XBRL 报送节奏，这里会自动跳过并打印提示 —— 这是已知
    限制，不是 bug，design_doc.md 5.2节已经说明 SEC EDGAR 是"美股"官方
    数据源，对 ADR/外国发行人覆盖天然更弱。
    """
    ticker_to_cik = edgar.load_ticker_to_cik()
    all_rows = []

    for t in tickers:
        cik = ticker_to_cik.get(t.upper())
        if cik is None:
            print(f"[factors] WARNING: {t} not found in SEC ticker map, skipping fundamentals")
            continue

        panel = edgar.build_fundamentals_panel(t, cik)
        if panel is None or panel.empty:
            print(f"[factors] WARNING: no usable XBRL fundamentals for {t} (likely foreign private issuer / 20-F filer), skipping")
            continue

        px = prices[prices["ticker"] == t][["date", "adj_close"]].sort_values("date").copy()
        panel = panel.rename(columns={"filed": "date"}).sort_values("date")

        merged = pd.merge_asof(
            px, panel, on="date", direction="backward",
            # 只信任"至少已公开1天"的数据，避免当天盘前刚披露就当成盘中已知信息
        )
        merged["ticker"] = t
        market_cap = merged["shares_outstanding"] * merged["adj_close"]
        # 和 edgar.py 里 ROE 的处理保持一致：股东权益<=0或市值算不出来时，
        # 账面市值比没有经济意义，置为空（用 np.nan 而不是 pd.NA，保持纯
        # float64 dtype，兼容 alphalens 内部的 numpy isfinite 运算）。
        valid = (merged["stockholders_equity"] > 0) & (market_cap > 0)
        merged["book_to_market"] = float("nan")
        merged.loc[valid, "book_to_market"] = merged.loc[valid, "stockholders_equity"] / market_cap[valid]
        merged["book_to_market"] = merged["book_to_market"].astype("float64")
        # 市值本身也保留下来，作为逐日、逐股票的真实市值时间序列（不是
        # "今天的市值"套用到整个历史），用于按市值分段做IC检验（design_doc.md
        # 12节问题15，排查book_to_market扩池后变弱的原因时发现：同一只股票
        # 十年前的市值和今天可能差好几个量级，不能用"今天市值"倒推它十年前
        # 算不算大盘股）。市值算不出来（market_cap<=0，比如shares_outstanding
        # 缺失）时留空，不强行填0。
        merged["market_cap"] = market_cap.where(market_cap > 0, float("nan")).astype("float64")

        # --- fcf_yield（2026-07-24新增）：TTM自由现金流 / 市值，跟B/M一样
        #     需要价格数据才能算出比率，所以放在这里而不是edgar.py ---
        merged["fcf_yield"] = float("nan")
        valid_fcf = market_cap > 0
        merged.loc[valid_fcf, "fcf_yield"] = merged.loc[valid_fcf, "fcf_ttm"] / market_cap[valid_fcf]
        merged["fcf_yield"] = merged["fcf_yield"].astype("float64")

        all_rows.append(
            merged[["date", "ticker", "roe_ttm", "book_to_market", "market_cap", "fcf_yield", "margin_trend"]]
        )

    if not all_rows:
        return pd.DataFrame(
            columns=["date", "ticker", "roe_ttm", "book_to_market", "market_cap", "fcf_yield", "margin_trend"]
        )
    result = pd.concat(all_rows, ignore_index=True)
    # 防御性兜底：任何遗漏路径产生的 inf/-inf 一律当缺失值处理，不能流入
    # 后续的 IC 检验/分层回测，否则会让统计量彻底失真（比如 mean() 变成 -inf）。
    for col in ["roe_ttm", "book_to_market", "fcf_yield", "margin_trend"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").replace(
            [float("inf"), float("-inf")], float("nan")
        )
        result[col] = _winsorize_cross_sectional(result, col)
    return result


def _winsorize_cross_sectional(df: pd.DataFrame, col: str, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按日期做横截面winsorize（1%/99%分位数截尾），不是丢弃极端值而是
    压缩到分位数边界。排查 ROE 因子反常方向（design_doc.md 12节问题12）
    时发现的真问题：股东权益接近0的公司（大量回购的科技股常见，比如
    Ubiquiti大量回购后权益极小、TTM净利润/权益算出ROE=246即24600%）
    会产生和"真实盈利质量"无关的极端比率，36只股票在样本里出现过
    |ROE|>500%，这类极端值虽然Rank IC本身是按排名算、单个极端值理论上
    不该扭曲相关系数，但这些公司的极端值会在财报未更新的整个季度里
    持续占据分层回测的最高/最低组，实际观察到会拖累分层收益的单调性。
    B/M用同样处理保持方法一致，虽然B/M本身分布没有ROE那么病态。
    """
    return df.groupby("date")[col].transform(
        lambda s: s.clip(s.quantile(lower), s.quantile(upper)) if s.notna().sum() >= 5 else s
    )


def build_factor_panel(prices: pd.DataFrame, tickers: list[str], start: str, include_insider: bool = True) -> pd.DataFrame:
    """把动量 + 质量 + 价值 + 资金流四个因子合并成一张宽表：index=(date, ticker)

    include_insider: Form 4 内部人交易因子需要逐笔filing拉XML，300+只
    股票全量跑要几十分钟到一个多小时（见 insider.py），默认开启但留了
    开关，方便只想快速跑通前3个因子时跳过（比如调试阶段）。
    """
    mom = compute_momentum(prices)
    fund = compute_fundamental_factors(prices, tickers)

    panel = mom.merge(fund, on=["date", "ticker"], how="left")

    if include_insider:
        flow = insider.compute_insider_flow_factor(prices, tickers, start=start)
        panel = panel.merge(flow, on=["date", "ticker"], how="left")

    return panel.set_index(["date", "ticker"]).sort_index()
