"""
逐日组合模拟回测，对应"个人小资金账户+重视止损/不想亏本"这个目标
（design_doc.md 12节新增：Phase 1因子研究之后的下一层，不是学术IC检验，
是"如果真按这个信号买卖，账户实际会经历什么"）。

**和 validate.py 的 IC/分层回测本质不同**：IC只测"排序准不准"，是横截面
统计量，不模拟真实持仓路径，测不出回撤、止损这类"path-dependent"（依赖
具体走过的路径，不是只看排序对不对）的东西。止损规则天然是path-dependent
的——同一只股票买入后先跌20%再涨50%、和先涨50%再跌20%，最终排序分数
一样，但止损规则下的实际结果天差地别，必须真正逐日模拟持仓才能测出来。

**策略设计**（先用最保守、最没有过拟合风险的版本）：
- 选股：只用动量因子（Phase 1精修里唯一在全部市值段都稳健为正的因子，
  见 design_doc.md 12节Fama-MacBeth回归结果），不掺别的因子，避免"止损
  规则效果"和"选股因子选择"两个变量搅在一起分不清楚。
- 组合规模：默认10只（对应用户明确说的1-2万美元小账户，如果用之前
  统计检验时的15-20只分散持仓，每只只有1000-2000美元，摊得太薄，止损
  的意义也被稀释）。
- 再平衡频率：默认21个交易日（约1个月）一次，跟动量因子本身63日/126日
  的信号更新节奏比，调仓不算频繁，个人手动操作也跟得上。
- 止损规则两种，可以分别开关或叠加：
  - **硬止损**（hard_stop_loss）：跌破买入价的X%，立即卖出。
  - **移动止损**（trailing_stop）：跌破"持有期内最高价"的Y%，立即卖出——
    这个直接对应用户说的"可以提前卖出，但最好不要亏本"：一旦浮盈过，
    移动止损能锁定部分利润再退出，不需要等到亏本才走。
  两种止损都是**每个交易日都检查**（不是只在再平衡日检查），一旦触发
  当天收盘价卖出，空出来的仓位空仓到下次再平衡（保守假设，不假设能
  立刻找到替代标的），这个空窗期收益按0%算，不是虚构一个替代收益。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str  # "rebalance_dropped" / "hard_stop" / "trailing_stop" / "backtest_end"

    @property
    def ret(self) -> float:
        """**是税前/未扣交易成本的毛收益**（用原始成交价算），只用来看
        "哪只股票贡献了多少涨跌"这类归因分析。真正反映到手净收益的是
        `BacktestResult.equity_curve`——那条线上已经在每笔交易时刻扣了
        `transaction_cost`，两者不是同一个东西，别混着解读。"""
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)

    def summary(self) -> dict:
        eq = self.equity_curve
        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else float("nan")
        running_max = eq.cummax()
        drawdown = eq / running_max - 1
        max_drawdown = drawdown.min()
        daily_ret = eq.pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else float("nan")

        closed = self.trades  # "backtest_end"（回测结束强制平仓）也算一笔真实交易，计入胜率统计
        n_trades = len(closed)
        wins = [t for t in closed if t.ret > 0]
        losses = [t for t in closed if t.ret <= 0]
        win_rate = len(wins) / n_trades if n_trades else float("nan")
        avg_win = np.mean([t.ret for t in wins]) if wins else float("nan")
        avg_loss = np.mean([t.ret for t in losses]) if losses else float("nan")
        stopped_out = [t for t in closed if t.exit_reason in ("hard_stop", "trailing_stop")]

        return {
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "pct_stopped_out": len(stopped_out) / n_trades if n_trades else float("nan"),
        }


def simulate_portfolio(
    prices_wide: pd.DataFrame,
    factor_panel: pd.DataFrame,
    factor_col: str,
    top_n: int = 10,
    rebalance_freq: int = 21,
    hard_stop_loss: float | None = None,
    trailing_stop: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_market_cap: float | None = None,
    transaction_cost: float = 0.0,
    regime_filter: pd.Series | None = None,
    max_drawdown_from_peak: float | None = None,
    drawdown_lookback: int = 126,
    position_scale: pd.Series | None = None,
) -> BacktestResult:
    """逐日模拟一个等权组合，返回净值曲线 + 每笔交易明细。

    regime_filter: 布尔型Series（index=date），True=允许持仓（"risk-on"），
    False=强制清仓、空仓等待（"risk-off"）。**这是个股止损解决不了的问题
    的补丁**：2000-2026全周期回测发现，止损规则在2008年金融危机里最大
    回撤依然到了-65%，原因是危机时全市场一起跌，止损把你从一只跌的股票
    里踢出来，下一次调仓换进来的"当前动量最强"的股票往往也在跌，止损
    没有真正的安全港可去。regime_filter是在**组合整体层面**加一道闸门
    （典型用法：大盘指数是否高于自己的200日均线，见 build_qqq_regime_filter），
    跟个股止损是互补关系——止损管"这只股票本身出了问题"，regime_filter
    管"整个市场都在下跌，先别进场"。逐日检查（不是只在再平衡日检查），
    一旦转为risk-off立即清仓，不再等下一次再平衡。

    hard_stop_loss / trailing_stop: 0.10 表示"跌破买入价/持有期最高价的
    10%就卖出"，None表示不启用这条规则。两个可以同时开、也可以只开一个、
    或者都不开（跑一个"无止损"的基准对照）。

    min_market_cap: **逐日、真实市值门槛**，不是"今天市值够格就让它整段
    历史都能被选中"（design_doc.md 12节：发现之前的股票池用当前市值筛选，
    等于用后见之明让还很小的公司在2016年就能被选中，2016年的投资者不可能
    提前知道这只股票以后会长这么大）。用Phase 1已经算出来的逐日真实市值
    （factor_panel里的market_cap列）做这道过滤：某只股票只有在**当天**
    市值真的达标，才允许被选进候选池。这只解决了"用今天市值倒推历史
    是否够格"这一层偏差，**不解决"AI产业链主题池本身是用2026年信息
    整理的"这一层更深的偏差**——后者是主题投资天然要面对的问题，不是
    这一步能修的。

    transaction_cost: 单边交易成本（比如0.001=0.1%），买入和卖出各收
    一次，近似佣金+滑点+买卖价差，不建模税（税率因人、因账户类型差异
    太大，留给用户自己按实际情况估算，不是本函数假装能算准的东西）。

    max_drawdown_from_peak: **新增（2026-07-23，用户实盘选股时发现）**——
    动量只看窗口首尾两个价格点，看不见中间走过的路径：一只从低点平滑
    涨上来的股票、和一只涨上来之后又从高点腰斩回来的股票，只要首尾
    价格一样，动量分数就一样，但后者往往是"追高接盘"更危险的情形。
    这个参数是在动量排名之上再加一道过滤：`drawdown_lookback`（默认
    跟动量同窗口126日）内，现价距这段时间最高点的回撤超过这个比例就
    排除出候选池，不管动量分数多高。0.20表示"距126日最高点跌超20%就
    排除"，None表示不启用（默认关闭，不影响任何已有的历史回测结果）。
    这条规则**没有被原始回测验证过**，加进来之后必须重新跑一遍完整
    历史对比，不能假设"看起来更安全"就等于"回测数字更好"。

    position_scale: **2026-07-28新增**，连续型仓位缩放（0-1之间的Series，
    index=date），回应云端routine那天研究出的Daniel & Moskowitz《Momentum
    Crashes》(2016)发现——regime_filter是"清仓/满仓"二元开关，论文提示
    这个二元开关可能恰好卡在"恐慌期后市场反弹"这个动量策略最容易漏接
    反弹的窗口。position_scale不清仓、不产生任何交易（因此不计交易成本），
    只是把每天的持仓收益按当天的scale值打折算进净值曲线——scale=0.5表示
    "当天的涨跌只影响一半的净值"，相当于经济上把一半仓位换成了现金但没有
    真的产生买卖动作，是这个回测框架里能加的最小改动。**跟regime_filter
    是独立的两个机制，可以同时开、也可以只开一个**，用来做A/B对比：
    见 build_vix_scale_filter。止损/选股逻辑完全不受这个参数影响——只
    影响"账户净值对价格变动的敏感度"，不影响"持有哪些股票"。
    """
    factor_wide = factor_panel[factor_col].unstack("ticker")
    market_cap_wide = factor_panel["market_cap"].unstack("ticker") if min_market_cap else None
    peak_wide = (
        prices_wide.rolling(drawdown_lookback, min_periods=drawdown_lookback).max()
        if max_drawdown_from_peak is not None else None
    )
    dates = prices_wide.index
    if start_date:
        dates = dates[dates >= start_date]
    if end_date:
        dates = dates[dates <= end_date]

    holdings: dict[str, dict] = {}  # ticker -> {entry_date, entry_price, peak_price}
    trades: list[Trade] = []
    equity = [1.0]
    equity_dates = [dates[0]]

    for i, today in enumerate(dates):
        # 1) 先用"昨天收盘后的持仓"结算今天的组合收益——必须在任何止损/
        #    调仓判断之前算，不然今天被止损/调出的仓位，触发止损那天本身
        #    的涨跌会被漏算（止损恰恰是因为今天跌破阈值才触发的，最容易
        #    漏掉的就是"导致触发的这一天"本身的跌幅，这是逐日模拟最容易
        #    踩的坑）。等权持仓，空仓部分按现金（0收益）算。
        if i > 0:
            prev_date = dates[i - 1]
            scale_today = float(position_scale.get(today, 1.0)) if position_scale is not None else 1.0
            day_ret = 0.0
            for ticker, pos in holdings.items():
                if ticker in prices_wide.columns:
                    prev_price = (
                        prices_wide.loc[prev_date, ticker] if prev_date >= pos["entry_date"] else pos["entry_price"]
                    )
                    curr_price = prices_wide.loc[today, ticker]
                    if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                        day_ret += (curr_price / prev_price - 1) / top_n * scale_today
            equity.append(equity[-1] * (1 + day_ret))
            equity_dates.append(today)

        # 2) 用今天的收盘价检查止损（不管是不是再平衡日），触发就今天卖出
        for ticker in list(holdings.keys()):
            if ticker not in prices_wide.columns or pd.isna(prices_wide.loc[today, ticker]):
                continue
            price = prices_wide.loc[today, ticker]
            pos = holdings[ticker]
            pos["peak_price"] = max(pos["peak_price"], price)

            exit_reason = None
            if hard_stop_loss is not None and price <= pos["entry_price"] * (1 - hard_stop_loss):
                exit_reason = "hard_stop"
            elif trailing_stop is not None and price <= pos["peak_price"] * (1 - trailing_stop):
                exit_reason = "trailing_stop"

            if exit_reason:
                trades.append(Trade(ticker, pos["entry_date"], pos["entry_price"], today, price, exit_reason))
                del holdings[ticker]
                equity[-1] *= (1 - transaction_cost / top_n)  # 卖出的交易成本，见函数docstring

        # 2.5) regime_filter：大盘转熊就立即清仓，不等下一次再平衡（逐日
        #      检查，跟止损同一优先级——这一步专门对付"全市场一起跌，止损
        #      也无处可逃"的系统性风险，见函数docstring）
        is_risk_on = True
        if regime_filter is not None:
            is_risk_on = bool(regime_filter.get(today, True))
            if not is_risk_on:
                for ticker in list(holdings.keys()):
                    if ticker in prices_wide.columns and pd.notna(prices_wide.loc[today, ticker]):
                        price = prices_wide.loc[today, ticker]
                        pos = holdings[ticker]
                        trades.append(Trade(ticker, pos["entry_date"], pos["entry_price"], today, price, "regime_risk_off"))
                        del holdings[ticker]
                        equity[-1] *= (1 - transaction_cost / top_n)

        # 3) 再平衡日：重新选股，调整持仓（用今天收盘价成交）
        if i % rebalance_freq == 0:
            if today in factor_wide.index and is_risk_on:
                scores = factor_wide.loc[today].dropna()
                available_prices = prices_wide.loc[today].dropna()
                scores = scores[scores.index.isin(available_prices.index)]
                if market_cap_wide is not None and today in market_cap_wide.index:
                    # point-in-time市值门槛：只有当天真实市值达标的股票才
                    # 允许入选，不是"今天市值够格就让它整段历史都能被选中"
                    eligible_cap = market_cap_wide.loc[today]
                    eligible_tickers = eligible_cap[eligible_cap >= min_market_cap].index
                    scores = scores[scores.index.isin(eligible_tickers)]
                if peak_wide is not None and today in peak_wide.index:
                    # 距drawdown_lookback日内最高点的回撤超过阈值就排除，见函数
                    # docstring：动量分数看不见"涨上来又腰斩回来"这种路径
                    peak_today = peak_wide.loc[today]
                    price_today = prices_wide.loc[today]
                    drawdown = price_today / peak_today - 1
                    ok_tickers = drawdown[drawdown >= -max_drawdown_from_peak].index
                    scores = scores[scores.index.isin(ok_tickers)]
                target = set(scores.sort_values(ascending=False).head(top_n).index)
            else:
                target = set(holdings.keys())

            for ticker in list(holdings.keys()):
                if ticker not in target:
                    price = prices_wide.loc[today, ticker]
                    pos = holdings[ticker]
                    trades.append(Trade(ticker, pos["entry_date"], pos["entry_price"], today, price, "rebalance_dropped"))
                    del holdings[ticker]
                    equity[-1] *= (1 - transaction_cost / top_n)

            for ticker in target:
                if ticker not in holdings:
                    price = prices_wide.loc[today, ticker]
                    if pd.isna(price):
                        continue
                    holdings[ticker] = {"entry_date": today, "entry_price": price, "peak_price": price}
                    equity[-1] *= (1 - transaction_cost / top_n)

    # 回测结束时强制平仓，计入交易记录（不然还持有的仓位不会被统计进win_rate）
    last_date = dates[-1]
    for ticker, pos in holdings.items():
        if ticker in prices_wide.columns:
            price = prices_wide.loc[last_date, ticker]
            if pd.notna(price):
                trades.append(Trade(ticker, pos["entry_date"], pos["entry_price"], last_date, price, "backtest_end"))

    equity_curve = pd.Series(equity, index=pd.DatetimeIndex(equity_dates))
    return BacktestResult(equity_curve=equity_curve, trades=trades)


def build_vix_scale_filter(
    vix_close: pd.Series,
    high_percentile: float = 0.80,
    min_scale: float = 0.5,
    min_history: int = 252,
) -> pd.Series:
    """VIX分位数驱动的连续仓位缩放，供 simulate_portfolio 的 position_scale
    参数使用（2026-07-28新增，回应design_doc.md之外的一次实盘触发的研究——
    Daniel & Moskowitz《Momentum Crashes》(2016)发现动量策略的亏损集中在
    "恐慌期"，且build_regime_filter那种二元清仓开关可能恰好卡在"恐慌期后
    市场反弹"这个动量最容易漏接的窗口）。

    做法：用**扩展窗口**（只用截至当天为止的历史，不用未来数据，避免前视
    偏差）算VIX今天的值在历史分布里的百分位排名，超过high_percentile就把
    仓位缩到min_scale，否则维持满仓（1.0）——是一个阶梯函数，不是论文里
    更复杂的"用动量收益均值/方差做回归预测"版本，是能在现有回测框架里
    低成本验证"连续缩放 vs 二元开关哪个更好"这个问题的最简单版本，不是
    对原论文的完整复现。

    high_percentile=0.80 / min_scale=0.5：直接对应云端routine那次研究
    提议的具体数字（"波动率超过80分位时缩到50%左右"），不是本次新调出来
    的参数——如果要做参数敏感性检查，应该在对比出"这个机制方向是否有效"
    之后再做，不要一开始就在两个自由维度上调，那是design_doc.md 2.4节
    警告的过拟合陷阱。

    min_history=252（约1年交易日）：百分位排名在样本太少时没有统计意义，
    历史不足一年时直接给满仓（1.0），不强行算一个基于几天数据的百分位。
    """
    expanding_rank = vix_close.expanding(min_periods=min_history).apply(
        lambda s: (s.iloc[-1] > s.iloc[:-1]).mean() if len(s) > 1 else 0.5
    )
    scale = pd.Series(1.0, index=vix_close.index)
    scale[expanding_rank >= high_percentile] = min_scale
    scale[expanding_rank.isna()] = 1.0  # 历史不足min_history时，见函数docstring
    return scale


def build_regime_filter(index_prices: pd.Series, window: int = 200) -> pd.Series:
    """经典的"200日均线"趋势过滤：指数收盘价高于自己的200日简单移动平均线
    就是risk-on，跌破就是risk-off。这是业界最标准、最不容易被质疑"调参数
    调出来的"版本（不是我们试了很多窗口挑出的最佳值，200日是这个方法本身
    几十年来的默认参数，选别的窗口天数才是真正的过拟合风险）。

    index_prices: 大盘指数（比如QQQ）的收盘价Series，index=date。
    """
    sma = index_prices.rolling(window, min_periods=window).mean()
    return (index_prices > sma).reindex(index_prices.index).fillna(False)
