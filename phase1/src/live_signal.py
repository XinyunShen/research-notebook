"""
"今天该买什么" 实盘信号脚本，对应 design_doc.md 12节问题18之后确认的
下一步（1参数敏感性检查→2本脚本→3纸上交易）。

**这不是新研究，是把 backtest.py 已经验证过的规则套到"今天"这一天**：
只用动量因子选股（Phase 1精修里唯一在全部市值段都稳健为正的因子），
参数照抄 README.md"参数敏感性检查"里验证过的稳健基准值，不在这里
重新调参——调参应该在 backtest.py 的历史回测里做，不该在实盘脚本里
现改，否则就是在用未来数据（这次的实盘结果）偷偷优化参数。

账户设定（用户2026-07-23确认）：小规模实验账户，4只等权持仓
（CAPITAL/TOP_N，见下方常量）——Schwab不支持零股，只能整股买入，仓位越少、每只分到
的钱越多，整股取整造成的实际权重偏差越小，这点和"个人操作不需要
像机构一样分散"的直觉刚好指向同一个方向，不是取舍。

**持仓状态追踪**（第二版新增）：第一版每次都是从零算"今天该买哪N只"，
不知道账户当前实际持有什么，21天后第二次跑时没法告诉你该换谁——
现在用 `output/live_portfolio_state.json` 记录当前持仓（ticker、
入场价、入场日、持有期内最高价），每次跑都先读上一次的状态，算出
SELL（止损触发或掉出前N名）/ HOLD（继续持有）/ BUY（新进前N名）
三类动作，而不是每次都重新报一遍完整买入清单。止损（硬止损/移动
止损）在这里也是逐次运行检查，不是逐日检查——**这仍然不等于backtest.py
里"每个交易日检查"的严格版本**，如果你几周不跑这个脚本，中途触发止损
也不会有人告诉你，真正逐日生效的止损请在下单时顺手在Schwab挂原生的
stop order / trailing stop order（见README讨论）。

**2026-07-28补充的一个真实教训**：2026-07-24买入的ATEX，B. Riley在
2026-06-04（买入前近7周）就已经把评级从Buy下调到Neutral（理由：900MHz
频谱变现存在不确定性），同期还有CFO辞职、内部人卖出股价下跌8.6%这些
公开可查的负面消息——**这条流水线从选股到下单，没有任何一步会去查
"这只股票最近有没有分析师下调评级/内部人减持/高管变动"，纯粹是价格
动量+估值快照（P/E、P/B）**。这不是bug，是这个脚本本身的能力边界：
它是纯Python脚本，没有联网搜索能力，查不了新闻。**这个检查必须由跑
这个脚本的人（或者在对话里协助的Claude Code）在下单前手动做**，本
脚本的输出只能提醒"该查"，做不到自动查——见下方main()末尾的打印提醒，
不要跳过。

用法：
    cd phase1 && source .venv/bin/activate && python -m src.live_signal
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config.universe import UNIVERSE  # noqa: E402
from src import backtest, data_fetch, factors, validate  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
LIVE_PRICE_CACHE = DATA_DIR / "live_prices.parquet"  # 独立缓存路径，
# 绝不能用 data_fetch.PRICE_CACHE（design_doc.md 记录过的教训：曾经因为
# 传了同一个默认cache_path，一次benchmark拉取直接把305只股票的主缓存
# 覆盖成只剩2只，此后所有额外拉取都必须用独立cache_path）。
STATE_PATH = OUTPUT_DIR / "live_portfolio_state.json"

# --- 照抄 README.md"参数敏感性检查"验证过的稳健基准值，不在这里重调 ---
TOP_N = 4
CAPITAL = 20_000
ALLOCATION_PER_POSITION = CAPITAL / TOP_N
HARD_STOP_LOSS = 0.10
TRAILING_STOP = 0.15
REBALANCE_FREQ_DAYS = 21
MIN_MARKET_CAP = 900_000_000
MOMENTUM_LOOKBACK = 126
REGIME_BENCHMARK = "QQQ"
REGIME_WINDOW = 200
EXTREME_MOMENTUM_THRESHOLD = 1.00  # 126日动量超过100%时打警告，提醒手动核实
MAX_DRAWDOWN_FROM_PEAK = 0.15  # 距126日内最高点回撤超过15%就排除，见backtest.py
# simulate_portfolio的max_drawdown_from_peak文档：2026-07-23用户实盘选股时发现，
# 动量只看首尾两点看不见"涨上来又腰斩回来"这种路径，加这道过滤后回测验证过
# （见design_doc.md/README.md"新增：距高点回撤过滤"一节）——4只/10只两种持仓
# 配置、15/20/25%三个附近阈值都一致改善收益和回撤，不是挑出来的巧合数字。
DRAWDOWN_LOOKBACK = 126  # 跟动量同窗口，不引入额外的、没理由的独立参数

# 跨账户wash sale协调（见 portfolio/README.md"跨账户注意"一节）：如果
# account_1/account_2的INTC税损收割批次已经卖出，30天内两个账户+这个
# 新账户都不能再买INTC，不然已实现的损失会被追溯作废。这里不做成硬性
# 过滤（研究脚本应该保持通用），只在INTC真的入选时打印警告，卖不卖
# 由用户根据当前是否处在wash sale窗口内自己判断。
WASH_SALE_WATCH_TICKERS = {"INTC"}


def load_state() -> dict:
    if STATE_PATH.exists():
        with STATE_PATH.open() as f:
            return json.load(f)
    return {"as_of": None, "positions": {}}


def save_state(as_of: str, positions: dict):
    with STATE_PATH.open("w") as f:
        json.dump({"as_of": as_of, "positions": positions}, f, indent=2)


def fmt_extreme(score: float) -> str:
    return "  ⚠️ 数值偏大，建议核实是否数据异常（拆股/spin-off等）再当真" \
        if abs(score) >= EXTREME_MOMENTUM_THRESHOLD else ""


def main():
    today = date.today()
    fetch_start = today - timedelta(days=730)  # 覆盖200日均线regime过滤 + 126日动量，留足缓冲

    state = load_state()
    current_positions = state["positions"]
    is_first_run = not current_positions

    print(f"[live_signal] 股票池共 {len(UNIVERSE)} 只，拉取 {fetch_start} 至今的最新行情 ...")
    tickers_to_fetch = sorted(set(UNIVERSE + [REGIME_BENCHMARK] + list(current_positions.keys())))
    prices = data_fetch.fetch_prices(
        tickers_to_fetch, start=str(fetch_start), end=str(today),
        force_refresh=True, cache_path=LIVE_PRICE_CACHE,
    )
    latest_date = prices["date"].max()
    px_today = prices[prices["date"] == latest_date].set_index("ticker")["adj_close"]
    print(f"[live_signal] 最新行情日期: {latest_date.date()}（如果不是今天/昨天的交易日，"
          f"检查是不是还没收盘或者数据源延迟）")

    # --- 动量因子（用最新价格现算，不用因子面板里可能滞后的缓存值） ---
    mom = factors.compute_momentum(prices, lookback=MOMENTUM_LOOKBACK)
    mom_today = mom[mom["date"] == latest_date].set_index("ticker")["momentum"].dropna()

    # --- 流动性过滤（21日滚动美元成交额 >= $1M，见 validate.MIN_DOLLAR_VOLUME） ---
    liq = validate.liquidity_mask(prices)
    liq_today = liq[liq["date"] == latest_date].set_index("ticker")["liquid"]

    # --- 市值过滤：复用pipeline最近一次算出的市值（shares_outstanding变化很慢，
    #     不需要为了这一步重新拉SEC EDGAR；README.md"组合回测"一节的教训是
    #     "不能用今天市值倒推历史资格"，这里是反过来判断"今天"本身够不够格，
    #     用最近一次算出的市值做近似完全没有point-in-time问题） ---
    factor_panel_path = OUTPUT_DIR / "factor_panel.parquet"
    if factor_panel_path.exists():
        fp = pd.read_parquet(factor_panel_path)
        fp_latest_date = fp.index.get_level_values("date").max()
        latest_mcap = fp["market_cap"].dropna().groupby(level="ticker").last()
        staleness_days = (pd.Timestamp(today) - fp_latest_date).days
        print(f"[live_signal] 市值数据来自 output/factor_panel.parquet 最近一次pipeline结果"
              f"（{fp_latest_date.date()}，{staleness_days}天前），近似当前市值")
        if staleness_days > 90:
            print(f"[live_signal] ⚠️  市值数据已经{staleness_days}天没更新了，"
                  f"建议先跑一遍 `python -m src.pipeline` 刷新，股本变化（增发/回购/分拆）"
                  f"不会自动反映在这个近似值里")
    else:
        latest_mcap = pd.Series(dtype=float)
        print("[live_signal] WARNING: 没找到 output/factor_panel.parquet，"
              "无法做市值过滤，先跑一遍 `python -m src.pipeline` 生成")

    # --- 距高点回撤过滤（2026-07-23新增，见README"新增：距高点回撤过滤"一节的
    #     回测验证）：126日内现价距最高点回撤超过15%就排除，不管动量分数多高 ---
    price_wide_recent = prices.pivot(index="date", columns="ticker", values="adj_close")
    peak_recent = price_wide_recent.rolling(DRAWDOWN_LOOKBACK, min_periods=DRAWDOWN_LOOKBACK).max()
    drawdown_today = (price_wide_recent.loc[latest_date] / peak_recent.loc[latest_date] - 1)

    eligible = mom_today.index
    eligible = eligible[liq_today.reindex(eligible).fillna(False)]
    eligible = eligible[latest_mcap.reindex(eligible).fillna(0) >= MIN_MARKET_CAP]
    excluded_by_drawdown = eligible[drawdown_today.reindex(eligible).fillna(-1) < -MAX_DRAWDOWN_FROM_PEAK]
    eligible = eligible[drawdown_today.reindex(eligible).fillna(-1) >= -MAX_DRAWDOWN_FROM_PEAK]

    ranked = mom_today.reindex(eligible).dropna().sort_values(ascending=False)

    # --- Regime过滤：QQQ收盘价是否高于自己的200日均线 ---
    qqq_close = prices[prices["ticker"] == REGIME_BENCHMARK].set_index("date")["adj_close"].sort_index()
    regime = backtest.build_regime_filter(qqq_close, window=REGIME_WINDOW)
    is_risk_on = bool(regime.reindex([latest_date]).iloc[0]) if latest_date in regime.index else False
    qqq_sma = qqq_close.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).mean().loc[latest_date]
    qqq_last = qqq_close.loc[latest_date]

    print(f"\n[live_signal] === Regime状态（{REGIME_BENCHMARK} vs {REGIME_WINDOW}日均线）===")
    print(f"  {REGIME_BENCHMARK}收盘 {qqq_last:.2f} vs {REGIME_WINDOW}日均线 {qqq_sma:.2f} "
          f"→ {'RISK-ON（正常持仓）' if is_risk_on else 'RISK-OFF（策略建议空仓，不新开仓）'}")

    if len(excluded_by_drawdown) > 0:
        excl_ranked = mom_today.reindex(excluded_by_drawdown).dropna().sort_values(ascending=False)
        print(f"\n[live_signal] === 距高点回撤过滤排除了{len(excl_ranked)}只（动量高但距126日高点跌超"
              f"{MAX_DRAWDOWN_FROM_PEAK*100:.0f}%）===")
        for t, score in excl_ranked.head(10).items():
            px = px_today.get(t, float("nan"))
            dd = drawdown_today.get(t, float("nan"))
            print(f"  {t:<8}动量{score*100:>7.1f}%  距高点回撤{dd*100:>6.1f}%  现价${px:.2f}")

    print(f"\n[live_signal] === 动量排名前15（{len(ranked)}只通过流动性+市值+距高点回撤门槛）===")
    print(f"  {'排名':<4}{'代码':<8}{'动量({}日)'.format(MOMENTUM_LOOKBACK):<14}{'最新收盘价':<10}")
    for i, (t, score) in enumerate(ranked.head(15).items(), 1):
        px = px_today.get(t, float("nan"))
        marker = ("  ← wash sale watch" if t in WASH_SALE_WATCH_TICKERS else "") + fmt_extreme(score)
        print(f"  {i:<4}{t:<8}{score*100:>8.1f}%      ${px:>8.2f}{marker}")

    # --- 1) 用当前持仓 + 今天的价格检查止损（逐次运行检查，不是逐日检查，见docstring） ---
    stop_sells = {}
    for t, pos in current_positions.items():
        px = px_today.get(t)
        if px is None or pd.isna(px):
            continue
        pos["peak_price"] = max(pos["peak_price"], px)
        if px <= pos["entry_price"] * (1 - HARD_STOP_LOSS):
            stop_sells[t] = ("hard_stop", px)
        elif px <= pos["peak_price"] * (1 - TRAILING_STOP):
            stop_sells[t] = ("trailing_stop", px)

    # --- 2) regime risk-off：强制清仓所有持仓，不进入正常调仓逻辑 ---
    if not is_risk_on:
        print(f"\n[live_signal] 大盘处于risk-off状态，策略建议**全部清仓、持有现金**，"
              f"不给出买入清单。等{REGIME_BENCHMARK}重新升破{REGIME_WINDOW}日均线再看。")
        if current_positions:
            print(f"[live_signal] === 需要卖出（regime risk-off强制清仓）===")
            for t, pos in current_positions.items():
                px = px_today.get(t, pos["entry_price"])
                print(f"  卖出 {t}: {pos['shares']}股 @ ${px:.2f}（入场价${pos['entry_price']:.2f}）")
        save_state(str(latest_date.date()), {})
        return

    # --- 3) 正常调仓：先扣掉止损触发的，剩下的跟新一轮target做diff ---
    remaining = {t: p for t, p in current_positions.items() if t not in stop_sells}
    target_list = list(ranked.head(TOP_N).index)
    target = set(target_list)

    rebalance_sells = {t: remaining[t] for t in remaining if t not in target}
    holds = {t: remaining[t] for t in remaining if t in target}
    buys = [t for t in target_list if t not in remaining]

    if stop_sells:
        print(f"\n[live_signal] === 卖出：止损触发 ===")
        for t, (reason, px) in stop_sells.items():
            pos = current_positions[t]
            reason_cn = "硬止损(-10%)" if reason == "hard_stop" else "移动止损(-15%)"
            print(f"  卖出 {t}: {pos['shares']}股 @ ${px:.2f}（入场价${pos['entry_price']:.2f}，触发{reason_cn}）")

    if rebalance_sells:
        print(f"\n[live_signal] === 卖出：掉出动量前{TOP_N}名 ===")
        for t, pos in rebalance_sells.items():
            px = px_today.get(t, pos["entry_price"])
            print(f"  卖出 {t}: {pos['shares']}股 @ ${px:.2f}（入场价${pos['entry_price']:.2f}）")

    if holds:
        print(f"\n[live_signal] === 继续持有 ===")
        for t, pos in holds.items():
            px = px_today.get(t, pos["entry_price"])
            print(f"  持有 {t}: {pos['shares']}股，入场价${pos['entry_price']:.2f}，"
                  f"现价${px:.2f}，持有期最高价${pos['peak_price']:.2f}")

    if buys:
        label = "首次建仓" if is_first_run else "新进前列，买入"
        print(f"\n[live_signal] === 买入：{label} ===")
        print(f"[live_signal] ⚠️  下单前必读（2026-07-28教训，见脚本顶部docstring）：这份名单"
              f"纯粹是价格动量+估值快照，没有查过近期有没有分析师下调评级/内部人减持/高管变动/"
              f"业绩指引下修——ATEX那次这些消息在买入前7周就已经公开，但流水线完全没查。"
              f"对下面每一只，先手动搜一下'TICKER analyst downgrade OR insider selling OR "
              f"guidance cut'，有实质性负面消息的，不要只因为动量分数高就买。")
    new_positions = dict(holds)
    for t in buys:
        px = px_today.get(t)
        score = ranked.get(t, float("nan"))
        shares = round(ALLOCATION_PER_POSITION / px)
        actual_dollars = shares * px
        deviation = actual_dollars / ALLOCATION_PER_POSITION - 1
        hard_stop_price = px * (1 - HARD_STOP_LOSS)
        marker = fmt_extreme(score)
        print(f"  买入 {t}: {shares}股 @ ${px:.2f} = ${actual_dollars:,.2f} "
              f"(目标${ALLOCATION_PER_POSITION:,.0f}的{deviation*100:+.1f}%)，"
              f"动量{score*100:.1f}%，硬止损价${hard_stop_price:.2f}(-{HARD_STOP_LOSS*100:.0f}%){marker}")
        if t in WASH_SALE_WATCH_TICKERS:
            print(f"     ⚠️  {t}在wash sale watch名单里——如果account_1/account_2的{t}税损收割"
                  f"批次已经卖出且还在30天窗口内，买这个会让已实现的损失被追溯作废，"
                  f"买之前先去portfolio/README.md确认窗口是否已经过了。")
        new_positions[t] = {
            "shares": shares, "entry_date": str(latest_date.date()),
            "entry_price": px, "peak_price": px,
        }

    if not stop_sells and not rebalance_sells and not buys and not is_first_run:
        print(f"\n[live_signal] 无变化——当前{len(holds)}只持仓都还在动量前{TOP_N}名，没有触发止损，不用动。")

    save_state(str(latest_date.date()), new_positions)

    rows = [
        {"date": str(latest_date.date()), "ticker": t, "action": "hold" if t in holds and t not in buys else "buy",
         "shares": p["shares"], "entry_price": p["entry_price"], "current_price": px_today.get(t, p["entry_price"]),
         "hard_stop_price": p["entry_price"] * (1 - HARD_STOP_LOSS), "trailing_stop_pct": TRAILING_STOP}
        for t, p in new_positions.items()
    ]
    out_path = OUTPUT_DIR / f"live_signal_{latest_date.date()}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[live_signal] 移动止损{TRAILING_STOP*100:.0f}%和硬止损{HARD_STOP_LOSS*100:.0f}%已经在这次运行里"
          f"检查过了，但只在你运行这个脚本的这一刻生效——不是逐日盯盘，两次运行之间触发的止损不会有人提醒你，"
          f"建议下单时顺手在Schwab挂原生的stop order/trailing stop order做真正的逐日执行。")
    print(f"[live_signal] 建议下次重新跑这个脚本的时间：约{REBALANCE_FREQ_DAYS}个交易日后"
          f"（对应backtest.py验证过的再平衡频率），不要因为手痒天天跑天天换仓。")
    print(f"[live_signal] 当前持仓状态已存到 {STATE_PATH}，结果已存到 {out_path}")


if __name__ == "__main__":
    main()
