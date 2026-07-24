"""
Phase 2 用到的产业链分层清单，对应 design_doc.md 3.2.1节映射表。

有意和 phase1/config/universe.py 里的 AI_SUPPLY_CHAIN_POOL 保持完全一致
（数据来源相同，都是3.2.1节那张表），这里独立复制一份而不是跨包import，
是为了让 phase2 不依赖 phase1 内部结构，两边各自独立可跑（design_doc.md
强调的"避免过度设计"原则：一份20行的静态数据字典，复制比硬耦合两个
包的import路径更简单可靠）。刷新3.2.1节映射表时，两处需要同步更新。
"""

AI_SUPPLY_CHAIN_POOL = {
    "L1": ["NVDA", "AMD", "AVGO", "TSM"],
    "L2": ["ASML", "AMAT", "LRCX", "KLAC", "MU", "AMKR"],
    "L3": ["VRT", "MOD", "NVT", "ETN", "ENTG", "MKSI", "LIN", "APD"],
    "L4": ["GEV", "ETN", "PWR", "CEG", "VST", "BE"],
}

# L2-L4 是我们要测"主题贝塔"的下游候选股（L1是主题热度指数的来源之一，
# 测L1对自己的beta没有意义，见 thematic_beta 的用法）
DOWNSTREAM_TICKERS = sorted(
    {t for layer in ("L2", "L3", "L4") for t in AI_SUPPLY_CHAIN_POOL[layer]}
)

L1_TICKERS = AI_SUPPLY_CHAIN_POOL["L1"]

# 主题ETF，用作 Tier 4 资金关注度代理（design_doc.md 5.7节）
ETF_TICKERS = ["SOXX", "SMH"]

ALL_TICKERS = sorted(set(L1_TICKERS) | set(DOWNSTREAM_TICKERS) | set(ETF_TICKERS))
