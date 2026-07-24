"""
股票池定义，对应 design_doc.md 第5.7节的"子池A/子池B"设计。

子池A（科技基础池）：GICS 信息技术 + 通讯服务板块的大中盘代表性公司，
    用于跑通因子模型（动量/估值/质量）。来源见 build_universe.py：
    Nasdaq 官方筛选器 API，市值>20亿美元、美国注册，快照写入
    tech_base_pool.csv（约230只，第8节"第零步"建议的300-500只量级）。
    刷新方法：cd phase1 && python -m config.build_universe（建议每季度跑一次）。
子池B（AI产业链主题池）：对应 design_doc.md 3.2.1 节的产业链映射表，
    刻意包含非 GICS 科技分类的公司（电力、冷却基础设施等），人工维护，
    不随子池A自动刷新。
"""
import csv
from pathlib import Path

_CSV_PATH = Path(__file__).resolve().parent / "tech_base_pool.csv"

# 子池A：科技基础池。优先从 tech_base_pool.csv 读取（跑过 build_universe.py
# 后生成，约230只）；如果还没生成过，退回一份小规模硬编码清单，保证
# pipeline 在没跑 build_universe.py 时也能跑通。
_FALLBACK_TECH_BASE_POOL = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "QCOM", "TXN",
    "INTC", "IBM", "NOW", "INTU", "AMAT", "LRCX", "KLAC", "MU", "PANW", "SNPS",
    "CDNS", "ADI", "MRVL", "CTSH", "ON", "MCHP", "NXPI",
    "GOOGL", "META", "NFLX", "CMCSA", "TMUS", "EA", "TTWO",
]

if _CSV_PATH.exists():
    with _CSV_PATH.open() as f:
        TECH_BASE_POOL = [row["symbol"] for row in csv.DictReader(f)]
else:
    TECH_BASE_POOL = _FALLBACK_TECH_BASE_POOL

# 子池B：AI产业链主题池（对应 design_doc.md 3.2.1 节映射表）
AI_SUPPLY_CHAIN_POOL = {
    "L1": ["NVDA", "AMD", "AVGO", "TSM"],
    "L2": ["ASML", "AMAT", "LRCX", "KLAC", "MU", "AMKR"],
    "L3": ["VRT", "MOD", "NVT", "ETN", "ENTG", "MKSI", "LIN", "APD"],
    "L4": ["GEV", "ETN", "PWR", "CEG", "VST", "BE"],
}

AI_SUPPLY_CHAIN_FLAT = sorted({t for layer in AI_SUPPLY_CHAIN_POOL.values() for t in layer})

# **主题错配排除名单**（design_doc.md 12节：动量回测发现这几只股票贡献了
# 大量极端收益，但跟"AI产业链"主题毫无关系，只是Nasdaq把它们分类成
# "Technology"板块才被子池A捡进来）：
# - MARA（MARA Holdings）、WULF（TeraWulf）：比特币矿企，股价涨跌本质
#   跟着比特币价格走，不是芯片/数据中心需求
# - MSTR（Strategy Inc，原MicroStrategy）：名义上是商业智能软件公司，
#   实际已经变成一个杠杆持有比特币的载体，股价同样主要跟着比特币走
# 这几只不是"数据质量bug"，是真实存在的公司，只是**主题归类错了**——
# 留着它们会让动量类策略的回测结果被比特币的暴涨暴跌污染，看起来像是
# "AI选股策略很赚钱"，实际赚的是比特币的钱。目前只手动核实出这三只，
# 不是穷举，之后如果发现新的同类错配还需要继续补充。
THEME_MISMATCH_EXCLUDE = {"MARA", "MSTR", "WULF"}

# 最终候选池 = (子池A ∪ 子池B) - 主题错配名单（design_doc.md 5.7节：两个池子取并集，不是主次关系）
UNIVERSE = sorted((set(TECH_BASE_POOL) | set(AI_SUPPLY_CHAIN_FLAT)) - THEME_MISMATCH_EXCLUDE)
