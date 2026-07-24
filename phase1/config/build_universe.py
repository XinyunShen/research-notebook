"""
维护脚本：从 Nasdaq 官方股票筛选器 API（免费、无需 key）拉取"科技+电信"
板块、市值>10亿美元、美国注册的公司，生成 tech_base_pool.csv。

这是 design_doc.md 5.7节"科技基础池"的数据来源，对应第10节"定期刷新"
的要求 —— 建议每季度重跑一次这个脚本，不要一次性用一年。

**门槛从20亿降到9亿美元**（design_doc.md 12节问题12：Phase 1精修，
扩大股票池到300-500只）：原20亿门槛只有约245家（去重前）。10亿门槛
去重后只有299只，堪堪卡在第8节"第零步"建议区间(300-500)的下限边缘，
不留余量；9亿门槛去重前306家，去重后有安全冗余，明确落在区间内，
同时不像7.5亿以下那样明显引入更多小盘/低流动性噪音。

用法：
    cd phase1 && source .venv/bin/activate && python -m config.build_universe
"""
from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
OUT_CSV = Path(__file__).resolve().parent / "tech_base_pool.csv"
MIN_MARKET_CAP = 900_000_000

# Nasdaq screener的"sector"字段是它自己的分类口径，不是严格GICS，
# 混进了少数明显不是科技/通讯的工业股/综合企业（screener数据质量限制），
# 手工排除这些已知的误分类，其余保留（design_doc.md已说明：小规模噪音
# 可接受，先跑通规模化流水线，比手工逐个精修更重要）。
KNOWN_MISCLASSIFIED = {
    "GE", "NEE", "EMR", "OTIS", "HUBB", "ROP", "JBL", "ARW", "AVT",
    "SANM", "PLXS", "CDW", "SNX", "ESE", "SPCX",
}

BAD_NAME_PATTERN = re.compile(
    r"Depositary|Warrant|Right|Unit\b|Preferred|Notes|Acquisition Corp|Trust Units|Ordinary Shares",
    re.I,
)


def _market_cap(row: dict) -> float:
    try:
        return float(row["marketCap"])
    except (ValueError, TypeError):
        return 0.0


def _company_key(name: str) -> str:
    """去掉股票类别后缀，让同一家公司的多个股票类别（比如Alphabet的
    GOOGL/GOOG）只保留市值最大的那一个。"""
    return re.sub(r"\s+(Class [A-Z]|Common Stock.*|Capital Stock.*)$", "", name).strip()


def fetch_and_build() -> list[dict]:
    r = requests.get(
        "https://api.nasdaq.com/api/screener/stocks",
        params={"tableonly": "true", "limit": "8000", "offset": "0", "download": "true"},
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()["data"]["rows"]

    candidates = [
        row for row in rows
        if row.get("sector") in ("Technology", "Telecommunications")
        and row.get("country") == "United States"
        and _market_cap(row) > MIN_MARKET_CAP
        and row["symbol"] not in KNOWN_MISCLASSIFIED
        and not BAD_NAME_PATTERN.search(row["name"])
    ]

    dedup: dict[str, dict] = {}
    for row in sorted(candidates, key=_market_cap, reverse=True):
        key = _company_key(row["name"])
        if key not in dedup:
            dedup[key] = row

    return sorted(dedup.values(), key=_market_cap, reverse=True)


def main():
    final = fetch_and_build()
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "name", "sector", "market_cap", "as_of"])
        today = date.today().isoformat()
        for row in final:
            writer.writerow([row["symbol"], row["name"], row["sector"], int(_market_cap(row)), today])
    print(f"[build_universe] 写入 {len(final)} 只股票到 {OUT_CSV}（快照日期 {date.today().isoformat()}）")


if __name__ == "__main__":
    main()
