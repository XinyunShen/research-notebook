"""
构造"科技板块基准"，用来把 thematic_beta 里的普通行业共振剔除掉——
回应 phase2/README.md 和 design_doc.md 12节问题8指出的方法论缺陷：
半导体设备股本来就和芯片龙头同涨同跌，这是行业贝塔，不是AI叙事贝塔，
之前的版本没有把这部分剔除。

做法：用 Phase 1 那244只股票的科技基础池（子池A），**排除掉**3.2.1节
产业链池里的所有股票（L1-L4，我们正在测的对象和它的信号来源），
剩下的作为一个干净、不重叠的"泛科技板块"基准，等权合成每日收益率。
然后 pipeline.py 会用 "股票收益 - 基准收益" 得到超额收益，再拿超额
收益去构造主题热度指数、算主题贝塔，这样测出来的才是"超出正常科技股
共同波动之外"的、更接近AI叙事本身的共振强度。

**仍然不完美**：SOXX/SMH（用于成交量代理资金流那一路信号）本身还是
会和部分下游候选股重叠，这次先不处理这个circularity问题（见
phase2/README.md 已记录的已知限制）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

PHASE1_ROOT = Path(__file__).resolve().parent.parent.parent / "phase1"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_benchmark_universe(exclude_tickers: set[str]) -> list[str]:
    """Phase 1 科技基础池，排除掉我们正在测的AI产业链股票（L1-L4）。"""
    universe_mod = _load_module("phase1_universe", PHASE1_ROOT / "config" / "universe.py")
    return sorted(set(universe_mod.TECH_BASE_POOL) - set(exclude_tickers))


def fetch_benchmark_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """直接复用 Phase 1 已经缓存好的行情数据（同一批股票 Phase 1 已经拉过），
    不需要重新联网下载。"""
    data_fetch = _load_module("phase1_data_fetch", PHASE1_ROOT / "src" / "data_fetch.py")
    return data_fetch.fetch_prices(tickers, start=start, end=end)  # 默认走 Phase 1 自己的缓存路径


def compute_benchmark_returns(prices_long: pd.DataFrame) -> pd.Series:
    """等权合成每日收益率，作为"泛科技板块"基准收益序列。"""
    px_wide = prices_long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    daily_returns = px_wide.pct_change()
    return daily_returns.mean(axis=1).rename("benchmark_return")
