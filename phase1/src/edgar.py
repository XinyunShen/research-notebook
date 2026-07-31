"""
SEC EDGAR 基本面数据获取，严格保留 `filed`（申报日期）字段用于事后做
point-in-time 对齐 —— 这是 design_doc.md 第2.5节"前视偏差"要求的核心工程手段：
任何一天的因子计算，只能用"那一天已经公开申报"的财务数据，不能用之后才
披露/修正的数字。

数据源：data.sec.gov 官方 XBRL API，完全免费，见 design_doc.md 第5.2节。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "edgar_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SEC 要求请求头带可识别的联系方式，见 https://www.sec.gov/os/webmaster-faq#developers
HEADERS = {"User-Agent": "Personal quant research project alyssa.xinyun@gmail.com"}

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_TICKER_MAP_CACHE = CACHE_DIR / "company_tickers.json"

_RATE_LIMIT_SECONDS = 0.15  # SEC 限制 <=10 req/s，留足余量


def _sleep():
    time.sleep(_RATE_LIMIT_SECONDS)


def load_ticker_to_cik() -> dict[str, str]:
    """下载/读取官方 ticker -> CIK 映射表，本地缓存。"""
    if _TICKER_MAP_CACHE.exists():
        raw = json.loads(_TICKER_MAP_CACHE.read_text())
    else:
        r = requests.get(_TICKER_MAP_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        raw = r.json()
        _TICKER_MAP_CACHE.write_text(json.dumps(raw))
        _sleep()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}


def fetch_company_facts(cik: str) -> dict | None:
    """拉取单家公司全部 XBRL facts（us-gaap + dei），本地缓存 JSON。"""
    cache_file = CACHE_DIR / f"{cik}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    _sleep()
    if r.status_code != 200:
        return None
    data = r.json()
    cache_file.write_text(json.dumps(data))
    return data


def _extract_instant(facts: dict, tag: str) -> pd.DataFrame:
    """提取某个时点型（instant）字段，比如股东权益 StockholdersEquity。
    返回列：end（财报截止日）、val、filed（申报日，用于as-of对齐）。
    """
    try:
        entries = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return pd.DataFrame(columns=["end", "val", "filed", "form"])
    df = pd.DataFrame(entries)
    df = df[df["form"].isin(["10-Q", "10-K"])]
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df[["end", "val", "filed", "form"]].sort_values("filed")


def _extract_single_quarter_duration(facts: dict, tag: str) -> pd.DataFrame:
    """提取区间型（duration）字段的"单季度"数值，比如净利润 NetIncomeLoss。

    XBRL 里 10-Q 同时会报"当季"和"年初至今累计"两种区间，必须按区间长度
    （约80-100天=单季度）过滤，否则会把累计值错当成单季度值，算出来的
    TTM（滚动12个月）净利润会严重失真。
    """
    try:
        entries = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "form"])
    df = pd.DataFrame(entries)
    df = df[df["form"].isin(["10-Q", "10-K"])]
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    days = (df["end"] - df["start"]).dt.days
    is_single_quarter = days.between(75, 100)
    is_annual = days.between(350, 380)
    df = df[is_single_quarter | is_annual].copy()
    return df[["start", "end", "val", "filed", "form"]].sort_values("filed")


_REVENUE_TAGS_PRIORITY = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606新准则，2018年后主流
    "Revenues",  # 旧准则，部分公司至今仍用
    "SalesRevenueNet",  # 更早的准则，少数公司
]


def _extract_revenue(facts: dict) -> pd.DataFrame:
    """营收科目在不同公司/不同年代用的XBRL标签不一致（2018年ASC 606新
    准则切换是主因），按优先级依次尝试，**只取第一个有数据的标签**，不
    合并多个标签拼成一条序列——不同准则下"营收"的确认口径可能有细微
    差异，混着拼会引入一条口径不一致的假连续序列，比"某些公司没有营收
    数据、直接跳过"更危险。
    """
    for tag in _REVENUE_TAGS_PRIORITY:
        df = _extract_single_quarter_duration(facts, tag)
        if not df.empty:
            return df
    return pd.DataFrame(columns=["start", "end", "val", "filed", "form"])


MIN_PLAUSIBLE_SHARES_OUTSTANDING = 10_000  # 见下方注释，过滤"壳公司占位股数"这类脏数据


def _extract_shares_outstanding(facts: dict) -> pd.DataFrame:
    """**踩过的坑**：新设立/刚完成分拆重组的公司，第一次XBRL申报时
    EntityCommonStockSharesOutstanding 有时会是一个占位数字（比如
    DXC Technology 2017年从CSC/HPE分拆重组后的第一份申报，股数报的是
    "100"，真实股数要等几个月后正式股权发行完成才更新到2.8亿）。这个
    脏数据会被 merge_asof 一直carry forward到下一次申报之前，导致
    market_cap = 100 * 股价 ≈ 0，book_to_market 算出天文数字（实测
    出现过28742倍这种荒谬值，还会因为cross-sectional winsorize用的是
    同一天的分位数边界，连累同一天其他正常股票的B/M也被错误地钳到这个
    脏数据设的上限）。真实上市公司不可能只有几万股以下的流通股（哪怕是
    伯克希尔A类股每股70万美元、股数最少的极端案例也有60万股以上），
    所以直接用一个远低于任何真实案例的下限（1万股）过滤掉这类占位值，
    当缺失处理，让 merge_asof 往前找上一个真实值（或者在没有更早真实值
    时保持缺失，而不是硬用一个荒谬值算出来的比率）。
    """
    try:
        entries = facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]
    except KeyError:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df = pd.DataFrame(entries)
    if df.empty:
        return df
    df = df[df["val"] >= MIN_PLAUSIBLE_SHARES_OUTSTANDING]
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df[["end", "val", "filed"]].sort_values("filed")


def _ttm_sum(df: pd.DataFrame, value_col: str = "val") -> pd.DataFrame:
    """把单季度数值滚动求和成TTM（最近4个季度），按 end 日期排序，
    ROE的net_income_ttm、下面新增的fcf_ttm/营收TTM都复用这个逻辑，
    避免同一个"单季度->TTM"的滚动求和写三份微妙不一致的实现。"""
    q = df.drop_duplicates(subset=["end"], keep="last").sort_values("end").copy()
    q["ttm"] = q[value_col].rolling(4, min_periods=4).sum()
    return q.dropna(subset=["ttm"])[["filed", "end", "ttm"]]


def build_fundamentals_panel(ticker: str, cik: str) -> pd.DataFrame | None:
    """为单只股票构建"逐次披露事件"的基本面面板，每一行代表一次新的
    财报数据点，附带 filed 日期，供 factors.py 用 merge_asof 做
    point-in-time 对齐。

    输出列：filed, stockholders_equity, net_income_ttm, shares_outstanding,
    fcf_ttm（TTM自由现金流=经营性现金流-资本开支，原始美元金额，除以
    market_cap算fcf_yield留给factors.py，因为那一步需要价格数据），
    margin_trend（TTM营业利润率相对一年前同口径TTM营业利润率的变化，
    2026-07-24新增，回应"哪些真实投资者会看的数据能变成可行因子"这个
    问题——跟ROE是不同的角度：ROE看"现在赚不赚钱"，这个看"利润率在
    变好还是变差"，2026-07-24单独测超大盘ROE发现信号很弱之后，尝试
    换一个跟现有因子重叠度更低的角度，而不是继续在ROE本身上打转）。
    """
    facts = fetch_company_facts(cik)
    if facts is None:
        return None

    se = _extract_instant(facts, "StockholdersEquity")
    ni = _extract_single_quarter_duration(facts, "NetIncomeLoss")
    so = _extract_shares_outstanding(facts)

    if se.empty or ni.empty:
        return None

    # TTM 净利润 = 最近4个"单季度"数值之和（按 end 日期排序后滚动求和）
    ni_q = _ttm_sum(ni).rename(columns={"ttm": "net_income_ttm"})

    se_q = se.drop_duplicates(subset=["end"], keep="last")[["filed", "end", "val"]].rename(
        columns={"val": "stockholders_equity"}
    )

    panel = pd.merge_asof(
        se_q.sort_values("filed"),
        ni_q.sort_values("filed")[["filed", "net_income_ttm"]],
        on="filed",
        direction="backward",
    )

    if not so.empty:
        so_q = so.drop_duplicates(subset=["filed"])[["filed", "val"]].rename(
            columns={"val": "shares_outstanding"}
        )
        panel = pd.merge_asof(
            panel.sort_values("filed"), so_q.sort_values("filed"), on="filed", direction="backward"
        )
    else:
        panel["shares_outstanding"] = float("nan")

    # --- 自由现金流TTM = 经营性现金流TTM - 资本开支TTM ---
    ocf = _extract_single_quarter_duration(facts, "NetCashProvidedByUsedInOperatingActivities")
    capex = _extract_single_quarter_duration(facts, "PaymentsToAcquirePropertyPlantAndEquipment")
    if not ocf.empty and not capex.empty:
        ocf_q = _ttm_sum(ocf).rename(columns={"ttm": "ocf_ttm"})
        capex_q = _ttm_sum(capex).rename(columns={"ttm": "capex_ttm"})
        fcf_df = pd.merge(ocf_q[["end", "filed", "ocf_ttm"]], capex_q[["end", "capex_ttm"]], on="end", how="inner")
        fcf_df["fcf_ttm"] = fcf_df["ocf_ttm"] - fcf_df["capex_ttm"]
        panel = pd.merge_asof(
            panel.sort_values("filed"),
            fcf_df[["filed", "fcf_ttm"]].sort_values("filed"),
            on="filed", direction="backward",
        )
    else:
        panel["fcf_ttm"] = float("nan")

    # --- 营业利润率的同比变化（margin_trend）---
    oi = _extract_single_quarter_duration(facts, "OperatingIncomeLoss")
    rev = _extract_revenue(facts)
    if not oi.empty and not rev.empty:
        oi_q = _ttm_sum(oi).rename(columns={"ttm": "operating_income_ttm"})
        rev_q = _ttm_sum(rev).rename(columns={"ttm": "revenue_ttm"})
        margin_df = pd.merge(
            oi_q[["end", "filed", "operating_income_ttm"]], rev_q[["end", "revenue_ttm"]], on="end", how="inner"
        ).sort_values("end")
        margin_df["operating_margin_ttm"] = float("nan")
        positive_rev = margin_df["revenue_ttm"] > 0
        margin_df.loc[positive_rev, "operating_margin_ttm"] = (
            margin_df.loc[positive_rev, "operating_income_ttm"] / margin_df.loc[positive_rev, "revenue_ttm"]
        )
        # 只有真正相隔约1年（4个季度）的两个点才算YoY变化，数据有缺口、
        # 季度不连续时不能悄悄拿"4行前"当"一年前"，会算出没有意义的数字
        end_gap_days = (margin_df["end"] - margin_df["end"].shift(4)).dt.days
        margin_trend = margin_df["operating_margin_ttm"] - margin_df["operating_margin_ttm"].shift(4)
        margin_df["margin_trend"] = margin_trend.where(end_gap_days.between(330, 400))
        panel = pd.merge_asof(
            panel.sort_values("filed"),
            margin_df[["filed", "margin_trend"]].sort_values("filed"),
            on="filed", direction="backward",
        )

        # --- 营收增速的加速度（revenue_growth_accel，2026-07-29新增）---
        # 回应用户提出的"看增量的增量"理论（2008次贷危机：房价还在涨，但
        # 涨幅逐季度收窄；今天新闻里SK海力士营收+257%/利润+557%但仍不及
        # 预期导致股价大跌，也是同一个机制）——市场price的不是"增长率"
        # 这个水平值本身，是"增长率相对上一期是在加速还是减速"，哪怕两个
        # 数字（增长率本身）都是正的、绝对值再高，减速本身就可能被解读成
        # 坏消息。这跟margin_trend（利润率的同比变化）是不同维度：margin_trend
        # 看"赚钱效率在变好还是变差"，这个看"增长本身在加速还是减速"，
        # 两者都是"变化率的变化率"这个思路的具体应用，但作用在不同分子上。
        rev_growth = rev_q[["end", "filed", "revenue_ttm"]].sort_values("end").reset_index(drop=True)
        end_gap_4q = (rev_growth["end"] - rev_growth["end"].shift(4)).dt.days
        prior_year_rev = rev_growth["revenue_ttm"].shift(4)
        yoy_growth = pd.Series(float("nan"), index=rev_growth.index)
        valid_yoy = end_gap_4q.between(330, 400) & (prior_year_rev > 0)
        yoy_growth[valid_yoy] = rev_growth.loc[valid_yoy, "revenue_ttm"] / prior_year_rev[valid_yoy] - 1
        # 加速度：这一期的同比增速 减去 上一季度的同比增速——只有真正相隔
        # 一个季度（~90天）的两个点才算，避免数据缺口把不相邻的两期错当
        # "上一季度"比较（同一类"不能用行号当日历"的坑，margin_trend那段
        # 已经踩过一次，这里延续同样的防御写法）。
        end_gap_1q = (rev_growth["end"] - rev_growth["end"].shift(1)).dt.days
        growth_accel = yoy_growth - yoy_growth.shift(1)
        rev_growth["revenue_growth_accel"] = growth_accel.where(end_gap_1q.between(75, 100))
        panel = pd.merge_asof(
            panel.sort_values("filed"),
            rev_growth[["filed", "revenue_growth_accel"]].sort_values("filed"),
            on="filed", direction="backward",
        )
    else:
        panel["margin_trend"] = float("nan")
        panel["revenue_growth_accel"] = float("nan")

    # 股东权益为负/为零时 ROE 没有经济意义（常见于大量股票回购的科技公司），
    # 学术上（Fama-French）标准做法是直接剔除负账面权益公司，而不是硬算出
    # 一个畸形的极端比率——否则会产生 inf/-inf，污染整个因子分布。
    # 用 np.nan（而不是 pd.NA）填充，保证列是纯 float64，兼容 numpy/alphalens
    # 的 isfinite 等运算；pd.NA 会把列变成 object dtype 导致下游报错。
    positive_equity = panel["stockholders_equity"] > 0
    panel["roe_ttm"] = float("nan")
    panel.loc[positive_equity, "roe_ttm"] = (
        panel.loc[positive_equity, "net_income_ttm"] / panel.loc[positive_equity, "stockholders_equity"]
    )
    panel["roe_ttm"] = panel["roe_ttm"].astype("float64")
    panel["fcf_ttm"] = panel["fcf_ttm"].astype("float64")
    panel["margin_trend"] = panel["margin_trend"].astype("float64")
    return panel.sort_values("filed").reset_index(drop=True)
