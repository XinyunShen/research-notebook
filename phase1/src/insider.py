"""
Form 4 内部人交易"净买入"资金流因子，对应 design_doc.md 5.3节 + 12节
问题12把"资金流因子"补进四因子组合（估值+质量+动量+资金流，见4节推荐
组合）的决定。

**为什么用 Form 4 不用 13F 机构持仓聚合数据**：13F按CUSIP披露持仓，SEC
官方数据集里没有免费的CUSIP-ticker对照表（CUSIP本身是S&P/CUSIP Global
Services的授权数据，不能自由分发），要把全市场机构持仓数据聚合到
ticker层面必须先解决这个映射问题，工程量和数据授权问题都超出这轮精修
的范围。Form 4 是design_doc.md 5.3节本身也推荐的数据源（"披露相对
及时"），关键优势是**按发行人（公司）CIK可查**，直接复用 edgar.py 已有
的 ticker->CIK 基础设施，不需要额外的映射层。

**因子构造**：只统计非衍生品交易(nonDerivativeTransaction)里
transactionCode 为 P（公开市场买入）或 S（公开市场卖出）的交易——这是
学术上标准的"内部人知情交易"定义（Lakonishok & Lee 2001 等经典内部人
交易文献），排除 G（赠与）、A（股权授予）、M（期权行权）、F（税务代扣）
等非自主决策性质的交易，这些不代表"内部人对公司前景的判断"。用**过去
63个交易日**（约1季度，和项目里其他因子的滚动窗口尺度一致）的净买入
股数（P的shares之和 - S的shares之和）除以最新流通股数，得到"内部人
净买入占流通股比例"。

**point-in-time对齐**：每笔交易用的是 filed（申报/公开披露日），不是
transactionDate（实际交易发生日）——按Section 16规则，交易发生后最多
有2个交易日的披露延迟，市场只有在filed之后才知道这笔交易，用
transactionDate会有前视偏差。这一点和factors.py里季度财报用filed对齐
的原则完全一致。

**工程规模问题**：SEC 没有提供"某只股票所有Form 4交易"的聚合API，
一只股票常见有几百条filing，每条都要单独拉一次原始XML。跑全量300+只
股票总请求量在数万到十万级别，串行要几小时，这里用多线程+全局限速
（控制在SEC公开的"不超过10次/秒"公平使用限制以内）压缩到大概几十分钟
到一个多小时，每只股票的解析结果落盘缓存（parquet），中断重跑不会
白拉，见 `data/insider_cache/`。
"""
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
import requests

from . import edgar

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "insider_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_REQ_PER_SECOND = 8.0  # SEC公平使用政策上限约10次/秒，留余量
WINDOW = 63  # 约1个季度的交易日数，和项目里其他滚动因子窗口尺度保持一致


class _RateLimiter:
    """全局限速器：不管多少个线程调用 wait()，聚合请求速率都不超过
    max_per_second——这是跨线程共享同一个限速器实例来实现的，不是
    每个线程各自限速（那样聚合速率会是 max_per_second * 线程数，
    会打爆SEC限流）。"""

    def __init__(self, max_per_second: float):
        self._interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = self._next_time - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_time = max(now, self._next_time) + self._interval


_LIMITER = _RateLimiter(MAX_REQ_PER_SECOND)

_REQUEST_TIMEOUT_SECONDS = 10
_GET_MAX_RETRIES = 3
_GET_RETRY_BACKOFF_SECONDS = 3
# 单个请求最坏情况耗时 ≈ 3*10 + 2*3 = 36秒（之前是 4*20+3*5=95秒）。
# 实测跑全量305只时，ALGM一只卡了92分钟——网络一抖，8个并发线程一起撞上
# 最坏情况重试链，代价会成倍放大，这里把超时和重试都调短，降低单次意外
# 的最大代价，不是消除重试（重试本身是对的，只是原来的参数太宽松）。

PER_TICKER_DEADLINE_SECONDS = 300  # 见 fetch_ticker_insider_transactions 的硬性兜底


def _get(url: str) -> requests.Response | None:
    """**踩过的坑**：全量305只股票跑几万次请求，中途SSL连接被重置
    （`ConnectionResetError`）这种瞬时网络错误几乎必然会遇到至少一次——
    第一版没有重试，跑了15只股票几十分钟后被一次连接重置直接干死整个
    进程，之前的进度全靠per-ticker缓存才没白费，但重跑还是要从头过一遍
    已缓存的ticker（虽然会命中缓存很快跳过）。现在对连接类异常做指数
    退避重试，重试耗尽后返回None，由调用方决定怎么处理缺失（通常是
    跳过这一条filing不是整个进程崩溃）。"""
    for attempt in range(1, _GET_MAX_RETRIES + 1):
        _LIMITER.wait()
        try:
            return requests.get(url, headers=edgar.HEADERS, timeout=_REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as e:
            is_last = attempt == _GET_MAX_RETRIES
            print(f"[insider] WARNING: 请求 {url} 第{attempt}次尝试失败: {e}"
                  f"{'，放弃' if is_last else f'，{_GET_RETRY_BACKOFF_SECONDS}秒后重试'}")
            if is_last:
                return None
            time.sleep(_GET_RETRY_BACKOFF_SECONDS)
    return None


def _list_form4_filings(cik: str, start: str) -> list[dict]:
    """返回该CIK在start日期之后的所有Form 4 filing索引
    [{accession, primary_doc, filed}, ...]，跨 recent + 分页归档文件。"""
    r = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if r is None or r.status_code != 200:
        return []
    data = r.json()
    filings: list[dict] = []

    def _collect(block: dict):
        for form, accession, primary_doc, filed in zip(
            block["form"], block["accessionNumber"], block["primaryDocument"], block["filingDate"]
        ):
            if form == "4" and filed >= start:
                filings.append({"accession": accession, "primary_doc": primary_doc, "filed": filed})

    _collect(data["filings"]["recent"])

    for older in data["filings"].get("files", []):
        if older["filingTo"] < start:
            continue  # 这一页归档文件全在start之前，跳过整页省一次请求
        r2 = _get(f"https://data.sec.gov/submissions/{older['name']}")
        if r2 is not None and r2.status_code == 200:
            _collect(r2.json())

    return filings


def _parse_aff10b5one(root: ET.Element) -> bool | None:
    """解析文档级的 aff10b5One 标志（2026-07-31新增）——Rule 10b5-1计划性
    交易的勾选框，**是整份filing级别的字段，不是逐笔交易级别的**（实测
    确认：这个元素直接挂在 <ownershipDocument> 根节点下，不在每条
    nonDerivativeTransaction里面），所以一份filing里的所有P/S交易共享
    同一个标志。**取值格式不统一**：实测同期不同公司/不同schema版本里
    见过"0"/"1"（INTC/NVDA/MSFT）和"true"/"false"（AAPL）两种写法，
    必须都处理，不能只认数字。缺失（字段不存在）时返回None，不当0处理——
    "没有这个字段"和"这个字段明确填了0"是不同的信息状态，不能悄悄合并。
    """
    el = root.find("./aff10b5One")
    if el is None or el.text is None or not el.text.strip():
        return None
    val = el.text.strip().lower()
    if val in ("1", "true"):
        return True
    if val in ("0", "false"):
        return False
    return None


def _fetch_and_parse_transactions(cik_nolead: str, accession: str, primary_doc: str, filed: str) -> list[dict]:
    """拉单条filing的原始XML（不是XSLT渲染的HTML，是同一目录下的raw XML
    文件，把 primaryDocument 路径里的 xsl.../ 前缀去掉、取basename即是），
    解析出P/S两种代码的非衍生品交易，附带这份filing的aff10b5One标志
    （2026-07-31新增，回应design_doc.md早就记录的TODO——区分"计划性
    卖出"和"自主决策交易"，前者是几个月前设定好的公式化执行，不携带
    "内部人现在怎么看公司"的信息，学术文献（Jagolinzer 2009等）证实
    几乎没有预测力，混在一起会稀释真正有信息含量的自主交易信号）。"""
    accession_nodash = accession.replace("-", "")
    raw_filename = primary_doc.split("/")[-1]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{raw_filename}"
    r = _get(url)
    if r is None or r.status_code != 200:
        return []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []

    is_10b5_1 = _parse_aff10b5one(root)

    rows = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code_el = txn.find("./transactionCoding/transactionCode")
        code = code_el.text.strip() if code_el is not None and code_el.text else None
        if code not in ("P", "S"):
            continue
        shares_el = txn.find("./transactionAmounts/transactionShares/value")
        if shares_el is None or shares_el.text is None:
            continue
        try:
            shares = float(shares_el.text)
        except ValueError:
            continue
        rows.append({"filed": filed, "code": code, "shares": shares, "is_10b5_1_plan": is_10b5_1})
    return rows


def fetch_ticker_insider_transactions(
    ticker: str, cik: str, start: str, max_workers: int = 8, force_refresh: bool = False
) -> pd.DataFrame:
    """单只股票的P/S内部人交易明细，本地缓存。返回列：filed, code, shares,
    is_10b5_1_plan（2026-07-31新增，见_parse_aff10b5one/_fetch_and_parse_transactions
    注释；True=计划性交易，False=自主交易，None=filing没有这个字段）

    **硬性兜底**（PER_TICKER_DEADLINE_SECONDS）：实测跑全量305只时，网络
    一抖，ALGM 一只卡了92分钟——重试参数已经调紧（见_get注释），但为了
    彻底杜绝"某只股票把整个流水线拖住几十分钟"，这里额外加一层保险：
    单只股票超过5分钟就不再等剩下的filing，用已经拿到的部分数据落盘，
    差的那部分下次force_refresh重跑才会补全。**这意味着缓存里个别股票
    的数据可能不完整**（比如网络卡顿时抓的那批），因子层面这只是让该
    股票某些窗口的样本更稀疏，不是构造性错误——比因为一只股票的网络问题
    拖累全部305只跑不完要划算。
    """
    cache_path = CACHE_DIR / f"{ticker}.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    cik_nolead = str(int(cik))
    filings = _list_form4_filings(cik, start)

    all_rows: list[dict] = []
    if filings:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {
                executor.submit(
                    _fetch_and_parse_transactions, cik_nolead, f["accession"], f["primary_doc"], f["filed"]
                ): f
                for f in filings
            }
            done, not_done = wait(futures, timeout=PER_TICKER_DEADLINE_SECONDS)
            for fut in done:
                all_rows.extend(fut.result())
            if not_done:
                print(f"[insider] WARNING: {ticker} 超过{PER_TICKER_DEADLINE_SECONDS}秒硬性兜底，"
                      f"{len(not_done)}/{len(filings)}条filing没跑完，先用已拿到的{len(done)}条落盘，"
                      "不再等剩下的（常见原因是网络抖动，不是这只股票本身filing异常多）")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    result = pd.DataFrame(all_rows, columns=["filed", "code", "shares", "is_10b5_1_plan"])
    if not result.empty:
        result["filed"] = pd.to_datetime(result["filed"])
    result.to_parquet(cache_path, index=False)
    return result


def compute_insider_flow_factor(
    prices: pd.DataFrame, tickers: list[str], start: str, window: int = WINDOW,
    exclude_10b5_1: bool = False,
) -> pd.DataFrame:
    """返回长表：columns=[date, ticker, insider_net_buy_pct]

    insider_net_buy_pct = 过去window个交易日内，Form 4公开市场买入(P)减
    卖出(S)的净股数，除以最新流通股数。shares_outstanding复用
    edgar.build_fundamentals_panel（已经在factors.py的流程里缓存过，
    这里不会产生新的网络请求），用merge_asof做point-in-time对齐。

    exclude_10b5_1: **2026-07-31新增**，True时剔除标记为Rule 10b5-1
    计划性交易（`is_10b5_1_plan is True`）的记录，只用自主决策交易算净
    买入——见design_doc.md早就记录的TODO + 学术文献（Jagolinzer 2009等）：
    计划性交易是几个月前设定好的公式化执行，不携带"内部人现在怎么看
    公司"的信息，混在一起会稀释真正有信息含量的自主交易信号。缺失
    该字段的旧记录（is_10b5_1_plan为None，通常是刷新前缓存的数据）
    保守地当作"非计划性"处理，不剔除——不能因为字段缺失就武断地丢弃
    数据，缺失和"确认是自主交易"在这里做同样处理是刻意的保守选择。
    """
    ticker_to_cik = edgar.load_ticker_to_cik()
    all_rows = []

    for i, t in enumerate(tickers):
        cik = ticker_to_cik.get(t.upper())
        if cik is None:
            print(f"[insider] WARNING: {t} not found in SEC ticker map, skipping")
            continue

        print(f"[insider] ({i + 1}/{len(tickers)}) {t} ...")
        txns = fetch_ticker_insider_transactions(t, cik, start=start)

        px = prices[prices["ticker"] == t][["date"]].sort_values("date").copy()
        if px.empty:
            continue

        if txns.empty:
            px["insider_net_buy_pct"] = float("nan")
            all_rows.append(px.assign(ticker=t)[["date", "ticker", "insider_net_buy_pct"]])
            continue

        txns = txns.copy()
        if exclude_10b5_1 and "is_10b5_1_plan" in txns.columns:
            txns = txns[txns["is_10b5_1_plan"] != True]  # noqa: E712（三态字段，不能写成~txns[...]）
        txns["signed_shares"] = txns["shares"] * txns["code"].map({"P": 1.0, "S": -1.0})
        daily_net = txns.groupby("filed")["signed_shares"].sum()
        daily_net = daily_net.reindex(pd.DatetimeIndex(px["date"]), fill_value=0.0)
        rolling_net = daily_net.rolling(window, min_periods=1).sum()

        panel = edgar.build_fundamentals_panel(t, cik)
        if panel is None or panel.empty:
            so_series = pd.Series(float("nan"), index=pd.DatetimeIndex(px["date"]))
        else:
            so_df = panel[["filed", "shares_outstanding"]].rename(columns={"filed": "date"}).sort_values("date")
            merged_so = pd.merge_asof(px[["date"]].sort_values("date"), so_df, on="date", direction="backward")
            so_series = merged_so.set_index("date")["shares_outstanding"]

        net_buy_pct = rolling_net.values / so_series.reindex(rolling_net.index).values
        result = px.assign(ticker=t, insider_net_buy_pct=net_buy_pct)
        all_rows.append(result[["date", "ticker", "insider_net_buy_pct"]])

    if not all_rows:
        return pd.DataFrame(columns=["date", "ticker", "insider_net_buy_pct"])
    out = pd.concat(all_rows, ignore_index=True)
    out["insider_net_buy_pct"] = pd.to_numeric(out["insider_net_buy_pct"], errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )
    return out
