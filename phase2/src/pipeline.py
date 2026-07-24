"""
Phase 2 主入口：构造"主题热度指数" -> 算下游股票的"主题贝塔" -> 用
Alphalens 检验主题贝塔本身有没有预测力。对应 design_doc.md 3.2节 +
8节"验证机制B"。

本版本加入了"剔除普通行业共振"的修复（design_doc.md 12节问题8）：
用 Phase 1 科技基础池（排除AI产业链池本身）构造一个泛科技板块基准，
把L1核心股和下游候选股的收益都换成"超额收益"（原始收益-基准收益）
再去构造主题热度指数、算主题贝塔。**同时保留原始（未剔除）版本**，
两者都跑、都存、都打印，方便直接对比"剔除行业共振前后差多少"，
不是偷偷把旧结果换掉。

这版还接入了 Finnhub 新闻热度信号（design_doc.md 12节问题9）：
设置了 FINNHUB_API_KEY 环境变量就会自动拉取L1核心股（NVDA/AMD/AVGO/
TSM）的新闻提及量趋势，作为主题热度指数的第三路信号；没设置就自动
跳过，退回只用"L1动量 + ETF成交量异动"两路（见 news_finnhub.py）。

这版还修复了 circularity（design_doc.md 12节问题11）：用 Google Trends
搜索热度（`src/google_trends.py`，关键词见 KEYWORDS，不含任何被测公司名）
替换掉有自我印证问题的 SOXX/SMH 成交量分量。**同样保留旧版本一起跑**，
新增第三条 trends 对比线，raw/excess 两条不受影响。Google Trends走的是
非官方 pytrends 库，限流经常发生，拉取失败时会自动跳过这条对比线、
退回只跑 raw/excess 两条（见 google_trends.py 的重试和已知限制说明）。

这版还加了**AI叙事诞生前/后分段检验**（design_doc.md 12节问题17，回应
"AI这种最近2-3年才火的东西，怎么能用10年前的数据回测"这个质疑）：整个
2016-2026回测窗口里，"AI叙事"作为能推动股价共振的力量，公认起点大约是
2022年11月ChatGPT发布往后（此前这些下游候选股，比如Constellation
Energy，跟AI投资叙事毫无关系，2023-2024年跟云厂商签购电协议之后才被
市场当成"AI电力股"）。用SPLIT_DATE=2023-01-01把回测拆成"叙事前"
（2016-2022）和"叙事后"（2023-2026）两段分别算IC——如果机制B真的是
AI叙事驱动的，信号应该集中在"叙事后"这段，"叙事前"应该接近噪音；如果
两段差不多，反而说明当前测到的东西可能和"AI叙事"没什么关系，是别的
更早就存在的普通行业效应。

用法：
    cd phase2 && source ../phase1/.venv/bin/activate   # 复用phase1的venv
    export FINNHUB_API_KEY=你的key   # 可选，不设置就跳过新闻信号
    python -m src.pipeline
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as theme_config  # noqa: E402
from src import benchmark, google_trends, news_finnhub, theme_index, thematic_beta, validate  # noqa: E402


def _load_module(name: str, path: Path):
    """用文件路径直接加载 phase1 的 data_fetch.py，避免和 phase2 自己的
    `src` 包在 sys.path 上撞名（两边都有名叫 src 的包，直接 import src
    会拿到不确定是哪一边的模块）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE1_SRC = ROOT.parent / "phase1" / "src"
data_fetch = _load_module("phase1_data_fetch", PHASE1_SRC / "data_fetch.py")

START_DATE = "2016-01-01"
SPLIT_DATE = "2023-01-01"  # ChatGPT 2022-11发布，AI叙事公认起点，见上方模块docstring
PHASE2_CACHE = ROOT / "data" / "prices.parquet"
OUTPUT_DIR = ROOT / "output"


def main():
    print(f"[pipeline] 拉取价格数据: L1={theme_config.L1_TICKERS}, "
          f"下游候选={theme_config.DOWNSTREAM_TICKERS}, ETF={theme_config.ETF_TICKERS}")
    PHASE2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices = data_fetch.fetch_prices(
        theme_config.ALL_TICKERS, start=START_DATE, end=str(date.today()), cache_path=PHASE2_CACHE
    )
    print(f"[pipeline] 行情数据行数: {len(prices)}, 覆盖股票数: {prices['ticker'].nunique()}")

    px_wide = validate.prices_wide(prices)
    vol_wide = validate.volumes_wide(prices)
    raw_returns_wide = px_wide.pct_change()

    print("[pipeline] 构造泛科技板块基准（剔除AI产业链池本身，避免自我印证）...")
    benchmark_universe = benchmark.load_benchmark_universe(exclude_tickers=set(theme_config.ALL_TICKERS))
    print(f"[pipeline] 基准股票池 {len(benchmark_universe)} 只（复用Phase 1缓存，不需要重新联网）")
    benchmark_prices = benchmark.fetch_benchmark_prices(benchmark_universe, start=START_DATE, end=str(date.today()))
    benchmark_ret = benchmark.compute_benchmark_returns(benchmark_prices)
    excess_returns_wide = raw_returns_wide.subtract(benchmark_ret, axis=0)

    news_by_ticker = None
    if news_finnhub.api_key_available():
        print(f"[pipeline] FINNHUB_API_KEY 已设置，拉取L1核心股新闻热度: {theme_config.L1_TICKERS} "
              "（只拉L1，不拉下游候选股，避免重蹈circularity的覆辙，见 news_finnhub.py）...")
        news_by_ticker = news_finnhub.fetch_news_for_tickers(
            theme_config.L1_TICKERS, start=START_DATE, end=str(date.today())
        )
        print(f"[pipeline] 取到新闻数据的股票: {sorted(news_by_ticker.keys())}")
    else:
        print("[pipeline] 未设置 FINNHUB_API_KEY，跳过新闻热度信号（见 phase2/src/news_finnhub.py 如何启用）")

    print(f"[pipeline] 拉取Google Trends搜索热度: {google_trends.KEYWORDS} "
          "（用于替换有circularity问题的ETF成交量信号，见 google_trends.py）...")
    combined_trends = google_trends.build_combined_trends_series(
        google_trends.KEYWORDS, start=START_DATE, end=str(date.today())
    )
    if combined_trends is None:
        print("[pipeline] Google Trends全部拉取失败（pytrends限流/网络问题），跳过circularity修复版本的对比，"
              "只跑 raw/excess 两条")
    else:
        print(f"[pipeline] Google Trends拿到 {combined_trends.notna().sum()} 个周频数据点，"
              f"覆盖 {combined_trends.dropna().index.min().date()} ~ {combined_trends.dropna().index.max().date()}")

    results = {}
    news_heat_insufficient = False
    passes = [
        ("raw", "raw（未剔除行业共振，旧版本，仅作对比）", None, None, None),
        ("excess", "excess（已剔除行业共振，ETF成交量版）", excess_returns_wide, excess_returns_wide, None),
    ]
    if combined_trends is not None:
        passes.append((
            "trends",
            "trends（已剔除行业共振 + Google Trends替换ETF，修复circularity）",
            excess_returns_wide, excess_returns_wide, combined_trends,
        ))

    for tag, label, excess_arg, returns_arg, trends_arg in passes:
        print(f"\n========== 构造主题热度指数：{label} ==========")
        heat_df = theme_index.build_theme_heat_index(
            prices_wide=px_wide,
            volumes_wide=vol_wide,
            l1_tickers=theme_config.L1_TICKERS,
            etf_tickers=theme_config.ETF_TICKERS,
            news_by_ticker=news_by_ticker,
            excess_returns_wide=excess_arg,
            google_trends_series=trends_arg,
        )
        heat_df.to_csv(OUTPUT_DIR / f"theme_heat_index_{tag}.csv")
        print(heat_df.describe())
        if "news_heat_z" in heat_df.columns:
            valid_news_days = int(heat_df["news_heat_z"].notna().sum())
            if valid_news_days < 63:
                news_heat_insufficient = True
                print(f"[pipeline] WARNING: news_heat_z 只有 {valid_news_days} 天非空（<63天滚动zscore"
                      "所需的最少样本），说明Finnhub免费层给的历史新闻深度不够，这一路信号在本次"
                      "回测里实际上没有生效（被skipna平均悄悄忽略了），不是“新闻信号没用”的结论，"
                      "只是免费数据源覆盖不到2016-2026这段历史，见 theme_index.py news_heat_component 注释。")

        print(f"[pipeline] 计算下游股票的主题贝塔（{label}，滚动126日窗口）...")
        beta_long = thematic_beta.compute_thematic_beta(
            prices_wide=px_wide,
            theme_heat_index=heat_df["theme_heat_index"],
            tickers=theme_config.DOWNSTREAM_TICKERS,
            returns_wide=returns_arg,
        )
        beta_panel = beta_long.set_index(["date", "ticker"])["thematic_beta"]
        beta_panel.to_frame().to_parquet(OUTPUT_DIR / f"thematic_beta_{tag}.parquet")
        print(beta_panel.describe())

        print(f"[pipeline] 用Alphalens检验主题贝塔（{label}）是否对未来收益有预测力 ...")
        results[tag] = validate.run_ic_and_quantile_analysis(beta_panel, px_wide, f"thematic_beta_{tag}")

        dates = beta_panel.index.get_level_values("date")
        pre_panel = beta_panel[dates < SPLIT_DATE]
        post_panel = beta_panel[dates >= SPLIT_DATE]
        print(f"[pipeline] AI叙事诞生前/后分段检验（{label}，分界{SPLIT_DATE}）...")
        results[f"{tag}_pre"] = validate.run_ic_and_quantile_analysis(
            pre_panel, px_wide, f"thematic_beta_{tag}_pre2023"
        )
        results[f"{tag}_post"] = validate.run_ic_and_quantile_analysis(
            post_panel, px_wide, f"thematic_beta_{tag}_post2023"
        )

    print("\n[pipeline] ===== 对比：raw / excess（ETF成交量）/ trends（Google Trends修circularity）=====")
    for tag, _, _, _, _ in passes:
        ic_summary = results.get(tag, {}).get("ic_summary")
        if ic_summary is not None:
            print(f"-- {tag} --")
            print(ic_summary.round(4))

    print(f"\n[pipeline] ===== AI叙事前（2016-2022） vs 叙事后（2023-2026，分界{SPLIT_DATE}）对比 =====")
    for tag, _, _, _, _ in passes:
        pre_ic = results.get(f"{tag}_pre", {}).get("ic_summary")
        post_ic = results.get(f"{tag}_post", {}).get("ic_summary")
        print(f"-- {tag} 叙事前 --")
        print(pre_ic.round(4) if pre_ic is not None else "(无数据)")
        print(f"-- {tag} 叙事后 --")
        print(post_ic.round(4) if post_ic is not None else "(无数据)")

    print("\n[pipeline] 完成。输出见 phase2/output/（*_raw / *_excess / *_trends 三个后缀区分）")
    print(f"[pipeline] 重要提醒：下游候选股只有{len(theme_config.DOWNSTREAM_TICKERS)}只，")
    print("           这是 design_doc.md 第8节'第零步'里提到的产业链传导策略天然的样本量")
    print("           短板，这里的结果只能是方向性参考，不是统计上可信的结论。")
    if not news_by_ticker:
        news_status = "未设置FINNHUB_API_KEY，未接入"
    elif news_heat_insufficient:
        news_status = "已取到数据但历史深度不够（Finnhub免费层限制，见上方WARNING），本次回测里未实际生效"
    else:
        news_status = "已接入并生效"
    print(f"           新闻热度信号：{news_status}。")
    trends_status = "已接入（trends对比线）" if combined_trends is not None else "拉取失败，本次跳过，只有raw/excess两条"
    print(f"           Google Trends修circularity：{trends_status}。")
    return results


if __name__ == "__main__":
    main()
