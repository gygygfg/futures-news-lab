"""影响因素词云: 从新闻中提取关键词并量化。

对每个品种, 统计其相关新闻的关键词:
- value  出现频次(决定字号)
- sent   含该词新闻的平均情绪(决定颜色: 红=偏多/蓝=偏空)
- impact 含该词新闻当日的平均|未来2周收益|%(关联影响力, 悬停显示)

输出 web/news_wordcloud.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from news_db import PROJ, connect

RAW = PROJ / "data" / "raw" / "daily"
OUT = PROJ / "web" / "news_wordcloud.json"
TOP_N = 60

SYMBOLS = ["RB0", "CU0", "AU0", "SC0", "IF0", "T0"]
NAMES = {"RB0": "螺纹钢", "CU0": "沪铜", "AU0": "沪金", "SC0": "原油", "IF0": "沪深300", "T0": "10年国债"}

ALLOWED_EN = {"OPEC", "MLF", "LPR", "CPI", "PPI", "GDP", "PMI", "WTI", "EIA", "FOMC", "ETF", "SHFE", "CFTC"}

STOPWORDS = set((
    "的 了 和 与 及 或 在 是 为 对 等 将 已 有 也 这 那 一个 我们 你们 他们 以及 表示 称 据 月 日 年 时 分 秒 "
    "记者 报道 消息 显示 认为 指出 方面 相关 进行 目前 今日 昨日 本周 上周 今年 去年 中国 美国 全球 市场 "
    "同比 环比 数据 分析 预期 影响 情况 问题 工作 发展 经济 政策 会议 发布 举行 国家 部门 公司 企业 集团 "
    "百分点 亿美元 亿元 万千瓦 昨日 今天 明天 上午 下午 晚间 早间 近日 近期 以来 左右 分别 超过 达到 同比增 "
    "以上 以下 其中 由于 因为 所以 但是 不过 而且 并且 同时 此外 另外 目前 当前 未来 预计 可能 或将 将会 "
    "上涨 下跌 走强 走弱 收涨 收跌 涨幅 跌幅 上涨 下降 增长 减少 增加 扩大 收窄 上升 回落 上涨 下滑 "
    # 经济日历元数据/事件名样板词
    "地区 公布 重要性 前值 预期 变动 每日 上期 截至 日当周 当周 竞拍 个月期 中标 总金额 配置 倍数 投标 增减 "
    "初值 终值 修正值 "
    # 通用泛词
    "推动 建设 持续 我国 推进 进一步 实施 加强 促进 提升 完善 优化 加快 全国 各地 有关 相关 扎实 有效 "
    "不断 积极 稳定 重要 明显 总体 基本 切实 认真 深入 全面 着力 大力 坚持 统筹 "
    "实现 领域 重点 支持 高质量 更新 星期 百分比 快讯 文化 世界 产业 项目 上市 举行 发展 方面 "
    "体系 技术 创新 服务 生产 历史 传统 本次 日讯 习近平 主席 总书记").split())


def tokenize(text: str) -> list[str]:
    import jieba
    out = []
    for t in jieba.lcut(text):
        t = t.strip()
        if not t or t in STOPWORDS:
            continue
        if t in ALLOWED_EN:
            out.append(t)
            continue
        if len(t) < 2:
            continue
        if all(not ('\u4e00' <= ch <= '\u9fff') for ch in t):
            continue  # 仅保留含中文的多字词
        out.append(t)
    return out


def ret_fwd_map(sym: str) -> dict[str, float]:
    """当日 -> 未来2周累计收益的绝对值%(与 news_impact 影响时效一致)。"""
    from news_impact import HORIZON_TDAYS

    df = pd.read_csv(RAW / f"{sym}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["ret_fwd"] = ((df["close"].shift(-HORIZON_TDAYS) / df["close"] - 1.0) * 100).abs()
    df["d"] = df["date"].dt.strftime("%Y-%m-%d")
    return dict(zip(df["d"], df["ret_fwd"]))


def main() -> None:
    from news_impact import SOURCE_HOME, article_url

    conn = connect()
    df = pd.read_sql_query(
        "SELECT substr(dt,1,10) AS date, title, content, source, symbols, sentiment, url FROM news "
        "WHERE sentiment IS NOT NULL AND symbols != '' AND dt >= '2000-01-01'",
        conn,
    )
    rn = {s: ret_fwd_map(s) for s in SYMBOLS}

    freq = {s: defaultdict(int) for s in SYMBOLS}
    ssum = {s: defaultdict(float) for s in SYMBOLS}
    isum = {s: defaultdict(float) for s in SYMBOLS}
    icnt = {s: defaultdict(int) for s in SYMBOLS}
    cache: list[tuple] = []  # (date,title,url,source,sent,syms,kws)

    for r in df.itertuples():
        # 经济日历的 content 是程序生成的元数据, 只用标题避免样板词污染
        text = r.title if r.source == "经济日历" else f"{r.title} {r.content}"
        kws = set(tokenize(text))
        if not kws:
            continue
        syms = [s for s in str(r.symbols).split(",") if s in freq]
        cache.append((r.date, r.title, r.url or "", r.source, r.sentiment, syms, kws))
        for sym in syms:
            rv = rn[sym].get(r.date)
            for kw in kws:
                freq[sym][kw] += 1
                ssum[sym][kw] += float(r.sentiment)
                if rv is not None and not pd.isna(rv):
                    isum[sym][kw] += float(rv)
                    icnt[sym][kw] += 1

    # 每品种 top 关键词集合
    top = {s: {k for k, n in sorted(freq[s].items(), key=lambda kv: -kv[1])[:TOP_N] if freq[s][k] >= 3}
           for s in SYMBOLS}

    # 每个关键词收集相关新闻(用于点击弹窗的 top10 链接)
    knews = {s: defaultdict(list) for s in SYMBOLS}
    for (date, title, url, source, sent, syms, kws) in cache:
        for sym in syms:
            for kw in (kws & top[sym]):
                knews[sym][kw].append((date, title, url, source, sent))

    def pack(sym: str) -> list[dict]:
        items = []
        for kw, n in freq[sym].items():
            if kw not in top[sym]:
                continue
            news = sorted(knews[sym][kw], key=lambda x: x[0], reverse=True)[:10]
            items.append({
                "name": kw,
                "value": n,
                "sent": round(ssum[sym][kw] / n, 4),
                "impact": round(isum[sym][kw] / icnt[sym][kw], 3) if icnt[sym][kw] else None,
                "news": [
                    {"date": d, "title": (t or "")[:60], "url": article_url(sc, d, u),
                     "source": sc, "source_url": SOURCE_HOME.get(sc, ""),
                     "sent": None if s is None or pd.isna(s) else round(float(s), 3)}
                    for (d, t, u, sc, s) in news
                ],
            })
        items.sort(key=lambda x: x["value"], reverse=True)
        return items

    out = {"symbols": {s: {"name": NAMES[s], "words": pack(s)} for s in SYMBOLS}}
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[wordcloud] -> {OUT} ({OUT.stat().st_size/1024:.0f}KB)")
    for s in SYMBOLS:
        w = out["symbols"][s]["words"]
        top8 = ", ".join(f"{x['name']}({x['value']})" for x in w[:8])
        print(f"  {NAMES[s]:<8} {len(w):>2}词  例: {top8}")


if __name__ == "__main__":
    main()
