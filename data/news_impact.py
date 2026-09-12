"""新闻情绪对期货价格的影响分析。

- 按品种/日期聚合新闻情感(自建embedding打分), 使用处理后缓存 sentiment_daily。
- 影响时效设为 **2周**: 解释变量为"近2周衰减加权情绪", 被解释变量为"未来2周累计收益"。
- 与日线收益对齐, 计算: 同期相关、次日相关、2周相关、事件研究(情绪分位→2周收益)、OLS。
输出 web/news_impact.json 供网页展示。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from news_db import PROJ, SOURCE_META, connect, rebuild_sentiment_cache

RAW = PROJ / "data" / "raw" / "daily"
OUT = PROJ / "web" / "news_impact.json"

SYMBOLS = ["RB0", "CU0", "AU0", "SC0", "IF0", "T0"]
NAMES = {"RB0": "螺纹钢", "CU0": "沪铜", "AU0": "沪金", "SC0": "原油", "IF0": "沪深300", "T0": "10年国债"}

# 新闻影响时效: 2周(14 自然日 / 约 10 个交易日)
HORIZON_DAYS = 14
HORIZON_TDAYS = 10

# 来源主页(当新闻本身无原文链接时的回退)
SOURCE_HOME = {k: v["home"] for k, v in SOURCE_META.items()}


def article_url(source: str, dt: str, url: str) -> str:
    """真实文章/内容页链接: 原文优先; 经济日历->汇通财经日历当日页。"""
    if url:
        return url
    day = (dt or "")[:10].replace("-", "")
    if len(day) != 8:
        return ""
    if source == "经济日历":
        return f"https://rl.fx678.com/date/{day}.html"
    return ""


def daily_sentiment() -> pd.DataFrame:
    """已打分新闻明细(供显著日/新闻列表检索)。"""
    conn = connect()
    return pd.read_sql_query(
        "SELECT substr(dt,1,10) AS date, symbols, sentiment, title, source, url, content FROM news "
        "WHERE sentiment IS NOT NULL AND symbols != '' AND dt >= '2000-01-01'",
        conn,
    )


def symbol_sentiment_cache() -> pd.DataFrame:
    """读取处理后缓存: 按品种/日期的情感聚合。"""
    conn = connect()
    return pd.read_sql_query(
        "SELECT date, symbol, n, sent FROM sentiment_daily WHERE date >= '2000-01-01'",
        conn,
    )


def macro_sentiment() -> pd.DataFrame:
    conn = connect()
    return pd.read_sql_query(
        "SELECT substr(dt,1,10) AS date, AVG(sentiment) AS macro FROM news "
        "WHERE sentiment IS NOT NULL AND dt >= '2000-01-01' GROUP BY date",
        conn,
    )


def prices(sym: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{sym}.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change() * 100
    df["ret_next"] = df["ret"].shift(-1)
    # 未来2周(约10个交易日)累计收益%
    df["ret_fwd"] = (df["close"].shift(-HORIZON_TDAYS) / df["close"] - 1.0) * 100
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "close", "ret", "ret_next", "ret_fwd"]]


def _decayed_sentiment(by_date: pd.Series, index: pd.DatetimeIndex, horizon: int = HORIZON_DAYS) -> pd.Series:
    """近 horizon 天衰减加权情绪: 当日权重最高, 越久远权重线性递减, 过期(>horizon天)归零。"""
    if by_date.empty:
        return pd.Series(0.0, index=index)
    full = by_date.reindex(pd.date_range(index.min(), index.max(), freq="D"), fill_value=0.0)
    w = np.arange(1, horizon + 1, dtype=float)
    w = w / w.sum()
    arr = full.to_numpy()
    vals = np.zeros(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - horizon + 1)
        seg = arr[lo:i + 1]
        vals[i] = float((seg * w[-len(seg):]).sum())
    return pd.Series(vals, index=full.index).reindex(index).fillna(0.0)


def symbol_series(sym: str, sent_daily: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    p = prices(sym).set_index("date")
    idx = pd.to_datetime(p.index)

    s = sent_daily[sent_daily["symbol"] == sym].copy()
    s["date"] = pd.to_datetime(s["date"])
    sym_sent = _decayed_sentiment(s.set_index("date")["sent"], idx)

    m = macro.copy()
    m["date"] = pd.to_datetime(m["date"])
    macro_decay = _decayed_sentiment(m.set_index("date")["macro"], idx)

    df = p.copy()
    df["sym_sent"] = sym_sent.to_numpy()
    df["macro"] = macro_decay.to_numpy()
    return df.reset_index()


def analyse(sym: str, sent: pd.DataFrame, macro: pd.DataFrame, sent_daily: pd.DataFrame) -> dict:
    df = symbol_series(sym, sent, macro)

    def corr(a, b):
        m = df[[a, b]].dropna()
        if len(m) < 10 or m[a].std() == 0 or m[b].std() == 0:
            return None
        return float(m[a].corr(m[b]))

    # OLS: 未来2周收益 ~ 近2周情绪
    m = df[["sym_sent", "ret_fwd"]].dropna()
    if len(m) > 10 and m["sym_sent"].std() > 0:
        lr = stats.linregress(m["sym_sent"], m["ret_fwd"])
        slope, tstat, r2 = float(lr.slope), float(lr.rvalue), float(lr.rvalue ** 2)
    else:
        slope = tstat = r2 = None

    # 事件研究: 情绪分位 -> 未来2周收益均值
    ev = []
    valid = df[df["sym_sent"] != 0]
    if len(valid) >= 20:
        q = pd.qcut(valid["sym_sent"], 5, duplicates="drop")
        for k, grp in valid.groupby(q, observed=True):
            ev.append({"q": str(k), "mean_ret_fwd": float(grp["ret_fwd"].mean()), "n": int(len(grp))})

    # 显著相关日: |情绪| 与 |未来2周收益| 同时处于前 20%
    sig_dates: set[str] = set()
    mv = valid[valid["ret_fwd"].notna()]
    if len(mv) >= 20:
        ts = mv["sym_sent"].abs().quantile(0.8)
        tr = mv["ret_fwd"].abs().quantile(0.8)
        sig_dates = set(mv[(mv["sym_sent"].abs() >= ts) & (mv["ret_fwd"].abs() >= tr)]["date"])

    # 该品种相关新闻按日期归档: 显著日给出全部新闻, 其余日期只给条数(控制体积)
    sub = sent_daily[sent_daily["symbols"].str.contains(sym, na=False)]
    sig_news: dict[str, list] = {d: [] for d in sig_dates}
    news_count: dict[str, int] = {}
    for r in sub.itertuples():
        news_count[r.date] = news_count.get(r.date, 0) + 1
        if r.date in sig_dates:
            content = (r.content or "").strip()
            sig_news[r.date].append({
                "title": (r.title or "")[:70],
                "summary": content[:140],
                "source": r.source,
                "url": article_url(r.source, r.date, r.url or ""),
                "source_url": SOURCE_HOME.get(r.source, ""),
                "sent": None if r.sentiment is None or pd.isna(r.sentiment) else round(float(r.sentiment), 3),
            })

    # 时间序列(供网页): 仅含有新闻的日子
    ts = df[df["sym_sent"] != 0][["date", "sym_sent", "macro", "ret_fwd", "ret_next"]].tail(400)

    return {
        "symbol": sym,
        "name": NAMES[sym],
        "horizon_days": HORIZON_DAYS,
        "n_days": int((df["sym_sent"] != 0).sum()),
        "n_news": int(df["sym_sent"].ne(0).sum()),
        "corr_same": corr("sym_sent", "ret"),
        "corr_next": corr("sym_sent", "ret_next"),
        "corr_fwd": corr("sym_sent", "ret_fwd"),
        "macro_corr_fwd": corr("macro", "ret_fwd"),
        "slope": slope,
        "tstat": tstat,
        "r2": r2,
        "event": ev,
        "sig_dates": sorted(sig_dates),
        "sig_news": sig_news,
        "news_count": news_count,
        "top_news": symbol_news(sym, 20),
        "series": [
            {"date": r.date, "sent": round(float(r.sym_sent), 4),
             "ret_fwd": None if pd.isna(r.ret_fwd) else round(float(r.ret_fwd), 3),
             "ret_next": None if pd.isna(r.ret_next) else round(float(r.ret_next), 3)}
            for r in ts.itertuples()
        ],
    }


def main() -> None:
    conn = connect()
    rebuild_sentiment_cache(conn)
    sent_daily = daily_sentiment()
    sent = symbol_sentiment_cache()
    macro = macro_sentiment()
    out = {"symbols": [], "horizon_days": HORIZON_DAYS, "news": recent_news(30)}
    print(f"{'品种':<8}{'情感天数':>8}{'同期相关':>9}{'次日相关':>9}{'2周相关':>9}{'宏观2周':>9}{'斜率':>8}{'t值':>7}{'R²':>7}")
    for sym in SYMBOLS:
        r = analyse(sym, sent, macro, sent_daily)
        out["symbols"].append(r)
        f = lambda x: "  —  " if x is None else f"{x:+.3f}"
        print(f"{r['name']:<8}{r['n_days']:>8}{f(r['corr_same']):>9}{f(r['corr_next']):>9}"
              f"{f(r['corr_fwd']):>9}{f(r['macro_corr_fwd']):>9}{f(r['slope']):>8}{f(r['tstat']):>7}{f(r['r2']):>7}")
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[impact] -> {OUT} (影响时效 {HORIZON_DAYS} 天)")


def symbol_news(sym: str, n: int = 20) -> list[dict]:
    """该品种情感最强的 Top N 新闻(按|情绪|降序)。"""
    conn = connect()
    rows = conn.execute(
        "SELECT dt, source, title, symbols, sentiment, url FROM news "
        "WHERE sentiment IS NOT NULL AND (','||symbols||',') LIKE ? "
        "ORDER BY (url != '') DESC, ABS(sentiment) DESC LIMIT ?",
        (f"%,{sym},%", n),
    ).fetchall()
    return [
        {"dt": r[0], "source": r[1], "title": (r[2] or "")[:80], "symbols": r[3],
         "sent": None if r[4] is None else round(float(r[4]), 4),
         "url": article_url(r[1], r[0], r[5] or ""), "source_url": SOURCE_HOME.get(r[1], "")}
        for r in rows
    ]


def recent_news(n: int) -> list[dict]:
    conn = connect()
    rows = conn.execute(
        "SELECT dt, source, title, symbols, sentiment, url FROM news "
        "WHERE sentiment IS NOT NULL ORDER BY dt DESC LIMIT ?", (n,)
    ).fetchall()
    return [
        {"dt": r[0], "source": r[1], "title": (r[2] or "")[:80], "symbols": r[3],
         "sent": None if r[4] is None else round(float(r[4]), 4),
         "url": article_url(r[1], r[0], r[5] or ""), "source_url": SOURCE_HOME.get(r[1], "")}
        for r in rows
    ]


if __name__ == "__main__":
    main()
