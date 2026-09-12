"""新闻库(SQLite)与品种关联工具。

表 news:
  id           TEXT PK  (source+dt+title 的哈希)
  source       TEXT     来源
  dt           TEXT     ISO 时间
  title        TEXT
  content      TEXT
  url          TEXT
  symbols      TEXT     逗号分隔的相关品种代码(可空)
  sentiment    REAL     自建embedding情感分[-1,1](可空)
  collected_at TEXT

表 sentiment_daily (处理后缓存):
  date   TEXT
  symbol TEXT
  n      INTEGER  当日该品种相关新闻数
  sent   REAL     当日该品种平均情感
  PRIMARY KEY(date, symbol)
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
NEWS_DIR = PROJ / "data" / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = NEWS_DIR / "news.db"

# 品种 -> 关联关键词(命中即关联)
SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "RB0": ["螺纹钢", "钢材", "钢价", "钢铁", "铁矿石", "钢厂", "高炉", "热卷", "焦炭", "焦煤",
            "地产", "房地产", "基建", "粗钢", "限产", "钢厂利润"],
    "CU0": ["铜", "铜价", "电解铜", "铜矿", "精炼铜", "有色金属", "铜精矿", "伦铜", "沪铜"],
    "AU0": ["黄金", "金价", "贵金属", "美联储", "加息", "降息", "美元指数", "避险", "白银",
            "实际利率", "非农", "通胀"],
    "SC0": ["原油", "油价", "OPEC", "欧佩克", "减产", "增产", "石油", "汽油", "柴油", "炼油",
            "页岩油", "布伦特", "WTI", "能源"],
    "IF0": ["股指", "股市", "A股", "沪深300", "上证", "大盘", "股票", "权益市场", "北向资金"],
    "T0": ["国债", "债券", "收益率", "利率", "央行", "降准", "货币政策", "MLF", "LPR",
           "公开市场操作", "资金面", "流动性"],
}

# 来源元数据: kind=article 为可读文章/快讯, structured 为结构化文本, video 为视频/回放(采集时丢弃)
SOURCE_META: dict[str, dict[str, str]] = {
    "财联社": {"home": "https://www.cls.cn/telegraph", "kind": "article"},
    "东方财富": {"home": "https://kuaixun.eastmoney.com/", "kind": "article"},
    "同花顺": {"home": "https://news.10jqka.com.cn/realtimenews.html", "kind": "article"},
    "新浪财经": {"home": "https://finance.sina.com.cn/7x24/", "kind": "article"},
    "上海金属网": {"home": "https://www.shmet.com/", "kind": "article"},
    "财经早餐": {"home": "https://finance.eastmoney.com/", "kind": "article"},
    "富途牛牛": {"home": "https://news.futunn.com/main", "kind": "article"},
    "财新网": {"home": "https://www.caixin.com/", "kind": "article"},
    "华尔街见闻": {"home": "https://wallstreetcn.com/live/global", "kind": "article"},
    "金十数据": {"home": "https://www.jin10.com/", "kind": "article"},
    "经济日历": {"home": "https://data.eastmoney.com/cjsj/", "kind": "structured"},
    "华尔街见闻日历": {"home": "https://wallstreetcn.com/calendar", "kind": "structured"},
    "新闻联播": {"home": "https://tv.cctv.com/lm/xwlb/", "kind": "video"},
}

# 标题命中即视为视频/直播/回放类(无法快速看出主体内容), 采集时丢弃
VIDEO_MARKERS = ("新闻联播", "联播", "视频", "直播", "回放", "完整版", "纪录片", "微视频",
                 "长视频", "实况", "录像", "发布会回放", "全程回顾")


def is_readable(title: str, content: str = "") -> bool:
    """可读性过滤: 丢弃视频/直播/回放标题, 以及正文过短(无法快速看出主体)的条目。"""
    t = (title or "").strip()
    c = (content or "").strip()
    if any(m in t for m in VIDEO_MARKERS):
        return False
    if len(t) < 6 and len(c) < 20:
        return False
    return True


SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id           TEXT PRIMARY KEY,
    source       TEXT,
    dt           TEXT,
    title        TEXT,
    content      TEXT,
    url          TEXT,
    symbols      TEXT,
    sentiment    REAL,
    collected_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_dt ON news(dt);
CREATE INDEX IF NOT EXISTS idx_news_symbols ON news(symbols);
CREATE TABLE IF NOT EXISTS sentiment_daily (
    date   TEXT,
    symbol TEXT,
    n      INTEGER,
    sent   REAL,
    PRIMARY KEY (date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_sent_daily_symbol ON sentiment_daily(symbol);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def make_id(source: str, dt: str, title: str) -> str:
    return hashlib.md5(f"{source}|{dt}|{title}".encode("utf-8")).hexdigest()


def tag_symbols(text: str) -> str:
    hits = [sym for sym, kws in SYMBOL_KEYWORDS.items() if any(k in text for k in kws)]
    return ",".join(hits)


def upsert(conn: sqlite3.Connection, rows: list[dict | None]) -> int:
    """插入去重(自动丢弃可读性过滤后的 None), 返回新增条数。"""
    rows = [r for r in rows if r]
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.executemany(
        """INSERT OR IGNORE INTO news
           (id, source, dt, title, content, url, symbols, sentiment, collected_at)
           VALUES (:id, :source, :dt, :title, :content, :url, :symbols, NULL, :collected_at)""",
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    return after - before


def build_row(source: str, dt: str, title: str, content: str, url: str = "") -> dict | None:
    """构造待入库行; 不可读(视频/回放/过短)返回 None。"""
    if not is_readable(title, content):
        return None
    text = f"{title} {content}"
    return {
        "id": make_id(source, dt, title),
        "source": source,
        "dt": dt,
        "title": title.strip(),
        "content": (content or "").strip(),
        "url": url or "",
        "symbols": tag_symbols(text),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }


def purge_non_readable(conn: sqlite3.Connection) -> int:
    """删除已入库的视频/回放类来源(如新闻联播)条目。"""
    removed = 0
    for src, meta in SOURCE_META.items():
        if meta["kind"] == "video":
            removed += conn.execute("DELETE FROM news WHERE source=?", (src,)).rowcount
    conn.commit()
    return removed


def rebuild_sentiment_cache(conn: sqlite3.Connection) -> int:
    """重建"处理后缓存" sentiment_daily: 按品种/日期聚合已打分新闻情感。

    影响分析与词云只读该缓存做聚合, 避免每次全量扫描 news 表。
    """
    conn.executescript("DELETE FROM sentiment_daily;")
    # 一条新闻可关联多个品种, 需按逗号拆分展开
    cur = conn.execute(
        "SELECT substr(dt,1,10) AS date, symbols, sentiment FROM news "
        "WHERE sentiment IS NOT NULL AND symbols != '' AND dt >= '2000-01-01'"
    )
    agg: dict[tuple[str, str], list[float]] = {}
    for date, symbols, sent in cur:
        for sym in str(symbols).split(","):
            sym = sym.strip()
            if not sym:
                continue
            agg.setdefault((date, sym), []).append(float(sent))
    conn.executemany(
        "INSERT OR REPLACE INTO sentiment_daily(date, symbol, n, sent) VALUES (?,?,?,?)",
        [(d, s, len(v), sum(v) / len(v)) for (d, s), v in agg.items()],
    )
    conn.commit()
    return len(agg)


def stats(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    tagged = conn.execute("SELECT COUNT(*) FROM news WHERE symbols != ''").fetchone()[0]
    scored = conn.execute("SELECT COUNT(*) FROM news WHERE sentiment IS NOT NULL").fetchone()[0]
    print(f"[news.db] 总数={total} 已关联品种={tagged} 已情感打分={scored}")
    for src, n in conn.execute("SELECT source, COUNT(*) FROM news GROUP BY source ORDER BY 2 DESC"):
        print(f"    {src:<20} {n}")


if __name__ == "__main__":
    stats(connect())
