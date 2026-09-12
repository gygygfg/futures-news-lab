"""新闻采集: 以可读文章/快讯为主, 丢弃视频/回放类来源。

来源分层:
- 文章源(article): 财联社/东方财富/同花顺/新浪财经/上海金属网/财经早餐/富途牛牛/财新网
  —— 正文可读, 带链接, 可快速看出主体内容。
- 结构化文本(structured): 经济日历(事件+公布/预期/前值)。
- 已移除: 新闻联播(视频回放, 无法快速看出主体)。

用法:
  uv run python data/fetch_news.py live
  uv run python data/fetch_news.py backfill --start 2024-01-01 --end 2026-09-10
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import akshare as ak
import pandas as pd
import requests

from news_db import NEWS_DIR, build_row, connect, purge_non_readable, rebuild_sentiment_cache, stats, upsert

CACHE_DIR = NEWS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def _cached_rows(key: str, producer) -> list[dict]:
    """磁盘缓存: 同一来源/日期只抓一次, 大范围回填可断点续跑、重复运行秒回。"""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    rows = [r for r in producer()]
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


# ---------------- 历史回填 ----------------

def fetch_econ_rows(d: date) -> list[dict]:
    """经济日历(结构化文本, 非视频)。"""
    key = f"econ_{d.strftime('%Y%m%d')}"
    return _cached_rows(key, lambda: _fetch_econ_rows_uncached(d))


def _fetch_econ_rows_uncached(d: date) -> list[dict]:
    df = ak.news_economic_baidu(date=d.strftime("%Y%m%d"))
    rows = []
    for _, r in df.iterrows():
        title = _s(r.get("事件"))
        if not title:
            continue
        tm = _s(r.get("时间")) or "00:00"
        content = (f"地区:{_s(r.get('地区'))} 公布:{_s(r.get('公布'))} 预期:{_s(r.get('预期'))} "
                   f"前值:{_s(r.get('前值'))} 重要性:{_s(r.get('重要性'))}")
        rows.append(build_row("经济日历", f"{d.isoformat()} {tm}", title, content))
    return rows


def fetch_ws_calendar_rows(d: date) -> list[dict]:
    """华尔街见闻财经日历(结构化文本, 带事件链接)。"""
    key = f"wscal_{d.strftime('%Y%m%d')}"
    return _cached_rows(key, lambda: _fetch_ws_calendar_uncached(d))


def _fetch_ws_calendar_uncached(d: date) -> list[dict]:
    df = ak.macro_info_ws(date=d.strftime("%Y%m%d"))
    rows = []
    for _, r in df.iterrows():
        title = _s(r.get("事件"))
        if not title:
            continue
        tm = _s(r.get("时间")) or f"{d.isoformat()} 00:00:00"
        content = (f"地区:{_s(r.get('地区'))} 今值:{_s(r.get('今值'))} 预期:{_s(r.get('预期'))} "
                   f"前值:{_s(r.get('前值'))} 重要性:{_s(r.get('重要性'))}")
        rows.append(build_row("华尔街见闻日历", tm, title, content, _s(r.get("链接"))))
    return rows


def fetch_day(d: date) -> list[dict]:
    rows: list[dict] = []
    for name, fn in (("经济日历", fetch_econ_rows), ("华尔街见闻日历", fetch_ws_calendar_rows)):
        try:
            rows += fn(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name} {d}: {type(exc).__name__}: {exc}")
    return rows


# ---------------- 文章源(可读性高) ----------------

_DATE_IN_URL = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _date_from_url(url: str) -> str:
    m = _DATE_IN_URL.search(url or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def _feed_cjzc() -> list[dict]:
    """东方财富财经早餐: 每日一篇可读摘要。"""
    df = ak.stock_info_cjzc_em()
    return [build_row("财经早餐", _s(r.get("发布时间")), _s(r.get("标题")),
                      _s(r.get("摘要")), _s(r.get("链接"))) for _, r in df.iterrows()]


def _feed_futu() -> list[dict]:
    """富途牛牛快讯: 短文章, 带正文与原文链接。"""
    df = ak.stock_info_global_futu()
    return [build_row("富途牛牛", _s(r.get("发布时间")), _s(r.get("标题")),
                      _s(r.get("内容")), _s(r.get("链接"))) for _, r in df.iterrows()]


def _feed_caixin() -> list[dict]:
    """财新网要闻: 摘要式文章, 从链接解析发布日期。"""
    df = ak.stock_news_main_cx()
    rows = []
    for _, r in df.iterrows():
        url = _s(r.get("url"))
        title = _s(r.get("summary"))
        rows.append(build_row("财新网", _date_from_url(url), title, title, url))
    return rows


# 无日期参数的文章源(抓取最新一页), 用于 live 与回填
ARTICLE_FEEDS = {"财经早餐": _feed_cjzc, "富途牛牛": _feed_futu, "财新网": _feed_caixin}


def _in_range(dt: str, start: date, end: date) -> bool:
    day = (dt or "")[:10]
    if len(day) != 10:
        return False
    try:
        d = datetime.fromisoformat(day).date()
    except ValueError:
        return False
    return start <= d <= end


def fetch_article_feeds(start: date | None = None, end: date | None = None) -> list[dict]:
    """抓取全部文章源; 给定日期区间时按区间过滤。"""
    rows: list[dict] = []
    for name, fn in ARTICLE_FEEDS.items():
        try:
            got = _cached_rows(f"feed_{name}_{date.today().isoformat()}", fn)
            if start and end:
                got = [r for r in got if r and _in_range(r["dt"], start, end)]
            rows += got
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {type(exc).__name__}: {exc}")
    return rows


def run_backfill(start: date, end: date, workers: int = 8) -> None:
    conn = connect()
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(fetch_day, dates), 1):
            total += upsert(conn, rows)
            if i % 30 == 0 or i == len(dates):
                print(f"  [{i}/{len(dates)}] {dates[i-1].isoformat()} 累计新增 {total}", flush=True)
    art = upsert(conn, fetch_article_feeds(start, end))
    total += art
    purged = purge_non_readable(conn)
    print(f"[backfill] 经济日历+文章 完成, 新增 {total} 条(其中文章 {art}), 清理视频类 {purged} 条")
    rebuild_sentiment_cache(conn)
    stats(conn)


# ---------------- 实时快照 ----------------

def live_cls(conn) -> int:
    df = ak.stock_info_global_cls(symbol="全部")
    rows = []
    for _, r in df.iterrows():
        dt = f"{_s(r.get('发布日期'))} {_s(r.get('发布时间'))}".strip()
        rows.append(build_row("财联社", dt, _s(r.get("标题")) or _s(r.get("内容"))[:40], _s(r.get("内容"))))
    return upsert(conn, rows)


def live_em(conn) -> int:
    df = ak.stock_info_global_em()
    rows = [build_row("东方财富", _s(r.get("发布时间")), _s(r.get("标题")), _s(r.get("摘要")), _s(r.get("链接")))
            for _, r in df.iterrows()]
    return upsert(conn, rows)


def live_ths(conn) -> int:
    df = ak.stock_info_global_ths()
    rows = [build_row("同花顺", _s(r.get("发布时间")), _s(r.get("标题")), _s(r.get("内容")), _s(r.get("链接")))
            for _, r in df.iterrows()]
    return upsert(conn, rows)


def live_sina(conn) -> int:
    df = ak.stock_info_global_sina()
    rows = [build_row("新浪财经", _s(r.get("时间")), _s(r.get("内容"))[:40], _s(r.get("内容")))
            for _, r in df.iterrows()]
    return upsert(conn, rows)


def live_shmet(conn) -> int:
    df = ak.futures_news_shmet(symbol="全部")
    rows = []
    for _, r in df.iterrows():
        content = _s(r.get("内容"))
        dt = _s(r.get("发布时间"))
        rows.append(build_row("上海金属网", dt, content[:40], content))
    return upsert(conn, rows)


def live_articles(conn) -> int:
    """可读文章源实时抓取(财经早餐/富途/财新)。"""
    return upsert(conn, fetch_article_feeds())


def live_ws_calendar(conn) -> int:
    """华尔街见闻日历(当日)。"""
    return upsert(conn, fetch_ws_calendar_rows(date.today()))


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def live_wscn(conn) -> int:
    """华尔街见闻 7x24 快讯(可读文章, 带原文链接)。"""
    r = requests.get(
        "https://api-one.wallstcn.com/apiv1/content/lives",
        params={"channel": "global-channel", "limit": 50}, headers=_UA, timeout=20,
    )
    items = (r.json().get("data") or {}).get("items") or []
    rows = []
    for it in items:
        ts = it.get("display_time")
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
        text = _s(it.get("content_text"))
        title = _s(it.get("title")) or text[:40]
        rows.append(build_row("华尔街见闻", dt, title, text, _s(it.get("uri"))))
    return upsert(conn, rows)


def live_jin10(conn) -> int:
    """金十数据快讯(可读文章, 标题嵌于正文【】)。"""
    h = {**_UA, "Referer": "https://www.jin10.com/", "x-app-id": "bVBF4FyRTn5NJF5n", "x-version": "1.0.0"}
    r = requests.get(
        "https://flash-api.jin10.com/get_flash_list",
        params={"channel": "-8200", "vip": "1"}, headers=h, timeout=20,
    )
    data = r.json().get("data") or []
    rows = []
    for it in data:
        d = it.get("data") or {}
        content = _s(d.get("content"))
        m = re.match(r"^【([^】]+)】", content)
        title = m.group(1) if m else content[:40]
        url = _s(d.get("source_link")) or f"https://flash.jin10.com/detail/{_s(it.get('id'))}"
        rows.append(build_row("金十数据", _s(it.get("time")), title, content, url))
    return upsert(conn, rows)


LIVE_SOURCES = {
    "财联社": live_cls, "东方财富": live_em, "同花顺": live_ths,
    "新浪财经": live_sina, "上海金属网": live_shmet, "文章源": live_articles,
    "华尔街见闻": live_wscn, "金十数据": live_jin10, "华尔街见闻日历": live_ws_calendar,
}


def run_live() -> None:
    conn = connect()
    total = 0
    for name, fn in LIVE_SOURCES.items():
        try:
            n = fn(conn)
            total += n
            print(f"  {name:<10} 新增 {n}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {type(exc).__name__}: {exc}")
    purge_non_readable(conn)
    rebuild_sentiment_cache(conn)
    print(f"[live] 完成, 新增 {total} 条")
    stats(conn)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("live")
    b = sub.add_parser("backfill")
    b.add_argument("--start", required=True)
    b.add_argument("--end", default=date.today().isoformat())
    b.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if args.mode == "live":
        run_live()
    else:
        run_backfill(datetime.fromisoformat(args.start).date(),
                     datetime.fromisoformat(args.end).date(), workers=args.workers)


if __name__ == "__main__":
    main()
