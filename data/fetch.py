"""采集中国商品/金融期货主力连续合约数据(akshare/新浪源)。

- 日线: 默认取**全部可用历史**(RB/CU/AU 可回溯至 2005~2009, SC/IF/T 至上市日), 可选 --start 截断
- 60分钟/小时: 近期约8.5个月(免费源历史上限)

数据质量: 数值化后剔除 空值/非正收盘/最高<最低/重复日期/零成交 等异常行, 保证精度。

输出:
- data/raw/daily/{symbol}.csv   (date,open,high,low,close,volume,hold,settle)
- data/raw/hourly/{symbol}.csv  (date,open,high,low,close,volume,hold)
- data/metadata.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import akshare as ak
import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"

# symbol(主力连续后缀0) -> 品种元数据
SYMBOLS: dict[str, dict[str, str]] = {
    "RB0": {
        "品种名称": "螺纹钢",
        "交易所": "上海期货交易所",
        "交割对象主体": "螺纹钢(标准品 20mm HRB400)",
    },
    "CU0": {
        "品种名称": "沪铜",
        "交易所": "上海期货交易所",
        "交割对象主体": "阴极铜(标准铜)",
    },
    "AU0": {
        "品种名称": "沪金",
        "交易所": "上海期货交易所",
        "交割对象主体": "黄金(99.99%及以上)",
    },
    "SC0": {
        "品种名称": "原油",
        "交易所": "上海国际能源交易中心",
        "交割对象主体": "中质含硫原油",
    },
    "IF0": {
        "品种名称": "沪深300股指",
        "交易所": "中国金融期货交易所",
        "交割对象主体": "沪深300指数(现金交割)",
    },
    "T0": {
        "品种名称": "10年期国债",
        "交易所": "中国金融期货交易所",
        "交割对象主体": "10年期国债(虚拟券)",
    },
}

DAILY_HISTORY_YEARS = 0  # 0 = 全部可用历史(可由 --start 覆盖)


def _df_daily(symbol: str) -> pd.DataFrame:
    df = ak.futures_main_sina(symbol=symbol)
    df = df.rename(columns={
        "日期": "date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "volume",
        "持仓量": "hold",
        "动态结算价": "settle",
    })
    df["date"] = pd.to_datetime(df["date"])
    keep = ["date", "open", "high", "low", "close", "volume", "hold", "settle"]
    return df[keep].sort_values("date").reset_index(drop=True)


def _df_hourly(symbol: str) -> pd.DataFrame:
    df = ak.futures_zh_minute_sina(symbol=symbol, period="60")
    df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close", "volume", "hold"]].sort_values("date").reset_index(drop=True)


def _clean(df: pd.DataFrame, drop_zero_volume: bool = False) -> tuple[pd.DataFrame, dict]:
    """清洗并校验: 返回(清洗后数据, 质量统计)。"""
    raw_n = len(df)
    cols = [c for c in ("open", "high", "low", "close", "volume", "hold", "settle") if c in df.columns]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_ohlc = df[["open", "high", "low", "close"]].isna().any(axis=1)
    nonpos = (df["close"] <= 0) | (df["high"] < df["low"]) | (df["low"] <= 0)
    df = df[~bad_ohlc & ~nonpos]
    dup = df["date"].duplicated(keep="last")
    df = df[~dup]
    zero_vol = 0
    if drop_zero_volume:
        zv = df["volume"].fillna(0) <= 0
        zero_vol = int(zv.sum())
        df = df[~zv]

    df = df.sort_values("date").reset_index(drop=True)
    for col in ("volume", "hold"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")
    quality = {
        "原始行数": raw_n,
        "剔除_空值或异常价": int((bad_ohlc | nonpos).sum()),
        "剔除_重复日期": int(dup.sum()),
        "剔除_零成交": zero_vol,
        "有效行数": int(len(df)),
    }
    return df, quality


def _write_daily(symbol: str, daily_dir: Path, cutoff: date | None) -> dict:
    d, quality = _clean(_df_daily(symbol), drop_zero_volume=True)
    if cutoff is not None:
        d = d[d["date"].dt.date >= cutoff].reset_index(drop=True)
    d.to_csv(daily_dir / f"{symbol}.csv", index=False)
    return {
        "精度": "1天",
        "时间跨度": f"{d['date'].iloc[0].date()} ~ {d['date'].iloc[-1].date()}",
        "样本量": int(len(d)),
        "数据质量": quality,
    }


def _write_hourly(symbol: str, hourly_dir: Path) -> dict:
    h, quality = _clean(_df_hourly(symbol))
    h.to_csv(hourly_dir / f"{symbol}.csv", index=False)
    return {
        "精度": "60分钟",
        "时间跨度": f"{h['date'].iloc[0]} ~ {h['date'].iloc[-1]}",
        "样本量": int(len(h)),
        "数据质量": quality,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="采集期货主力连续数据")
    p.add_argument("--start", default=None, help="日线起始日期 YYYY-MM-DD(默认全部可用历史)")
    p.add_argument("--years", type=int, default=DAILY_HISTORY_YEARS,
                   help="日线保留年数(0=全部; 与 --start 同时给出时以 --start 为准)")
    args = p.parse_args()
    cutoff = date.fromisoformat(args.start) if args.start else (
        date(date.today().year - args.years, 1, 1) if args.years > 0 else None
    )

    daily_dir = RAW / "daily"
    hourly_dir = RAW / "hourly"
    daily_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict] = {}
    for symbol, meta in SYMBOLS.items():
        print(f"[fetch] {symbol} ({meta['品种名称']}) ...")

        try:
            daily_meta = _write_daily(symbol, daily_dir, cutoff)
            q = daily_meta["数据质量"]
            print(f"  日线 {daily_meta['样本量']} ({daily_meta['时间跨度']}) "
                  f"清洗: -{q['剔除_空值或异常价']}/-{q['剔除_重复日期']}/-{q['剔除_零成交']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! 日线失败: {type(exc).__name__}: {exc}")
            daily_meta = {"精度": "1天", "时间跨度": "N/A", "样本量": 0}

        try:
            hourly_meta = _write_hourly(symbol, hourly_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! 小时失败: {type(exc).__name__}: {exc}")
            hourly_meta = {"精度": "60分钟", "时间跨度": "N/A", "样本量": 0}

        metadata[symbol] = {
            **meta,
            "主力连续合约": symbol,
            "日线": daily_meta,
            "小时": hourly_meta,
        }

    meta_path = RAW.parent / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] metadata -> {meta_path}")
    for symbol, m in metadata.items():
        d, h = m["日线"], m["小时"]
        print(f"  {symbol:>4}: 日线 {d['样本量']} ({d['时间跨度']}) | 小时 {h['样本量']} ({h['时间跨度']})")


if __name__ == "__main__":
    main()
