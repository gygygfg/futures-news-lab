"""采集中国商品/金融期货主力连续合约数据(akshare/新浪源)。

- 日线: 近5年(实际接口可回溯多年)
- 60分钟/小时: 近期约8.5个月(免费源历史上限)

输出:
- data/raw/daily/{symbol}.csv   (date,open,high,low,close,volume,hold,settle)
- data/raw/hourly/{symbol}.csv  (date,open,high,low,close,volume,hold)
- data/metadata.json
"""

from __future__ import annotations

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

DAILY_HISTORY_YEARS = 5  # 日线保留近5年, 满足>1年


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


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("volume", "hold"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return df


def main() -> None:
    daily_dir = RAW / "daily"
    hourly_dir = RAW / "hourly"
    daily_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)

    cutoff = date(date.today().year - DAILY_HISTORY_YEARS, 1, 1)
    metadata: dict[str, dict] = {}

    for symbol, meta in SYMBOLS.items():
        print(f"[fetch] {symbol} ({meta['品种名称']}) ...")

        # --- 日线 ---
        try:
            d = _df_daily(symbol)
            d = _clean(d)
            d = d[d["date"].dt.date >= cutoff].reset_index(drop=True)
            d.to_csv(daily_dir / f"{symbol}.csv", index=False)
            daily_meta = {
                "精度": "1天",
                "时间跨度": f"{d['date'].iloc[0].date()} ~ {d['date'].iloc[-1].date()}",
                "样本量": int(len(d)),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"  ! 日线失败: {type(exc).__name__}: {exc}")
            daily_meta = {"精度": "1天", "时间跨度": "N/A", "样本量": 0}

        # --- 小时(60分钟) ---
        try:
            h = _df_hourly(symbol)
            h = _clean(h)
            h.to_csv(hourly_dir / f"{symbol}.csv", index=False)
            hourly_meta = {
                "精度": "60分钟",
                "时间跨度": f"{h['date'].iloc[0]} ~ {h['date'].iloc[-1]}",
                "样本量": int(len(h)),
            }
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
