"""演示数据可量化性: 时序图 + 收益率统计 + 简单均线策略。

输出图片到 data/demo/
- 01_daily_prices.png     6品种日线价格(归一化)
- 02_daily_volume.png     RB 日线价格+成交量
- 03_hourly_kline.png     RB 近8个月60分钟K线
- 04_rainbow.png          均线策略净值(日线, RB/CU/AU/IF)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

RAW = Path(__file__).resolve().parent / "raw"
OUT = Path(__file__).resolve().parent / "demo"
OUT.mkdir(exist_ok=True)

SYMBOLS = ["RB0", "CU0", "AU0", "SC0", "IF0", "T0"]
NAMES = {
    "RB0": "螺纹钢", "CU0": "沪铜", "AU0": "沪金",
    "SC0": "原油", "IF0": "沪深300", "T0": "10年国债",
}
plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_daily(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / "daily" / f"{symbol}.csv", parse_dates=["date"])
    return df.set_index("date")


def daily_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


def strategy_equity(df: pd.DataFrame, fast: int = 10, slow: int = 40) -> pd.DataFrame:
    """最简双均线策略: 金叉持有多头, 死叉空仓。"""
    out = pd.DataFrame(index=df.index)
    out["fast"] = df["close"].rolling(fast).mean()
    out["slow"] = df["close"].rolling(slow).mean()
    out["pos"] = (out["fast"] > out["slow"]).astype(int)
    out["ret"] = df["close"].pct_change().fillna(0.0)
    out["strat"] = (out["pos"].shift(1).fillna(0) * out["ret"]).cumsum()
    out["buyhold"] = out["ret"].cumsum()
    return out.dropna()


def plot_daily_prices() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, sym in zip(axes.ravel(), SYMBOLS):
        d = load_daily(sym)
        norm = d["close"] / d["close"].iloc[0] * 100
        ax.plot(d.index, norm, lw=1.1)
        ax.set_title(f"{NAMES[sym]} ({sym})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("近5年 主力连续日线价格(归一化=100)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "01_daily_prices.png", dpi=110)
    plt.close(fig)


def plot_rb_volume() -> None:
    d = load_daily("RB0")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(d.index, d["close"], lw=1.2, color="tab:red")
    ax1.set_title(f"{NAMES['RB0']} 日线: 收盘价 + 成交量", fontsize=12)
    ax1.grid(alpha=0.3)
    ax2.bar(d.index, d["volume"], width=1.5, color="tab:blue", alpha=0.6)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "02_daily_volume.png", dpi=110)
    plt.close(fig)


def plot_hourly_kline() -> None:
    h = pd.read_csv(RAW / "hourly" / "RB0.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(13, 6))
    w = h["date"].diff().median() / 2
    for _, r in h.iterrows():
        c = "tab:red" if r["close"] >= r["open"] else "tab:green"
        ax.vlines(r["date"], r["low"], r["high"], color=c, lw=0.5)
        ax.bar(r["date"], r["close"] - r["open"], bottom=r["open"], width=w, color=c, alpha=0.9)
    ax.set_title(f"{NAMES['RB0']} 60分钟K线 (近{len(h)}根, {h['date'].iloc[0]:%Y-%m-%d}~{h['date'].iloc[-1]:%Y-%m-%d})",
                 fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_hourly_kline.png", dpi=110)
    plt.close(fig)


def plot_strategy() -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for sym in ["RB0", "CU0", "AU0", "IF0"]:
        eq = strategy_equity(load_daily(sym))
        ax.plot(eq.index, eq["strat"], lw=1.2, label=f"{NAMES[sym]} MA10×40 策略")
    ax.plot(eq.index, eq["buyhold"], lw=1.0, ls="--", color="tab:gray", label="buy & hold (IF0)")
    ax.set_title("双均线策略累积收益(日线, 近5年)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "04_strategy.png", dpi=110)
    plt.close(fig)


def print_stats() -> None:
    print("=== 日线收益率统计(近5年) ===")
    print(f"{'品种':<8}{'年化波动':>10}{'年化收益':>10}{'最大回撤':>10}")
    for sym in SYMBOLS:
        ret = daily_returns(load_daily(sym))
        base = load_daily(sym)["close"]
        ann_vol = ret.std() * (252 ** 0.5)
        ann_ret = base.iloc[-1] / base.iloc[0] - 1
        mdd = (base / base.cummax() - 1).min()
        print(f"{NAMES[sym]:<8}{ann_vol:>10.2%}{ann_ret:>10.1%}{mdd:>10.1%}")


def main() -> None:
    plot_daily_prices()
    plot_rb_volume()
    plot_hourly_kline()
    plot_strategy()
    print_stats()
    print(f"[demo] 图表已保存至 {OUT}/")


if __name__ == "__main__":
    main()
