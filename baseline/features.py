# -*- coding: utf-8 -*-
"""
Technical indicators for "will the next BTCUSDT 5-minute candle close up or down?".

One row = one 5m candle that has JUST CLOSED (index = candle open time, UTC). Every
indicator uses data up to the end of that candle only. The 15m and 1h timeframes are
rebuilt from the 5m data and only closed higher-timeframe candles are used (merge_asof
on close time); the part of the higher-timeframe candle still forming goes into the
separate *_dev columns.

Label y = 1 if the next 5m candle closes above its open, 0 if below, NaN for a doji
or a data gap.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M5_CSV = os.path.join(HERE, "..", "data", "BTCUSDT_5m.csv")
# last candle of the published run; set to None to use newer data as well
DATA_END = "2026-09-24 08:00"
COLS = ["Date", "Time", "Open", "High", "Low", "Close", "TickCount", "Volume", "Spread"]

# indicator group -> column names (used for the drop-one-group ablation)
GROUPS = {}


def _reg(group, *names):
    GROUPS.setdefault(group, []).extend(names)


def load_m5(start="2021-09-01"):
    df = pd.read_csv(M5_CSV, header=None, names=COLS,
                     dtype={"Date": str, "Time": str})
    df.index = pd.to_datetime(df["Date"] + df["Time"], format="%Y%m%d%H:%M:%S")
    df = df.loc[start:DATA_END, ["Open", "High", "Low", "Close", "TickCount"]]
    df.columns = ["o", "h", "l", "c", "n"]
    # full 5-minute grid: gaps become NaN so two distant candles are never joined
    full = pd.date_range(df.index[0], df.index[-1], freq="5min")
    return df.reindex(full)


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _atr(df, n=14):
    pc = df["c"].shift()
    tr = pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _streak(sign):
    """Signed count of consecutive same-color candles (+3 = three up candles in a row)."""
    grp = (sign != sign.shift()).cumsum()
    return sign * sign.groupby(grp).cumcount().add(1)


def _htf_block(df, p):
    """Reduced set for a higher timeframe: trend + momentum + volatility."""
    c = df["c"]
    atr = _atr(df)
    e9, e21, e50 = _ema(c, 9), _ema(c, 21), _ema(c, 50)
    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    out = pd.DataFrame(index=df.index)
    out[f"{p}_ema_stack"] = np.sign(e9 - e21) + np.sign(e21 - e50)
    out[f"{p}_dist_ema21_atr"] = (c - e21) / atr
    out[f"{p}_slope_ema21_atr"] = (e21 - e21.shift(3)) / atr
    out[f"{p}_rsi14"] = _rsi(c)
    out[f"{p}_macd_hist_atr"] = hist / atr
    out[f"{p}_macd_hist_chg"] = hist.diff() / atr
    out[f"{p}_ret1_atr"] = (c - c.shift()) / atr
    out[f"{p}_atr_pct"] = atr / c * 100
    return out


def build(start="2021-09-01"):
    GROUPS.clear()
    df = load_m5(start)
    c, o, h, l = df["c"], df["o"], df["h"], df["l"]
    f = pd.DataFrame(index=df.index)
    atr = _atr(df)

    # --- Trend
    emas = {n: _ema(c, n) for n in (9, 21, 50, 200)}
    for n, e in emas.items():
        f[f"dist_ema{n}_atr"] = (c - e) / atr
    for n in (9, 21, 50):
        f[f"slope_ema{n}_atr"] = (emas[n] - emas[n].shift(3)) / atr
    f["ema_stack"] = (np.sign(emas[9] - emas[21]) + np.sign(emas[21] - emas[50])
                      + np.sign(emas[50] - emas[200]))
    _reg("trend", *[x for x in f.columns])

    # --- Momentum
    rsi = _rsi(c)
    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    f["rsi14"] = rsi
    f["rsi14_chg3"] = rsi - rsi.shift(3)
    f["macd_hist_atr"] = hist / atr
    f["macd_hist_chg"] = hist.diff() / atr
    for k in (1, 3, 12):
        f[f"ret{k}_atr"] = (c - c.shift(k)) / atr
    _reg("momentum", "rsi14", "rsi14_chg3", "macd_hist_atr", "macd_hist_chg",
         "ret1_atr", "ret3_atr", "ret12_atr")

    # --- Volatility
    f["atr_pct"] = atr / c * 100
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    bbw = 4 * sd20 / ma20
    f["bb_width_pctile"] = bbw.rolling(288).rank(pct=True) * 100
    f["bb_pos"] = (c - ma20) / (2 * sd20)
    f["range_atr"] = (h - l) / atr
    _reg("volatility", "atr_pct", "bb_width_pctile", "bb_pos", "range_atr")

    # --- Candle shape
    rng = (h - l).replace(0, np.nan)
    f["body_ratio"] = ((c - o) / rng).fillna(0)
    f["upper_wick"] = ((h - np.maximum(o, c)) / rng).fillna(0)
    f["lower_wick"] = ((np.minimum(o, c) - l) / rng).fillna(0)
    f["streak"] = _streak(np.sign(c - o))
    _reg("candle_shape", "body_ratio", "upper_wick", "lower_wick", "streak")

    # --- Position in recent range
    for n in (20, 50):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"pos_in_range_{n}"] = (c - lo) / (hi - lo) * 100
    f["dist_high50_atr"] = (h.rolling(50).max() - c) / atr
    f["dist_low50_atr"] = (c - l.rolling(50).min()) / atr
    _reg("position", "pos_in_range_20", "pos_in_range_50", "dist_high50_atr", "dist_low50_atr")

    # --- Activity (number of trades, not volume)
    f["activity_ratio"] = np.log(df["n"] / df["n"].rolling(20).mean())
    f["activity_ratio_288"] = np.log(df["n"].rolling(12).mean() / df["n"].rolling(288).mean())
    _reg("activity", "activity_ratio", "activity_ratio_288")

    # --- Time (CLOSE time of the finished candle = open time of the candle to predict)
    t_next = f.index + pd.Timedelta("5min")
    hr = t_next.hour + t_next.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["hour"] = t_next.hour
    f["dow"] = t_next.dayofweek
    _reg("time", "hour_sin", "hour_cos", "hour", "dow")

    # --- Higher timeframes: closed candles only, joined on close time
    close_t = pd.Series(f.index + pd.Timedelta("5min"), index=f.index)
    for rule, p in (("15min", "m15"), ("1h", "h1")):
        agg = df.resample(rule, label="left", closed="left").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "n": "sum"}).dropna()
        blk = _htf_block(agg, p)
        blk.index = blk.index + pd.Timedelta(rule)          # index = higher-TF close time
        merged = pd.merge_asof(close_t.rename("t").to_frame(), blk,
                               left_on="t", right_index=True, direction="backward")
        merged.index = f.index
        for col in blk.columns:
            f[col] = merged[col]
        # higher-TF candle still forming: built from closed 5m candles, no look-ahead
        start_of = f.index.floor(rule)
        dev_open = o.groupby(start_of).transform("first")
        f[f"{p}_dev_ret_atr"] = (c - dev_open) / atr
        _reg(f"tf_{p}", *blk.columns, f"{p}_dev_ret_atr")

    # --- Label: the next candle
    no, nc = o.shift(-1), c.shift(-1)
    y = pd.Series(np.where(nc > no, 1.0, np.where(nc < no, 0.0, np.nan)), index=f.index)
    f["y"] = y
    f["prev_up"] = (c > o).astype(float)          # for the momentum / reversal baselines
    f["next_ret_bps"] = (nc / no - 1) * 1e4       # to translate into gross P&L
    return f


if __name__ == "__main__":
    f = build()
    out = os.path.join(HERE, "features_m5.parquet")
    f.to_parquet(out)
    print(f.shape, "->", out)
    print({g: len(v) for g, v in GROUPS.items()})
