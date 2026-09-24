# -*- coding: utf-8 -*-
"""
Download BTCUSDT candles from the Binance REST API into CSV files in this folder.
If a file does not exist yet it is downloaded from --start; otherwise new candles
are appended after the last one.

Format (no header):
    Date(YYYYMMDD),Time(HH:MM:SS),Open,High,Low,Close,TickCount,Volume,Spread
Note: TickCount and Volume both hold the NUMBER OF TRADES, not the coin volume;
Spread is always 0. Times are UTC, stamped at the candle OPEN.

Only closed candles are written. Run again at any time to extend the files:
    python update_binance.py                          # BTCUSDT_5m.csv from 2021-08-01
    python update_binance.py --interval 5m 15m 1h
"""
import argparse, json, os, sys, time, urllib.parse, urllib.request
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://api.binance.com/api/v3/klines"
STEP_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def last_open_ms(path):
    """Open time (ms, UTC) of the last row in the file."""
    with open(path, "rb") as f:
        f.seek(-512, os.SEEK_END)
        line = f.read().decode().strip().splitlines()[-1]
    d, t = line.split(",")[:2]
    ts = dt.datetime.strptime(d + t, "%Y%m%d%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)


def fetch(symbol, interval, start_ms):
    """Page through /api/v3/klines from start_ms; return closed candles only."""
    now_ms = int(time.time() * 1000)
    rows, cursor = [], start_ms
    while True:
        q = urllib.parse.urlencode({"symbol": symbol, "interval": interval,
                                    "startTime": cursor, "limit": 1000})
        with urllib.request.urlopen(f"{BASE}?{q}", timeout=20) as r:
            kl = json.load(r)
        if not kl:
            break
        rows.extend(k for k in kl if k[6] < now_ms)   # close_time already passed
        cursor = kl[-1][0] + 1
        if len(kl) < 1000:
            break
        time.sleep(0.2)
    return rows


def to_line(k):
    t = dt.datetime.fromtimestamp(k[0] / 1000, tz=dt.timezone.utc)
    n = int(k[8])
    o, h, l, c = (repr(float(x)) for x in k[1:5])
    return f"{t:%Y%m%d},{t:%H:%M:%S},{o},{h},{l},{c},{n},{n},0\n"


def update(symbol, interval, start):
    path = os.path.join(HERE, f"{symbol}_{interval}.csv")
    if os.path.exists(path):
        last = last_open_ms(path)
    else:   # pretend the "last candle" sits right before start
        t0 = dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        last = int(t0.timestamp() * 1000) - STEP_MS[interval]
    rows = fetch(symbol, interval, last + STEP_MS[interval])
    rows = [k for k in rows if k[0] > last]
    gaps = sum(1 for a, b in zip(rows, rows[1:]) if b[0] - a[0] != STEP_MS[interval])
    if rows and rows[0][0] - last != STEP_MS[interval]:
        gaps += 1
    with open(path, "a", newline="") as f:
        f.writelines(to_line(k) for k in rows)
    fmt = lambda ms: dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    end = fmt(rows[-1][0]) if rows else fmt(last)
    print(f"{symbol} {interval}: +{len(rows):,} candles, {fmt(last)} -> {end} UTC, gaps: {gaps}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", choices=list(STEP_MS), nargs="*", default=["5m"])
    ap.add_argument("--start", default="2021-08-01", help="YYYY-MM-DD, used only when the file does not exist")
    a = ap.parse_args()
    for iv in a.interval:
        update(a.symbol.upper(), iv, a.start)
