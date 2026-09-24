# -*- coding: utf-8 -*-
"""
Build the set of samples to ask Jev about, for candidate A_top12.

One sample = one 5m candle that has just closed. The state holds EXACTLY the 12
indicators of A_top12 (the same information boosting used in pass 1), in two formats:
    state_num   numbers + a definition for each field
    state_text  neutral plain-English description (hints at no rule)
Blind: no symbol, no date, no absolute price.

Three sample groups:
    valid      500 candles, 2024-07..12   -> fits the boosting+Jev blend weights (2c)
    test_rand  1,500 random candles, 2025 -> 2026-09 -> main comparison
    test_conf  300 candles from boosting's 10% most confident (disjoint from test_rand)

    python make_dataset.py        -> dataset.jsonl
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "baseline")
sys.path.insert(0, BASE)

CAND = "A_top12"
N_VALID, N_TEST, N_CONF = 500, 1500, 300
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def session(h):
    return "Asia" if h < 7 else "Europe" if h < 13 else "US" if h < 21 else "late US / pre-Asia"


def rsi_state(v):
    return ("overbought" if v >= 70 else "high" if v >= 60 else "neutral" if v > 40
            else "low" if v > 30 else "oversold")


def move_word(x):
    a = abs(x)
    size = ("barely moved" if a < 0.3 else "moved moderately" if a < 1 else
            "moved strongly" if a < 2 else "moved very strongly")
    return size if a < 0.3 else f"{size} {'up' if x > 0 else 'down'}"


def side_word(x):
    a = abs(x)
    how = "right at" if a < 0.2 else "slightly" if a < 1 else "clearly" if a < 2.5 else "far"
    return "right at" if a < 0.2 else f"{how} {'above' if x > 0 else 'below'}"


def r(x, n=2):
    return round(float(x), n)


def state_num(row, t_next):
    m15_done = (t_next.minute % 15) // 5 or 3
    h1_done = t_next.minute // 5 or 12
    return {
        "task_context": "Snapshot taken at the close of a 5-minute candle of a liquid 24/7 market.",
        "indicators": {
            "m15_current_candle_move_atr5": r(row.m15_dev_ret_atr),
            "m15_current_candle_bars_done": f"{m15_done}/3",
            "h1_current_candle_move_atr5": r(row.h1_dev_ret_atr),
            "h1_current_candle_bars_done": f"{h1_done}/12",
            "dist_close_to_ema9_atr5": r(row.dist_ema9_atr),
            "dist_close_to_ema200_atr5": r(row.dist_ema200_atr),
            "last_closed_m15_candle_move_atr15": r(row.m15_ret1_atr),
            "last_closed_h1_candle_move_atr60": r(row.h1_ret1_atr),
            "move_last_3_candles_atr5": r(row.ret3_atr),
            "same_color_streak": int(row.streak),
            "rsi14_m5": r(row.rsi14, 1),
            "rsi14_m15": r(row.m15_rsi14, 1),
            "next_candle_weekday": WEEKDAYS[t_next.dayofweek],
            "next_candle_hour_utc": t_next.hour,
        },
        "definitions": {
            "atr5 / atr15 / atr60": "14-period ATR of the 5m / 15m / 1h chart; moves are (price change) / ATR",
            "*_current_candle_move": "close now minus open of the higher-timeframe candle still forming",
            "dist_close_to_emaN": "(close - EMA N on 5m) / atr5; positive = close above the EMA",
            "same_color_streak": "consecutive 5m candles of the same color; +3 = three up candles, -2 = two down",
        },
    }


def state_text(row, t_next):
    m15_done = (t_next.minute % 15) // 5 or 3
    h1_done = t_next.minute // 5 or 12
    s = int(row.streak)
    streak = (f"The last {abs(s)} five-minute candles all closed {'up' if s > 0 else 'down'}."
              if abs(s) > 1 else
              f"The last five-minute candle closed {'up' if s > 0 else 'down' if s < 0 else 'flat'} "
              f"and the one before closed the other way.")
    lines = [
        f"A 5-minute candle has just closed. The next candle starts on {WEEKDAYS[t_next.dayofweek]} "
        f"at {t_next:%H:%M} UTC ({session(t_next.hour)} session).",
        f"Current 15-minute candle ({m15_done} of 3 bars done): has {move_word(row.m15_dev_ret_atr)} "
        f"({row.m15_dev_ret_atr:+.2f} ATR of the 5m chart) since its open.",
        f"Current 1-hour candle ({h1_done} of 12 bars done): has {move_word(row.h1_dev_ret_atr)} "
        f"({row.h1_dev_ret_atr:+.2f} ATR of the 5m chart) since its open.",
        f"The last completed 15-minute candle {move_word(row.m15_ret1_atr)} "
        f"({row.m15_ret1_atr:+.2f} ATR of the 15m chart).",
        f"The last completed 1-hour candle {move_word(row.h1_ret1_atr)} "
        f"({row.h1_ret1_atr:+.2f} ATR of the 1h chart).",
        f"Over the last 3 five-minute candles price {move_word(row.ret3_atr)} "
        f"({row.ret3_atr:+.2f} ATR).",
        streak,
        f"Close is {side_word(row.dist_ema9_atr)} the 9-period EMA ({row.dist_ema9_atr:+.2f} ATR) "
        f"and {side_word(row.dist_ema200_atr)} the 200-period EMA ({row.dist_ema200_atr:+.2f} ATR) "
        f"on the 5-minute chart.",
        f"RSI(14) is {row.rsi14:.1f} on the 5-minute chart ({rsi_state(row.rsi14)}) and "
        f"{row.m15_rsi14:.1f} on the 15-minute chart ({rsi_state(row.m15_rsi14)}).",
    ]
    return {"market_snapshot": " ".join(lines)}


def main():
    cols = json.load(open(os.path.join(BASE, "candidates.json"), encoding="utf-8"))[CAND]
    f = pd.read_parquet(os.path.join(BASE, "features_m5.parquet"))
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["y"])
    run1 = pd.read_parquet(os.path.join(BASE, "run1_boosting.parquet"))

    # boosting on the validation period: fit on 2022-01..2024-06 (as during candidate selection)
    fit = f.loc["2022-01-01":"2024-06-30 23:59"]
    va = f.loc["2024-07-01":"2024-12-31 23:59"]
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=105, max_leaf_nodes=31,
                                       min_samples_leaf=500, l2_regularization=1.0,
                                       early_stopping=False, random_state=0).fit(fit[cols], fit["y"])
    p_va = pd.Series(m.predict_proba(va[cols])[:, 1], index=va.index)

    rng = np.random.default_rng(2026)
    te_idx = run1.index
    rand = pd.Index(rng.choice(te_idx, N_TEST, replace=False)).sort_values()
    conf = np.abs(run1[CAND] - 0.5)
    pool = te_idx[(conf >= conf.quantile(0.9)).to_numpy()].difference(rand)
    conf_pick = pd.Index(rng.choice(pool, N_CONF, replace=False)).sort_values()
    va_pick = pd.Index(rng.choice(va.index, N_VALID, replace=False)).sort_values()

    out = os.path.join(HERE, "dataset.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for split, idx, pser in (("valid", va_pick, p_va), ("test_rand", rand, run1[CAND]),
                                 ("test_conf", conf_pick, run1[CAND])):
            for t in idx:
                row = f.loc[t]
                t_next = t + pd.Timedelta("5min")
                rec = {
                    "id": f"{split}-{t:%Y%m%d%H%M}", "split": split, "t": f"{t:%Y-%m-%d %H:%M}",
                    "y": int(row.y), "next_ret_bps": r(row.next_ret_bps, 3),
                    "p_boost": r(pser.loc[t], 5),
                    "ctrl_rsi_above_50": int(row.rsi14 > 50),
                    "ctrl_close_above_ema9": int(row.dist_ema9_atr > 0),
                    "state_num": state_num(row, t_next),
                    "state_text": state_text(row, t_next),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} samples -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
