# jev-btc-test

**Can Jev predict the next BTC 5-minute candle?**

[Jev](https://typesafe.ai) (TypeSafe AI's System One model) is increasingly pitched as something
you can plug into a trading bot. This repo tests that claim on one concrete, measurable task,
and compares Jev with a plain gradient-boosting model that receives **exactly the same inputs**.

**Short answer: no.** Jev reads the indicators correctly, but it interprets them in the opposite
direction to how BTC 5-minute candles actually behave. Adding Jev to the boosting model, as a
replacement, a filter, or a blend, did not improve it.

## The test

Task: when a BTCUSDT 5-minute candle closes, predict whether the **next** candle closes above its open.

1. **Indicators.** 51 technical indicators on 5m, plus 15m and 1h rebuilt from 5m data using
   only closed candles (no look-ahead): EMA 9/21/50/200, RSI, MACD, ATR, Bollinger, candle shape,
   same-color streaks, position in range, trade count, time of day.
2. **Candidate selection.** Gradient boosting trained on 2022-01 → 2024-06; indicators ranked by
   permutation importance on a separate validation window (2024-07 → 2024-12). The strongest
   candidate, **A_top12**, keeps 12 indicators.
3. **Pass 1: boosting.** Refit on 2022-2024 with the 12 indicators, scored on the unseen test
   period 2025-01-01 → 2026-09-24 (181,290 candles).
4. **Pass 2: Jev.** For 1,500 random test candles, the same 12 indicators are sent to Jev
   (`jev-1.13.0`), in two formats: numbers with definitions, and a plain-English description.
   Data is blind: no symbol, no dates, no absolute prices. Two control questions whose answers
   are in the state check that Jev reads it correctly.

## Results

### The indicators do carry a (small) signal

| Method (full test period, 181,290 candles) | Accuracy | AUC (95% CI) |
|---|---|---|
| Always predict the base rate | 49.9% | 0.500 |
| Momentum (last candle up → up) | 49.5% | 0.495 |
| Reversal (last candle up → down) | 50.6% | 0.506 |
| Gradient boosting, 51 indicators | 51.9% | 0.527 (0.524–0.530) |
| **Gradient boosting, A_top12 (12 indicators)** | **51.7%** | **0.526 (0.524–0.530)** |

The signal is stable across 2025 H1, 2025 H2 and 2026. It is **not** profitable on Binance: even
the 1% most confident predictions earn ~1.9 bps gross per trade, against ~15–20 bps of round-trip fees.

### Adding Jev does not help

Same 1,500 test candles for every row:

| | Method | Result | vs boosting |
|---|---|---|---|
| Pass 1 | Boosting (A_top12) | AUC 0.522, accuracy 51.6% | baseline |
| 2a | Jev alone, numbers | AUC 0.486, accuracy 48.1% | worse |
| 2a | Jev alone, text | AUC 0.494, accuracy 49.3% | worse |
| 2b | Keep only candles where Jev agrees with boosting | 49.4% (num) / 51.6% (text) | worse than boosting filtering the same number of candles itself (51.7% / 53.0%) |
| 2c | Blend boosting + Jev (weights fitted on validation) | AUC 0.521 / 0.524 | no change; Jev's weight ≈ 0 |

Jev answered the control questions correctly **99.7–99.8%** of the time, so this is not a
parsing problem.

### Why: Jev reads the indicators backwards

Rank correlation between each indicator and the predicted probability of an up candle:

| Indicator | Boosting | Jev (numbers) | Jev (text) |
|---|---|---|---|
| Distance from close to EMA9 | −0.80 | **+0.83** | **+0.72** |
| Move over the last 3 candles | −0.66 | **+0.76** | **+0.70** |
| Same-color streak | −0.53 | **+0.65** | **+0.87** |
| RSI(14), 5m | −0.76 | **+0.77** | **+0.51** |

Boosting learned from three years of data that on 5-minute BTC, a stretched move tends to
**revert** slightly. Jev reads the same numbers like a textbook: price above the EMA, high RSI and
a green streak mean **continuation**. On the 300 candles where boosting was most confident
(58% accuracy), Jev disagreed with 294 of them (numeric format).

Full tables: [baseline/report.md](baseline/report.md), [baseline/candidates.md](baseline/candidates.md),
[jev/report_jev.md](jev/report_jev.md).

## What this does and does not show

- One model version (`jev-1.13.0`), one market (BTCUSDT on Binance), one timeframe (5m),
  price-derived inputs only.
- With 1,500 samples, each individual gap between Jev and boosting is within statistical noise;
  the consistent direction across all three ways of adding Jev, and the reversed reading of the
  indicators, are the robust findings.
- It says nothing about tasks where Jev gets information that is not in the price itself
  (news, filings, order flow, prediction-market context).
- The boosting signal itself is too small to trade after fees. Price-only 5-minute prediction is hard.

## Reproduce

Python 3.11+.

```bash
pip install -r requirements.txt

# 1. data: ~530k 5m candles from Binance's public API (a few minutes)
python data/update_binance.py

# 2. baselines + candidate selection + pass 1 (~20 min on 4 cores)
cd baseline
python run_baseline.py      # -> report.md
python candidates.py        # -> candidates.md, candidates.json
cd ..

# 3. pass 2: needs a TypeSafe API key (https://console.typesafe.ai/keys)
#    put TYPESAFE_API_KEY=... in a .env file at the repo root, or export it
cd jev
python make_dataset.py              # rebuilds dataset.jsonl (also included in the repo)
python run_jev.py --variant num --dry-run
python run_jev.py --variant num
python run_jev.py --variant text
python evaluate.py                  # -> report_jev.md
```

`baseline/features.py` pins the data end to `2026-09-24 08:00` UTC so the published numbers
reproduce exactly; set `DATA_END = None` to include newer candles. The full Jev run is
2 × 2,300 requests, ~3.6M input tokens, about **$0.19** at $0.042 per million tokens.
Raw Jev responses are not committed; `run_jev.py` produces your own.

## Layout

```
data/update_binance.py     download / extend BTCUSDT candles
baseline/features.py       the 51 indicators and the label
baseline/run_baseline.py   trivial baselines, logistic regression, boosting, ablations
baseline/candidates.py     candidate indicator sets + pass 1
jev/make_dataset.py        samples and the two state formats sent to Jev
jev/run_jev.py             calls Jev (resumable)
jev/evaluate.py            pass 2a / 2b / 2c vs pass 1
```

## License

MIT, see [LICENSE](LICENSE). Market data comes from Binance's public API and is not
redistributed here.
