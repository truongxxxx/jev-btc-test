# Baselines: will the next BTCUSDT 5-minute candle close up or down?

- Data: `data/BTCUSDT_5m.csv` (Binance). Label: next candle closes above its open; dojis dropped.
- Train: 2022-01-01 → 2024-12-31 (314,552 candles). Test: 2025-01-01 → 2026-09-24 07:55 UTC (181,290 candles).
- Share of up candles: train 0.501, test 0.499.
- 51 indicators in 9 groups. No Jev calls.

## 1. Results on the test period

| Method | Accuracy | AUC | AUC 95% CI | Brier skill |
|---|---|---|---|---|
| Always predict the base rate | 0.4992 | 0.5000 | 0.5000–0.5000 | -0.00001 |
| Momentum (last candle up → up) | 0.4945 | 0.4945 | 0.4921–0.4966 | -0.00084 |
| Reversal (last candle up → down) | 0.5055 | 0.5055 | 0.5031–0.5081 | +0.00004 |
| Logistic regression (all indicators) | 0.5153 | 0.5220 | 0.5196–0.5246 | +0.00030 |
| Gradient boosting (all indicators) | 0.5185 | 0.5273 | 0.5244–0.5301 | -0.00014 |

Boosting: 600 rounds (chosen on 2024-H2, valid AUC 0.5296).

## 2. Boosting by test sub-period

| Period | Candles | Accuracy | AUC |
|---|---|---|---|
| 2025 H1 | 52,024 | 0.5209 | 0.5304 |
| 2025 H2 | 52,798 | 0.5168 | 0.5260 |
| 2026 to date | 76,468 | 0.5181 | 0.5261 |

## 3. When boosting is most confident

Only candles whose prediction is furthest from 0.5. Gross P&L = mean return of the next candle in the predicted direction, **before fees**.

| Subset | Candles | Accuracy | Gross P&L / trade (bps) |
|---|---|---|---|
| All | 181,290 | 0.5185 | +0.16 |
| Top 10% most confident | 18,129 | 0.5609 | +0.87 |
| Top 1% most confident | 1,813 | 0.5896 | +1.85 |

For scale: the mean absolute move of a 5m candle in the test period is 8.8 bps. Binance spot taker fees for a round trip are ≈ 20 bps (15 bps when paid in BNB).

## 4. Drop one indicator group (boosting refit)

Exploratory only: scored on the test period, so it was NOT used to choose candidates (see `candidates.md`, which selects on the validation period). A large drop = the group carries signal; ~0 or negative = redundant.

| Group dropped | Columns | Remaining AUC | AUC drop |
|---|---|---|---|
| trend | 8 | 0.5264 | +0.0009 |
| momentum | 7 | 0.5271 | +0.0001 |
| volatility | 4 | 0.5264 | +0.0009 |
| candle_shape | 4 | 0.5254 | +0.0019 |
| position | 4 | 0.5276 | -0.0003 |
| activity | 2 | 0.5270 | +0.0002 |
| time | 4 | 0.5266 | +0.0007 |
| tf_m15 | 9 | 0.5260 | +0.0013 |
| tf_h1 | 9 | 0.5271 | +0.0002 |

## 5. One indicator group on its own

| Only group | AUC |
|---|---|
| trend | 0.5196 |
| momentum | 0.5213 |
| volatility | 0.5201 |
| candle_shape | 0.5129 |
| position | 0.5175 |
| activity | 0.4999 |
| time | 0.5016 |
| tf_m15 | 0.5222 |
| tf_h1 | 0.5180 |
