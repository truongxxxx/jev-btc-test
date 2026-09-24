# Candidate indicator sets + Pass 1 (gradient boosting)

Selected on validation 2024-07-01 → 2024-12-31. Tested on 2025-01-01 → 2026-09-24 (181,290 candles), scored once.

## Pass 1: boosting per candidate

| Candidate | Indicators | Valid AUC | Test AUC | Test accuracy | Accuracy, top 10% most confident |
|---|---|---|---|---|---|
| FULL_51 | 51 | 0.5301 | 0.5269 | 0.5184 | 0.556 |
| A_top12 | 12 | 0.528 | 0.5264 | 0.5172 | 0.5573 |
| B_top6 | 6 | 0.5253 | 0.524 | 0.5154 | 0.5477 |
| C_classic | 12 | 0.5203 | 0.5228 | 0.5142 | 0.5459 |
| D_trend | 8 | 0.522 | 0.5223 | 0.5156 | 0.5482 |

## Candidates

- **A_top12**: `m15_dev_ret_atr`, `h1_dev_ret_atr`, `dist_ema9_atr`, `m15_ret1_atr`, `streak`, `ret3_atr`, `dow`, `dist_ema200_atr`, `m15_rsi14`, `h1_ret1_atr`, `hour_cos`, `rsi14`
- **B_top6**: `m15_dev_ret_atr`, `h1_dev_ret_atr`, `dist_ema9_atr`, `m15_ret1_atr`, `streak`, `ret3_atr`
- **C_classic**: `ema_stack`, `dist_ema21_atr`, `slope_ema21_atr`, `rsi14`, `rsi14_chg3`, `macd_hist_atr`, `macd_hist_chg`, `m15_ema_stack`, `m15_rsi14`, `m15_macd_hist_atr`, `h1_ema_stack`, `h1_rsi14`
- **D_trend**: `dist_ema9_atr`, `dist_ema21_atr`, `dist_ema50_atr`, `dist_ema200_atr`, `slope_ema9_atr`, `slope_ema21_atr`, `slope_ema50_atr`, `ema_stack`

## Indicator ranking (permutation importance = AUC lost when shuffled, on validation)

| # | Indicator | Importance |
|---|---|---|
| 1 | `m15_dev_ret_atr` | +0.00581 |
| 2 | `h1_dev_ret_atr` | +0.00323 |
| 3 | `dist_ema9_atr` | +0.00252 |
| 4 | `m15_ret1_atr` | +0.00158 |
| 5 | `streak` | +0.00130 |
| 6 | `ret3_atr` | +0.00130 |
| 7 | `dow` | +0.00119 |
| 8 | `dist_ema200_atr` | +0.00112 |
| 9 | `m15_rsi14` | +0.00103 |
| 10 | `h1_ret1_atr` | +0.00090 |
| 11 | `hour_cos` | +0.00081 |
| 12 | `rsi14` | +0.00081 |
| 13 | `m15_slope_ema21_atr` | +0.00081 |
| 14 | `upper_wick` | +0.00080 |
| 15 | `m15_dist_ema21_atr` | +0.00078 |
| 16 | `lower_wick` | +0.00077 |
| 17 | `h1_macd_hist_chg` | +0.00074 |
| 18 | `m15_macd_hist_atr` | +0.00070 |
| 19 | `macd_hist_atr` | +0.00068 |
| 20 | `hour` | +0.00064 |
| 21 | `bb_pos` | +0.00051 |
| 22 | `hour_sin` | +0.00048 |
| 23 | `h1_dist_ema21_atr` | +0.00044 |
| 24 | `macd_hist_chg` | +0.00042 |
| 25 | `slope_ema9_atr` | +0.00041 |
| 26 | `pos_in_range_20` | +0.00037 |
| 27 | `h1_atr_pct` | +0.00033 |
| 28 | `range_atr` | +0.00033 |
| 29 | `h1_rsi14` | +0.00031 |
| 30 | `h1_macd_hist_atr` | +0.00030 |
| 31 | `m15_macd_hist_chg` | +0.00025 |
| 32 | `ret1_atr` | +0.00020 |
| 33 | `ret12_atr` | +0.00020 |
| 34 | `dist_ema50_atr` | +0.00019 |
| 35 | `dist_low50_atr` | +0.00012 |
| 36 | `slope_ema50_atr` | +0.00010 |
| 37 | `rsi14_chg3` | +0.00009 |
| 38 | `h1_slope_ema21_atr` | +0.00004 |
| 39 | `h1_ema_stack` | +0.00003 |
| 40 | `dist_high50_atr` | +0.00002 |
| 41 | `m15_ema_stack` | +0.00002 |
| 42 | `activity_ratio` | +0.00001 |
| 43 | `ema_stack` | -0.00000 |
| 44 | `activity_ratio_288` | -0.00005 |
| 45 | `dist_ema21_atr` | -0.00006 |
| 46 | `m15_atr_pct` | -0.00007 |
| 47 | `atr_pct` | -0.00013 |
| 48 | `slope_ema21_atr` | -0.00019 |
| 49 | `pos_in_range_50` | -0.00019 |
| 50 | `bb_width_pctile` | -0.00039 |
| 51 | `body_ratio` | -0.00076 |

## Validation AUC using a single group

| Group | Valid AUC |
|---|---|
| trend | 0.5220 |
| tf_m15 | 0.5212 |
| tf_h1 | 0.5203 |
| volatility | 0.5190 |
| momentum | 0.5188 |
| position | 0.5178 |
| candle_shape | 0.5140 |
| time | 0.5045 |
| activity | 0.5021 |
