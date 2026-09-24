# Pass 1 (gradient boosting) vs pass 2 (Jev), candidate A_top12

- Samples with complete results: valid 500/500, test_rand 1500/1500, test_conf 300/300. Model: jev-1.13.0.
- Share of up candles in test_rand: 0.514.

## 0. Does Jev read the state correctly?

| Format | Control RSI>50 | Control close>EMA9 | p_up + p_down (mean) | p_up: mean / std |
|---|---|---|---|---|
| num | 99.8% | 99.8% | 0.857 | 0.410 / 0.059 |
| text | 99.8% | 99.7% | 0.838 | 0.404 / 0.071 |

## 2a. Jev on its own vs boosting (test_rand)

| Method | Accuracy | AUC | AUC 95% CI | AUC difference vs boosting (95% CI) |
|---|---|---|---|---|
| **Pass 1: boosting** | 0.5160 | 0.5219 | 0.4933–0.5504 | — |
| Pass 2a: Jev (num) | 0.4813 | 0.4856 | 0.4593–0.5155 | -0.0363 (-0.0875…+0.0157) |
| Pass 2a: Jev (text) | 0.4933 | 0.4941 | 0.4629–0.5238 | -0.0278 (-0.0775…+0.0226) |

Jev direction threshold (median on valid): num 0.483, text 0.500. On 1,500 samples the boosting AUC alone has an error of about ±0.03, so only differences outside the confidence interval count.

### Which way does Jev read the indicators?

Rank correlation (Spearman) between p_up and each indicator on test_rand, with the true label and boosting for reference. Negative = after price rises, the next candle leans down (reversal).

| Indicator | True label | Boosting | Jev num | Jev text |
|---|---|---|---|---|
| `dist_close_to_ema9_atr5` | -0.020 | -0.800 | +0.829 | +0.722 |
| `m15_current_candle_move_atr5` | +0.005 | -0.519 | +0.691 | +0.751 |
| `h1_current_candle_move_atr5` | -0.020 | -0.584 | +0.647 | +0.574 |
| `move_last_3_candles_atr5` | -0.007 | -0.655 | +0.755 | +0.695 |
| `same_color_streak` | -0.013 | -0.528 | +0.645 | +0.868 |
| `rsi14_m5` | -0.021 | -0.758 | +0.766 | +0.505 |
| `rsi14_m15` | -0.014 | -0.458 | +0.504 | +0.150 |

Correlation between boosting and Jev: num -0.714, text -0.634.

## 2b. Jev as a filter on boosting

Keep a candle when both agree on direction. Fair control: boosting filters **the same number of candles** by its own confidence. Jev only helps if the 'agree' row beats the 'boosting self-filter' row.

| Subset | Format | Candles | Boosting accuracy | Gross P&L / trade (bps) |
|---|---|---|---|---|
| All of test_rand | — | 1500 | 0.5160 | -0.09 |
| Jev agrees | num | 360 | 0.4944 | -0.66 |
| Boosting self-filter, same count | num | 360 | 0.5167 | -0.44 |
| Jev disagrees | num | 1140 | 0.5228 | +0.09 |
| Jev agrees | text | 440 | 0.5159 | -0.54 |
| Boosting self-filter, same count | text | 440 | 0.5295 | -0.06 |
| Jev disagrees | text | 1060 | 0.5160 | +0.09 |

**Within the 300 candles where boosting is most confident (test_conf)**: boosting accuracy 0.5800.

| Format | Jev agrees: candles / accuracy | Jev disagrees: candles / accuracy |
|---|---|---|
| num | 6 / 0.8333 | 294 / 0.5748 |
| text | 17 / 0.5882 | 283 / 0.5795 |

## 2c. Blending boosting + Jev probabilities

Logistic regression on logit(p_boost), logit(p_jev), fitted on the 500 valid samples, applied to test_rand.

| Format | Boosting coef | Jev coef | Blended AUC | Difference vs boosting (95% CI) |
|---|---|---|---|---|
| num | +0.798 | -0.088 | 0.5209 | -0.0010 (-0.0044…+0.0023) |
| text | +1.014 | +0.163 | 0.5236 | +0.0017 (-0.0067…+0.0095) |

A Jev coefficient near 0 means that, on the validation samples, adding Jev did not help boosting.
