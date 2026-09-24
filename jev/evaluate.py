# -*- coding: utf-8 -*-
"""
Compare pass 1 (gradient boosting, candidate A_top12) with pass 2 (Jev given the same
12 indicators).

    2a  Jev on its own
    2b  Jev as a filter: keep only candles where boosting and Jev agree
    2c  blend the boosting and Jev probabilities (weights fitted on the 'valid' samples, 2024-07..12)

The main comparison is on 'test_rand' (1,500 random candles, 2025 -> 2026-09). Jev's
direction comes from p = p_up / (p_up + p_down), thresholded at its median on 'valid'
(fixed beforehand, never looks at test).

    python evaluate.py      -> report_jev.md
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(11)
B = 1000


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load():
    ds = pd.DataFrame([json.loads(l) for l in (HERE / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l])
    ind = pd.DataFrame(list(ds["state_num"].map(lambda s: s["indicators"])))
    ds = pd.concat([ds.drop(columns=["state_num", "state_text"]), ind], axis=1).set_index("id")
    for v in ("num", "text"):
        rows = [json.loads(l) for l in (HERE / f"results_{v}.jsonl").read_text(encoding="utf-8").splitlines() if l]
        r = pd.DataFrame([x for x in rows if "error" not in x]).drop_duplicates("id", keep="last").set_index("id")
        ds[f"{v}_up"], ds[f"{v}_down"] = r["p_next_up"], r["p_next_down"]
        ds[f"{v}_crsi"], ds[f"{v}_cema"] = r["p_ctrl_rsi"], r["p_ctrl_ema9"]
        ds[f"{v}_p"] = ds[f"{v}_up"] / (ds[f"{v}_up"] + ds[f"{v}_down"])
        ds.attrs["model"] = r["model"].iloc[0]
    return ds


def boot_auc(y, p, q=None):
    """AUC + CI; with q also the CI of AUC(p) - AUC(q) (paired bootstrap)."""
    n = len(y); a, d = [], []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i]))
        if q is not None:
            d.append(a[-1] - roc_auc_score(y[i], q[i]))
    out = (roc_auc_score(y, p), *np.percentile(a, [2.5, 97.5]))
    return out + (tuple(np.percentile(d, [2.5, 97.5])) if q is not None else ())


def main():
    ds = load()
    va, te, tc = (ds[ds.split == s] for s in ("valid", "test_rand", "test_conf"))
    L = []; P = L.append
    ok = te.dropna(subset=["num_p", "text_p"])
    y, pb = ok["y"].to_numpy(), ok["p_boost"].to_numpy()

    P("# Pass 1 (gradient boosting) vs pass 2 (Jev), candidate A_top12\n")
    P(f"- Samples with complete results: valid {va.num_p.notna().sum()}/{len(va)}, test_rand {len(ok)}/{len(te)}, "
      f"test_conf {tc.num_p.notna().sum()}/{len(tc)}. Model: {ds.attrs['model']}.")
    P(f"- Share of up candles in test_rand: {y.mean():.3f}.\n")

    P("## 0. Does Jev read the state correctly?\n")
    P("| Format | Control RSI>50 | Control close>EMA9 | p_up + p_down (mean) | p_up: mean / std |")
    P("|---|---|---|---|---|")
    for v in ("num", "text"):
        d = ds.dropna(subset=[f"{v}_p"])
        cr = ((d[f"{v}_crsi"] > .5) == d.ctrl_rsi_above_50).mean()
        ce = ((d[f"{v}_cema"] > .5) == d.ctrl_close_above_ema9).mean()
        s = d[f"{v}_up"] + d[f"{v}_down"]
        P(f"| {v} | {cr:.1%} | {ce:.1%} | {s.mean():.3f} | {d[f'{v}_up'].mean():.3f} / {d[f'{v}_up'].std():.3f} |")

    P("\n## 2a. Jev on its own vs boosting (test_rand)\n")
    P("| Method | Accuracy | AUC | AUC 95% CI | AUC difference vs boosting (95% CI) |")
    P("|---|---|---|---|---|")
    a = boot_auc(y, pb)
    P(f"| **Pass 1: boosting** | {((pb > .5) == y).mean():.4f} | {a[0]:.4f} | {a[1]:.4f}–{a[2]:.4f} | — |")
    thr = {}
    for v in ("num", "text"):
        thr[v] = va[f"{v}_p"].median()
        pj = ok[f"{v}_p"].to_numpy()
        a = boot_auc(y, pj, pb)
        P(f"| Pass 2a: Jev ({v}) | {((pj > thr[v]) == y).mean():.4f} | {a[0]:.4f} | {a[1]:.4f}–{a[2]:.4f} | "
          f"{a[0]-roc_auc_score(y, pb):+.4f} ({a[3]:+.4f}…{a[4]:+.4f}) |")
    P(f"\nJev direction threshold (median on valid): num {thr['num']:.3f}, text {thr['text']:.3f}. "
      f"On 1,500 samples the boosting AUC alone has an error of about ±0.03, so only differences "
      f"outside the confidence interval count.\n")

    P("### Which way does Jev read the indicators?\n")
    P("Rank correlation (Spearman) between p_up and each indicator on test_rand, with the true "
      "label and boosting for reference. Negative = after price rises, the next candle leans "
      "down (reversal).\n")
    P("| Indicator | True label | Boosting | Jev num | Jev text |")
    P("|---|---|---|---|---|")
    for c in ["dist_close_to_ema9_atr5", "m15_current_candle_move_atr5", "h1_current_candle_move_atr5",
              "move_last_3_candles_atr5", "same_color_streak", "rsi14_m5", "rsi14_m15"]:
        x = ok[c].astype(float)
        P(f"| `{c}` | {x.corr(ok.y, method='spearman'):+.3f} | {x.corr(ok.p_boost, method='spearman'):+.3f} | "
          f"{x.corr(ok.num_p, method='spearman'):+.3f} | {x.corr(ok.text_p, method='spearman'):+.3f} |")
    P(f"\nCorrelation between boosting and Jev: num {ok.p_boost.corr(ok.num_p, method='spearman'):+.3f}, "
      f"text {ok.p_boost.corr(ok.text_p, method='spearman'):+.3f}.\n")

    P("## 2b. Jev as a filter on boosting\n")
    P("Keep a candle when both agree on direction. Fair control: boosting filters **the same number "
      "of candles** by its own confidence. Jev only helps if the 'agree' row beats the "
      "'boosting self-filter' row.\n")
    P("| Subset | Format | Candles | Boosting accuracy | Gross P&L / trade (bps) |")
    P("|---|---|---|---|---|")
    side_b = np.where(pb > .5, 1, -1); ret = ok["next_ret_bps"].to_numpy()
    P(f"| All of test_rand | — | {len(ok)} | {((pb > .5) == y).mean():.4f} | {(side_b*ret).mean():+.2f} |")
    for v in ("num", "text"):
        side_j = np.where(ok[f"{v}_p"].to_numpy() > thr[v], 1, -1)
        agree = side_j == side_b
        k = agree.sum()
        top = np.argsort(-np.abs(pb - .5))[:k]
        P(f"| Jev agrees | {v} | {k} | {((pb[agree] > .5) == y[agree]).mean():.4f} | "
          f"{(side_b[agree]*ret[agree]).mean():+.2f} |")
        P(f"| Boosting self-filter, same count | {v} | {k} | {((pb[top] > .5) == y[top]).mean():.4f} | "
          f"{(side_b[top]*ret[top]).mean():+.2f} |")
        P(f"| Jev disagrees | {v} | {len(ok)-k} | {((pb[~agree] > .5) == y[~agree]).mean():.4f} | "
          f"{(side_b[~agree]*ret[~agree]).mean():+.2f} |")
    c = tc.dropna(subset=["num_p", "text_p"])
    yc, pc = c.y.to_numpy(), c.p_boost.to_numpy(); sc = np.where(pc > .5, 1, -1)
    P(f"\n**Within the 300 candles where boosting is most confident (test_conf)**: boosting accuracy "
      f"{((pc > .5) == yc).mean():.4f}.\n")
    P("| Format | Jev agrees: candles / accuracy | Jev disagrees: candles / accuracy |")
    P("|---|---|---|")
    for v in ("num", "text"):
        ag = np.where(c[f"{v}_p"].to_numpy() > thr[v], 1, -1) == sc
        P(f"| {v} | {ag.sum()} / {((pc[ag] > .5) == yc[ag]).mean():.4f} | "
          f"{(~ag).sum()} / {((pc[~ag] > .5) == yc[~ag]).mean():.4f} |")

    P("\n## 2c. Blending boosting + Jev probabilities\n")
    P("Logistic regression on logit(p_boost), logit(p_jev), fitted on the 500 valid samples, "
      "applied to test_rand.\n")
    P("| Format | Boosting coef | Jev coef | Blended AUC | Difference vs boosting (95% CI) |")
    P("|---|---|---|---|---|")
    for v in ("num", "text"):
        vv = va.dropna(subset=[f"{v}_p"])
        Xv = np.c_[logit(vv.p_boost), logit(vv[f"{v}_p"])]
        lr = LogisticRegression(C=1.0).fit(Xv, vv.y)
        pm = lr.predict_proba(np.c_[logit(pb), logit(ok[f"{v}_p"])])[:, 1]
        a = boot_auc(y, pm, pb)
        P(f"| {v} | {lr.coef_[0][0]:+.3f} | {lr.coef_[0][1]:+.3f} | {a[0]:.4f} | "
          f"{a[0]-roc_auc_score(y, pb):+.4f} ({a[3]:+.4f}…{a[4]:+.4f}) |")
    P("\nA Jev coefficient near 0 means that, on the validation samples, adding Jev did not help boosting.")

    (HERE / "report_jev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
