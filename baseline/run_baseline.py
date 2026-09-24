# -*- coding: utf-8 -*-
"""
Baselines for "will the next BTCUSDT 5-minute candle close up or down?" (no Jev calls).

Time-based split:
    train  2022-01-01 .. 2024-12-31   (boosting rounds chosen on 2024-07..12)
    test   2025-01-01 .. end of data

Run:  python run_baseline.py            -> report.md in this folder
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = ("2022-01-01", "2024-12-31 23:59")
VALID = ("2024-07-01", "2024-12-31 23:59")
TEST = ("2025-01-01", None)
RNG = np.random.default_rng(7)


def auc_ci(y, p, reps=200):
    n = len(y)
    s = [roc_auc_score(y[i], p[i]) for i in (RNG.integers(0, n, n) for _ in range(reps))]
    return np.percentile(s, [2.5, 97.5])


def score(y, p):
    acc = ((p > 0.5) == y).mean()
    auc = roc_auc_score(y, p)
    base = y.mean()
    bss = 1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2)
    return acc, auc, bss


def hgb(max_iter):
    return HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=500,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def fit_hgb(Xtr, ytr, Xva, yva, cols):
    """Pick the number of rounds on VALID (time-ordered), then refit on all of TRAIN."""
    m = hgb(600).fit(Xtr[cols].loc[:"2024-06-30 23:59"], ytr.loc[:"2024-06-30 23:59"])
    aucs = [roc_auc_score(yva, p[:, 1]) for p in m.staged_predict_proba(Xva[cols])]
    best = int(np.argmax(aucs)) + 1
    best = max(best, 20)
    return hgb(best).fit(Xtr[cols], ytr), best, max(aucs)


def main():
    t0 = time.time()
    f = F.build()
    f.to_parquet(os.path.join(HERE, "features_m5.parquet"))   # reused by candidates.py and jev/
    cols = [c for g in F.GROUPS.values() for c in g]
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["y"])
    print(f"features: {len(cols)} columns, {len(f):,} valid rows  ({time.time()-t0:.0f}s)")

    tr = f.loc[TRAIN[0]:TRAIN[1]]
    va = f.loc[VALID[0]:VALID[1]]
    te = f.loc[TEST[0]:]
    Xtr, ytr, Xva, yva, Xte, yte = tr, tr["y"], va, va["y"], te, te["y"].to_numpy()
    lines = []
    P = lines.append

    P("# Baselines: will the next BTCUSDT 5-minute candle close up or down?\n")
    P("- Data: `data/BTCUSDT_5m.csv` (Binance). Label: next candle closes above its open; dojis dropped.")
    P(f"- Train: {tr.index[0]:%Y-%m-%d} → {tr.index[-1]:%Y-%m-%d} ({len(tr):,} candles). "
      f"Test: {te.index[0]:%Y-%m-%d} → {te.index[-1]:%Y-%m-%d %H:%M} UTC ({len(te):,} candles).")
    P(f"- Share of up candles: train {ytr.mean():.3f}, test {yte.mean():.3f}.")
    P(f"- {len(cols)} indicators in {len(F.GROUPS)} groups. No Jev calls.\n")

    res = {}
    # 1-3: trivial baselines
    res["Always predict the base rate"] = np.full(len(yte), ytr.mean())
    res["Momentum (last candle up → up)"] = te["prev_up"].to_numpy() * 0.02 + 0.49
    res["Reversal (last candle up → down)"] = (1 - te["prev_up"].to_numpy()) * 0.02 + 0.49
    # 4: logistic regression
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    lr.fit(Xtr[cols], ytr)
    res["Logistic regression (all indicators)"] = lr.predict_proba(Xte[cols])[:, 1]
    # 5: gradient boosting
    m, n_iter, va_auc = fit_hgb(Xtr, ytr, Xva, yva, cols)
    p_hgb = m.predict_proba(Xte[cols])[:, 1]
    res["Gradient boosting (all indicators)"] = p_hgb
    print(f"boosting: {n_iter} rounds, valid AUC {va_auc:.4f}  ({time.time()-t0:.0f}s)")

    P("## 1. Results on the test period\n")
    P("| Method | Accuracy | AUC | AUC 95% CI | Brier skill |")
    P("|---|---|---|---|---|")
    for k, p in res.items():
        acc, auc, bss = score(yte, p)
        ci = auc_ci(yte, p) if "Always" not in k else (0.5, 0.5)
        P(f"| {k} | {acc:.4f} | {auc:.4f} | {ci[0]:.4f}–{ci[1]:.4f} | {bss:+.5f} |")
    P(f"\nBoosting: {n_iter} rounds (chosen on 2024-H2, valid AUC {va_auc:.4f}).\n")

    P("## 2. Boosting by test sub-period\n")
    P("| Period | Candles | Accuracy | AUC |")
    P("|---|---|---|---|")
    per = pd.Series(p_hgb, index=te.index)
    for name, a, b in (("2025 H1", "2025-01", "2025-06"), ("2025 H2", "2025-07", "2025-12"),
                       ("2026 to date", "2026-01", None)):
        s = per.loc[a:b]; yy = te["y"].loc[a:b].to_numpy()
        P(f"| {name} | {len(s):,} | {((s.to_numpy()>0.5)==yy).mean():.4f} | {roc_auc_score(yy, s):.4f} |")

    P("\n## 3. When boosting is most confident\n")
    P("Only candles whose prediction is furthest from 0.5. Gross P&L = mean return of the next "
      "candle in the predicted direction, **before fees**.\n")
    P("| Subset | Candles | Accuracy | Gross P&L / trade (bps) |")
    P("|---|---|---|---|")
    conf = np.abs(p_hgb - 0.5)
    ret = te["next_ret_bps"].to_numpy()
    side = np.where(p_hgb > 0.5, 1, -1)
    for q, name in ((0, "All"), (0.9, "Top 10% most confident"), (0.99, "Top 1% most confident")):
        mask = conf >= np.quantile(conf, q)
        P(f"| {name} | {mask.sum():,} | {((p_hgb[mask]>0.5)==yte[mask]).mean():.4f} | "
          f"{(side[mask]*ret[mask]).mean():+.2f} |")
    P(f"\nFor scale: the mean absolute move of a 5m candle in the test period is "
      f"{np.abs(ret).mean():.1f} bps. Binance spot taker fees for a round trip are ≈ 20 bps "
      f"(15 bps when paid in BNB).\n")

    P("## 4. Drop one indicator group (boosting refit)\n")
    P("Exploratory only: scored on the test period, so it was NOT used to choose candidates "
      "(see `candidates.md`, which selects on the validation period). A large drop = the group "
      "carries signal; ~0 or negative = redundant.\n")
    P("| Group dropped | Columns | Remaining AUC | AUC drop |")
    P("|---|---|---|---|")
    full_auc = roc_auc_score(yte, p_hgb)
    for g, gc in F.GROUPS.items():
        keep = [c for c in cols if c not in gc]
        mg = hgb(n_iter).fit(Xtr[keep], ytr)
        a = roc_auc_score(yte, mg.predict_proba(Xte[keep])[:, 1])
        P(f"| {g} | {len(gc)} | {a:.4f} | {full_auc - a:+.4f} |")
        print(f"  drop {g}: {a:.4f}  ({time.time()-t0:.0f}s)")

    P("\n## 5. One indicator group on its own\n")
    P("| Only group | AUC |")
    P("|---|---|")
    for g, gc in F.GROUPS.items():
        mg = hgb(n_iter).fit(Xtr[gc], ytr)
        P(f"| {g} | {roc_auc_score(yte, mg.predict_proba(Xte[gc])[:, 1]):.4f} |")

    out = os.path.join(HERE, "report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"-> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
