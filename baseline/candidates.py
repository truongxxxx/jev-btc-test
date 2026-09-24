# -*- coding: utf-8 -*-
"""
Pick CANDIDATE indicator sets (train + validation only), then run PASS 1:
gradient boosting with each candidate, scored on the test period.

    fit    2022-01-01 .. 2024-06-30
    valid  2024-07-01 .. 2024-12-31   <- every choice (rounds, indicators) looks only here
    refit  2022-01-01 .. 2024-12-31
    test   2025-01-01 .. end          <- scored once, after the candidates are fixed

Writes: candidates.json (the candidate sets) + run1_boosting.parquet (p_up for every
test candle) + candidates.md (report).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
FIT_END, VA_START, TR_START, TR_END, TE_START = (
    "2024-06-30 23:59", "2024-07-01", "2022-01-01", "2024-12-31 23:59", "2025-01-01")

# candidate C: the classic chart-reader set, easy to put into words
CLASSIC = ["ema_stack", "dist_ema21_atr", "slope_ema21_atr", "rsi14", "rsi14_chg3",
           "macd_hist_atr", "macd_hist_chg", "m15_ema_stack", "m15_rsi14", "m15_macd_hist_atr",
           "h1_ema_stack", "h1_rsi14"]


def hgb(n):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=n, max_leaf_nodes=31, min_samples_leaf=500,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def fit_select(fit, va, cols, cap=1500):
    """Fit on `fit`, pick the number of rounds by valid AUC. Returns (model, rounds, valid AUC)."""
    m = hgb(cap).fit(fit[cols], fit["y"])
    aucs = [roc_auc_score(va["y"], p[:, 1]) for p in m.staged_predict_proba(va[cols])]
    k = max(int(np.argmax(aucs)) + 1, 20)
    return hgb(k).fit(fit[cols], fit["y"]), k, max(aucs)


def main():
    t0 = time.time()
    f = F.build()
    cols = [c for g in F.GROUPS.values() for c in g]
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["y"])
    fit, va = f.loc[TR_START:FIT_END], f.loc[VA_START:TR_END]
    tr, te = f.loc[TR_START:TR_END], f.loc[TE_START:]
    log = lambda s: print(f"{s}  ({time.time()-t0:.0f}s)", flush=True)

    # --- rank every indicator on the validation period
    m_all, k_all, a_all = fit_select(fit, va, cols)
    log(f"all {len(cols)} indicators: {k_all} rounds, valid AUC {a_all:.4f}")
    sub = va.sample(min(len(va), 50_000), random_state=1)
    pi = permutation_importance(m_all, sub[cols], sub["y"], scoring="roc_auc",
                                n_repeats=5, random_state=2, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    log("permutation importance done")

    # --- best single group on the validation period
    grp_auc = {}
    for g, gc in F.GROUPS.items():
        grp_auc[g] = fit_select(fit, va, gc, cap=600)[2]
    best_grp = max(grp_auc, key=grp_auc.get)
    log(f"best single group: {best_grp} {grp_auc[best_grp]:.4f}")

    cands = {
        "A_top12": list(imp.index[:12]),
        "B_top6": list(imp.index[:6]),
        "C_classic": CLASSIC,
        f"D_{best_grp}": F.GROUPS[best_grp],
    }

    # --- PASS 1: boosting per candidate, scored on test
    rows, preds = [], pd.DataFrame(index=te.index)
    preds["y"] = te["y"]
    preds["next_ret_bps"] = te["next_ret_bps"]
    for name, cc in [(f"FULL_{len(cols)}", cols)] + list(cands.items()):
        _, k, a_va = fit_select(fit, va, cc)
        m = hgb(k).fit(tr[cc], tr["y"])
        p = m.predict_proba(te[cc])[:, 1]
        preds[name] = p
        yy = te["y"].to_numpy()
        conf = np.abs(p - 0.5); top = conf >= np.quantile(conf, 0.9)
        rows.append(dict(name=name, n=len(cc), rounds=k, auc_valid=round(a_va, 4),
                         auc_test=round(roc_auc_score(yy, p), 4),
                         acc_test=round(float(((p > 0.5) == yy).mean()), 4),
                         acc_top10=round(float(((p[top] > 0.5) == yy[top]).mean()), 4)))
        log(f"{name}: {rows[-1]}")

    preds.to_parquet(os.path.join(HERE, "run1_boosting.parquet"))
    with open(os.path.join(HERE, "candidates.json"), "w", encoding="utf-8") as fh:
        json.dump(cands, fh, indent=2)

    L = ["# Candidate indicator sets + Pass 1 (gradient boosting)\n",
         f"Selected on validation {VA_START} → 2024-12-31. Tested on {te.index[0]:%Y-%m-%d} → "
         f"{te.index[-1]:%Y-%m-%d} ({len(te):,} candles), scored once.\n",
         "## Pass 1: boosting per candidate\n",
         "| Candidate | Indicators | Valid AUC | Test AUC | Test accuracy | Accuracy, top 10% most confident |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['n']} | {r['auc_valid']} | {r['auc_test']} | "
                 f"{r['acc_test']} | {r['acc_top10']} |")
    L += ["\n## Candidates\n"]
    for n, cc in cands.items():
        L.append(f"- **{n}**: " + ", ".join(f"`{c}`" for c in cc))
    L += ["\n## Indicator ranking (permutation importance = AUC lost when shuffled, on validation)\n",
          "| # | Indicator | Importance |", "|---|---|---|"]
    for i, (c, v) in enumerate(imp.items(), 1):
        L.append(f"| {i} | `{c}` | {v:+.5f} |")
    L += ["\n## Validation AUC using a single group\n", "| Group | Valid AUC |", "|---|---|"]
    for g, a in sorted(grp_auc.items(), key=lambda x: -x[1]):
        L.append(f"| {g} | {a:.4f} |")
    with open(os.path.join(HERE, "candidates.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("done -> candidates.md")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
