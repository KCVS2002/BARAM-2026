"""실험 v5: 시드 앙상블 (3 seeds × 19 quantiles) + smooth/interp 의사결정.

비교 기준: v2b + smooth = 0.6346 (단일 시드).
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, QUANTILES, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
SEEDS = [42, 202, 777]


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    shared_cols = base_cols + ["g_rated", "g_rotor", "g_id"]
    print(f"ready ({time.time()-t0:.0f}s)")

    rows = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights[tr_idx.to_numpy()]
        X, y, w = stack_groups(tr, base_cols, w_tr)

        # seed × quantile 모델
        models = {}
        for seed in SEEDS:
            params = {**BASE_PARAMS, "random_state": seed}
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
                m.fit(X[shared_cols], y, sample_weight=w)
                models[(seed, q)] = m

        row = {"fold": f"{va_start:%Y-%m}"}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            sub = va.loc[vam].sort_values("forecast_kst_dtm")
            actual = sub[tgt].to_numpy()
            Xv = group_X(sub, base_cols, tgt)
            qp = np.zeros((len(sub), len(QUANTILES)))
            for j, q in enumerate(QUANTILES):
                preds = [np.clip(models[(s, q)].predict(Xv[shared_cols]) * cap, 0, cap) for s in SEEDS]
                qp[:, j] = np.mean(preds, axis=0)
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            atoms = interp_atoms(qp)
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(atoms, cap, a_bar)
            s, _, _, _ = metric_single(actual, pred, cap)
            row[tgt] = s
        row["score"] = np.nanmean([row[t] for t in TARGET_COLS])
        rows.append(row)
        print(f"fold {row['fold']}: {row['score']:.4f} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    print(f"\n=== v5 CV mean: {res.score.mean():.4f} (std {res.score.std():.4f}) | 기준 0.6346 ===")
    res.to_csv(PROJECT / "experiments" / "v5_seed_ens_cv.csv", index=False, encoding="utf-8-sig")
    print(f"saved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
