"""실험 v7: v2b + NOAA 보조변수 (전 기간 커버 — v3b 보류 건 재시도).

v3b 실패 원인이었던 '결측이 연도와 얽힘'이 해소된 조건.
비교 기준: v2b + smooth/interp = 0.6346.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import label_weights
from scripts.train_v3_sister import fit_quantiles, group_X, predict_quantiles, stack_groups
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.external_noaa import build_noaa_features
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    noaa = build_noaa_features(PROJECT)
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(noaa, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    noaa_cols = [c for c in noaa.columns if c != "forecast_kst_dtm"]
    all_cols = base_cols + noaa_cols
    shared_cols = all_cols + ["g_rated", "g_rotor", "g_id"]
    cov = df["noaa_vvel850"].notna().mean()
    print(f"ready: NOAA coverage {cov*100:.1f}% ({time.time()-t0:.0f}s)")

    rows = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights[tr_idx.to_numpy()]
        X, y, w = stack_groups(tr, all_cols, w_tr)
        models = fit_quantiles(X, y, w, shared_cols)

        row = {"fold": f"{va_start:%Y-%m}"}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            sub = va.loc[vam].sort_values("forecast_kst_dtm")
            qp = predict_quantiles(models, group_X(sub, all_cols, tgt), shared_cols, cap)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(interp_atoms(qp), cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            row[tgt] = s
        row["score"] = np.nanmean([row[t] for t in TARGET_COLS])
        rows.append(row)
        print(f"fold {row['fold']}: {row['score']:.4f} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    print(f"\n=== v7 CV mean: {res.score.mean():.4f} (std {res.score.std():.4f}) | 기준 0.6346 ===")
    res.to_csv(PROJECT / "experiments" / "v7_noaa_cv.csv", index=False, encoding="utf-8-sig")
    print(f"saved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
