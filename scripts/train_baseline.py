"""LightGBM 베이스라인 + rolling-origin 시계열 CV.

- 그룹별 LightGBM (L1 목적함수), capacity-factor 타깃.
- CV: 2024년 2개월 단위 6개 fold (해당 fold 시작 이전 데이터만 학습 → 누수 없음).
- 리포트: 대회 metric (A >= 10% 용량 필터 포함) fold별/평균.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"


def load_all():
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    ldaps = pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig")
    return lab, ldaps, gfs


def main() -> None:
    t0 = time.time()
    lab, ldaps, gfs = load_all()
    feat = build_features(ldaps, gfs)
    print(f"features: {feat.shape} ({time.time()-t0:.0f}s)")

    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(
        feat, on="forecast_kst_dtm", how="left"
    )
    feature_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]

    # rolling-origin folds: 2024년을 2개월씩 6개
    folds = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        folds.append((va_start, va_end))

    params = dict(
        objective="l1",
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        verbose=-1,
    )

    results = []
    for va_start, va_end in folds:
        tr = df[df.forecast_kst_dtm < va_start]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        row = {"fold": f"{va_start:%Y-%m}"}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            model = lgb.LGBMRegressor(**params)
            model.fit(tr.loc[trm, feature_cols], tr.loc[trm, tgt] / cap)
            pred = np.clip(model.predict(va[feature_cols]) * cap, 0, cap)
            vam = va[tgt].notna()
            score, one_nmae, ficr, n = metric_single(
                va.loc[vam, tgt].to_numpy(), pred[vam.to_numpy()], cap
            )
            row[f"{tgt}_score"] = score
            row[f"{tgt}_1nmae"] = one_nmae
            row[f"{tgt}_ficr"] = ficr
            row[f"{tgt}_n"] = n
        row["score"] = np.nanmean([row[f"{t}_score"] for t in TARGET_COLS])
        results.append(row)
        print(f"fold {row['fold']}: score={row['score']:.4f} "
              + " ".join(f"{t[-1]}={row[f'{t}_score']:.4f}(n={row[f'{t}_n']})" for t in TARGET_COLS))

    res = pd.DataFrame(results)
    mean_1nmae = np.nanmean([res[f"{t}_1nmae"].mean() for t in TARGET_COLS])
    mean_ficr = np.nanmean([res[f"{t}_ficr"].mean() for t in TARGET_COLS])
    print(f"\n=== CV mean score: {res.score.mean():.4f} (std {res.score.std():.4f}) | "
          f"1-NMAE={mean_1nmae:.4f} FICR={mean_ficr:.4f} ===")
    out = PROJECT / "experiments"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "baseline_lgbm_cv.csv", index=False, encoding="utf-8-sig")
    print(f"saved experiments/baseline_lgbm_cv.csv ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
