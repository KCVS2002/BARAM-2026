"""OOF 원자 캐시 덤프 — 의사결정 계층 실험 가속용.

sub_009 구성(공유 LGBM 19q + AnEn K=150)의 fold별 검증 산출물을 npz로 저장:
  experiments/oof_cache/{fold}_{tgt}.npz
    dtm     : forecast_kst_dtm (int64 ns)
    actual  : 실측 발전량
    qp      : GBM 분위 예측 (n,19) — 정렬·사이클 평활화 완료 상태
    anen    : AnEn 원자 (n,150) — shift 재정렬 전 원본
    a_bar   : 학습기간 유효시간 평균 발전량 (스칼라)
    cap     : 그룹 설비용량 (스칼라)
이후 실험은 gbm=interp_atoms(qp), med, shift 등을 재료로 재학습 없이 조립.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
OUT = PROJECT / "experiments" / "oof_cache"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]

    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]
        X, y, w = stack_groups(tr, base_cols, w_tr)
        shared = base_cols + ["g_rated", "g_rotor", "g_id"]
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}

        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)

            a_tr = tr.loc[trm, tgt]
            a_bar = float(a_tr[a_tr >= cap * 0.10].mean())
            np.savez_compressed(
                OUT / f"{fold}_{tgt}.npz",
                dtm=sub["forecast_kst_dtm"].to_numpy().astype("datetime64[ns]").astype(np.int64),
                actual=sub[tgt].to_numpy(),
                qp=qp, anen=anen, a_bar=a_bar, cap=float(cap),
            )
        print(f"fold {fold} 저장 완료 ({time.time()-t0:.0f}s)", flush=True)
    print("=== OOF 캐시 덤프 완료 ===", flush=True)


if __name__ == "__main__":
    main()
