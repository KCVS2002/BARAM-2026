"""OOF 원자 캐시 v2 — 현행 공식(sub_029) 기준 재생성 (#89 하류 재점검용).

구세대 캐시(oof_cache/, sub_009 구성)와의 차이:
  - 피처: 124 base + IFS10 (현행 공식 main GBM과 동일)
  - 가중: q3 = clean × (1+3·cf²) (현행 공식)
  - AnEn: 이웃 200개 + 평균거리 d̄ 저장 (#77 거리 조건부 재배분 실험 재료)
저장: experiments/oof_cache_q3/{fold}_{tgt}.npz
  dtm(int64 ns) / actual / qp(n,19 정렬·평활 완료) / anen(n,200) / dist(n,200)
  / a_bar / cap
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
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
OUT = PROJECT / "experiments" / "oof_cache_q3"
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
    w_q3 = clean_w.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        w_q3[tgt] = w_q3[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols = base_cols + ifs10
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = w_q3[tr_idx.to_numpy()]
        X, y, w = stack_groups(tr, cols, w_tr)
        shared = cols + ["g_rated", "g_rotor", "g_id"]
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            order = np.argsort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            anen = np.take_along_axis((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, order, axis=1)

            a_tr = tr.loc[trm, tgt]
            a_bar = float(a_tr[a_tr >= cap * 0.10].mean())
            np.savez_compressed(
                OUT / f"{fold}_{tgt}.npz",
                dtm=sub["forecast_kst_dtm"].to_numpy().astype("datetime64[ns]").astype(np.int64),
                actual=sub[tgt].to_numpy(),
                qp=qp, anen=anen, dist=dist, a_bar=a_bar, cap=float(cap),
            )
        print(f"fold {fold} 저장 완료 ({time.time()-t0:.0f}s)", flush=True)
    print("=== OOF 캐시 v2 덤프 완료 ===", flush=True)


if __name__ == "__main__":
    main()
