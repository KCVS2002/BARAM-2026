"""v18: 이종 모델 원자 — XGBoost 멀티퀀타일을 sub_009 원자 풀에 결합.

가설: 시드 앙상블(동종 LGBM) 실패 원인은 원자 다양성 부재였음. 이종 부스팅 패밀리
(XGBoost, 다른 분할·정칙화·퀀타일 구현)는 오류 상관이 낮아 point/분포 양면에서
실질적 다양성을 추가할 수 있다.

변형 2종을 fold마다 동시 평가:
- xgb_raw  : XGB 원자를 재정렬 없이 그대로 결합 (point 다양성까지 활용)
- xgb_shift: GBM 중앙값으로 재정렬 후 결합 (분포 모양만 추가, AnEn과 동일 방식)

기준: sub_009 구성(gbm+anen) CV 0.6470 — 동일 코드로 base도 재계산해 fold별 비교.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILES_FULL,
    learning_rate=0.05,
    grow_policy="lossguide",
    max_depth=0,
    max_leaves=63,
    min_child_weight=40,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    seed=42,
    verbosity=0,
)
XGB_ROUNDS = 700


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in ("base", "xgb_raw", "xgb_shift")}
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
        dtrain = xgb.QuantileDMatrix(X[shared], y, weight=w)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=XGB_ROUNDS)
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in res}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm = interp_atoms(qp, n=150)
            med = np.median(gbm, axis=1)

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)

            xq = np.clip(booster.inplace_predict(Xv[shared].to_numpy()) * cap, 0, cap)
            xq.sort(axis=1)
            xq = np.sort(smooth_quantiles_by_day(xq, sub["forecast_kst_dtm"]), axis=1)
            xgb_raw = interp_atoms(xq, n=150)
            xgb_shift = np.clip(xgb_raw + (med - np.median(xgb_raw, axis=1))[:, None], 0, cap)

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            for v, extra in (("base", None), ("xgb_raw", xgb_raw), ("xgb_shift", xgb_shift)):
                parts = [gbm, anen] + ([extra] if extra is not None else [])
                atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
        for v in res:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: base={res['base'][fold]:.4f} raw={res['xgb_raw'][fold]:.4f} "
              f"shift={res['xgb_shift'][fold]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v18 XGB 이종 원자 ===")
    for v in res:
        print(f"{v}: {np.mean(list(res[v].values())):.4f}")


if __name__ == "__main__":
    main()
