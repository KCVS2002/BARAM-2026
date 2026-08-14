"""v38b: XGB 원자 풀링 — 분위 평균(v38 mix, 선명화 함정) 대신 **원자 풀에 병렬 추가**.

v38 진단: mix(50:50 분위 평균)는 두 가족의 분위 곡선을 평균해 앙상블을 선명화 →
FICR -0.0017 (v37 sb5와 동일 함정). 올바른 결합은 파이프라인의 기존 문법 —
GBM+AnEn+sister처럼 **독립 소스를 원자로 병렬 풀링** (다양성을 폭이 아닌 구성원으로).
- pool: 원자 = LGBM 150 + XGB 150(중앙값 재정렬) + AnEn 150. sister(OM, g3) 채택과
  동일 구조 (재정렬은 sub_012/013에서 검증된 raw/shift 중 shift — 소스 편향 제거).
- XGB 원자도 사이클내 3h 평활 적용 (LGBM과 동일 후처리).

변형: base(LGBM+AnEn, 현행) / pool(+XGB 150원자).
기준: 공식 피처셋(base124+IFS10) CV 0.6591. 하네스 v31b 동일.
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
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
XGB_PARAMS = dict(
    n_estimators=700, learning_rate=0.05, max_leaves=63, grow_policy="lossguide",
    max_depth=0, tree_method="hist", subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0, n_jobs=-1,
    objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES_FULL),
)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
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

    variants = ("base", "pool")
    res = {v: {} for v in variants}
    dec = {v: {} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        X, y, w = stack_groups(tr, cols, w_tr)
        shared = cols + ["g_rated", "g_rotor", "g_id"]
        lmodels = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        xmodel = xgb.XGBRegressor(**XGB_PARAMS)
        xmodel.fit(X[shared], y, sample_weight=w)
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        dc = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qp_l = np.column_stack([np.clip(lmodels[q].predict(Xv[shared]) * cap, 0, cap)
                                    for q in QUANTILES_FULL])
            qp_x = np.clip(xmodel.predict(Xv[shared]) * cap, 0, cap)
            if qp_x.ndim == 1:
                raise RuntimeError("XGB 다중분위 예측 형상 이상")
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            # LGBM 원자 (공통)
            qp = np.sort(qp_l, axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm = interp_atoms(qp, n=150)
            med = np.median(gbm, axis=1)
            anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
            # XGB 원자 (동일 후처리 + LGBM 중앙값으로 재정렬)
            qx = np.sort(qp_x, axis=1)
            qx = np.sort(smooth_quantiles_by_day(qx, sub["forecast_kst_dtm"]), axis=1)
            xatoms = interp_atoms(qx, n=150)
            xatoms = np.clip(xatoms + (med - np.median(xatoms, axis=1))[:, None], 0, cap)
            for v, parts in (("base", [gbm, anen]), ("pool", [gbm, xatoms, anen])):
                atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v38b XGB 원자 풀링 ===")
    for v in variants:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v:4s}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f}")


if __name__ == "__main__":
    main()
