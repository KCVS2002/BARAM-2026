"""v24: 정칙화 강화 프로브 (#51b 법칙의 역방향 — CV는 비용 측정용, 판정은 LB).

법칙(CLAUDE.md §7): 학습분포 적합도를 높이는 변경은 CV 승·LB 패 (3회 실증).
→ 역방향 가설: 적합도를 낮추는(강건화) 변경은 CV에서 소폭 지더라도 LB에서 이길 수 있다.
#9는 정칙화 완화(-0.009 LB)만 검증했고 강화 방향은 미검증.

변형:
- base   : BASE_PARAMS (LB 검증 현행)
- reg    : num_leaves 31, min_child_samples 120, subsample/colsample 0.7 (구조 정칙화)
- lowlr  : lr 0.025 × n_estimators 1400 (부드러운 적합, 나머지 base)
CV 손실이 ~0.002 이내면 LB 프로브 후보.
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
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
VARIANTS = {
    "base": BASE_PARAMS,
    "reg": {**BASE_PARAMS, "num_leaves": 31, "min_child_samples": 120,
            "subsample": 0.7, "colsample_bytree": 0.7},
    "lowlr": {**BASE_PARAMS, "learning_rate": 0.025, "n_estimators": 1400},
}


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

    res = {v: {} for v in VARIANTS}
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

        models = {v: {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
            for v, params in VARIANTS.items()}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            for v in VARIANTS:
                qp = np.column_stack([np.clip(models[v][q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm = interp_atoms(qp, n=150)
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v24 정칙화 강화 프로브 ===")
    for v in VARIANTS:
        print(f"{v}: {np.mean(list(res[v].values())):.4f}")


if __name__ == "__main__":
    main()
