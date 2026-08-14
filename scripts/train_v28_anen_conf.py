"""v28: AnEn 신뢰도(거리) 조건부 결합 — 구조적 규칙, 2024 상수 무의존.

가설: 아날로그 평균거리 d̄는 AnEn 분포의 신뢰도 신호다 (가까움 = 라이브러리 밀집
상황 = 실측 분포 정확 / 멂 = 희귀 상황 = 부정확). 시간별로 결합비를 d̄ 순위에 따라
조정하면 정보 손실 없이 원자 품질이 오른다.

- 규칙: 예측 집합 자체의 d̄ 3분위(예측 시점 계산 — 2024 측정 상수 아님)로
    가까움: anen 200 + gbm 100 / 중간: 150+150 / 멂: 100+200
- 대조(방향 반전) 변형 포함 — 메커니즘 검증용 (반전이 이기면 가설 기각)
기준: sub_009 구성 CV 0.6470.
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


def subsample_cols(a: np.ndarray, n: int) -> np.ndarray:
    idx = np.linspace(0, a.shape[1] - 1, n).astype(int)
    return a[:, idx]


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

    variants = ("base", "conf", "conf_inv")
    res = {v: {} for v in variants}
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

        fs = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm300 = interp_atoms(qp, n=300)  # 재배분용 상위 해상도
            gbm150 = subsample_cols(gbm300, 150)
            med = np.median(gbm150, axis=1)

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            lib = (tr_ok[tgt] / cap).to_numpy()
            anen200_raw = np.sort(lib[aidx] * cap, axis=1)  # (n, 200)
            d_bar = dist[:, :150].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)  # 0=가까움 1=중간 2=멂

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            for v in variants:
                n = len(sub)
                atoms_list = np.empty((n, 300))
                for i in range(n):
                    t_ = terc[i]
                    if v == "base":
                        na, ng = 150, 150
                    elif v == "conf":
                        na, ng = (200, 100) if t_ == 0 else ((150, 150) if t_ == 1 else (100, 200))
                    else:  # conf_inv (대조)
                        na, ng = (100, 200) if t_ == 0 else ((150, 150) if t_ == 1 else (200, 100))
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200_raw[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 299, ng), np.arange(300), gbm300[i])
                    atoms_list[i] = np.sort(np.concatenate([an, gb]))
                pred = optimize_submission(atoms_list, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v28 AnEn 신뢰도 조건부 결합 ===")
    for v in variants:
        print(f"{v}: {np.mean(list(res[v].values())):.4f}")


if __name__ == "__main__":
    main()
