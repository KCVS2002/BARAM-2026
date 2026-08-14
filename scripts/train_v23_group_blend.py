"""v23: 공유 GBM + 그룹별 GBM 분위 블렌드 (g1 풍향 조건부 편향 처방).

진단 (2026-07-15, OOF 캐시): 서풍(지배 풍향)에서 g1 편향 -4.4%p vs g2 -0.3%p.
원인: g1/g2는 g_rated·g_rotor 동일 → 공유 모델이 g_id 하나로만 구분, 그룹×풍향
상호작용 학습 부족. 재정렬 앵커를 통해 편향이 AnEn까지 전파.

처방: 그룹별 19분위 모델(자기 데이터만)을 별도 학습, 분위 단위로 공유와 가중 평균.
- w_grp ∈ {0.25, 0.5} × 그룹별 적용 (g3는 데이터 부족으로 공유 우위 예상 → 그룹별 w 확인)
- 블렌드 후 표준 파이프라인(평활→보간→AnEn 재정렬→FICR 최적화).
기준: sub_009 구성(gbm+anen) CV 0.6470. #4의 "그룹별 근소 우위" 복선 실행판.
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
W_GRID = (0.0, 0.25, 0.5)  # 그룹별 모델 가중


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

    res = {w: {t: [] for t in TARGET_COLS} for w in W_GRID}
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
        smodels = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        gmodels = {}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            Xg = tr.loc[trm, base_cols]
            yg = tr.loc[trm, tgt] / cap
            wg = w_tr.loc[trm.to_numpy(), tgt]
            gmodels[tgt] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                Xg, yg, sample_weight=wg) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            qs = np.column_stack([np.clip(smodels[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qg = np.column_stack([np.clip(gmodels[tgt][q].predict(Xv[base_cols]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            for wg_ in W_GRID:
                qp = (1 - wg_) * qs + wg_ * qg
                qp = np.sort(qp, axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm = interp_atoms(qp, n=150)
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                res[wg_][tgt].append(s)
        print(f"fold {fold}: " + " ".join(
            f"w{wg_}={np.nanmean([res[wg_][t][-1] for t in TARGET_COLS]):.4f}" for wg_ in W_GRID)
            + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v23 그룹 블렌드 (그룹별 평균) ===")
    for wg_ in W_GRID:
        tot = np.mean([np.nanmean(res[wg_][t]) for t in TARGET_COLS])
        per = " ".join(f"{t[-1]}={np.nanmean(res[wg_][t]):.4f}" for t in TARGET_COLS)
        print(f"w_grp={wg_}: 총 {tot:.4f} | {per}")


if __name__ == "__main__":
    main()
