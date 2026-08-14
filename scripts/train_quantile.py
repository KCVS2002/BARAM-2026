"""LightGBM 퀀타일 회귀 + FICR 의사결정 최적화 CV 평가.

비교군:
  (a) L1 점예측 (베이스라인과 동일)
  (b) 퀀타일 중앙값(q0.5)
  (c) FICR 기대점수 최적화 제출값  <- 핵심 차별화
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import optimize_submission
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

BASE_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    ldaps = pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig")
    feat = build_features(ldaps, gfs)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    feature_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    print(f"features ready ({time.time()-t0:.0f}s)")

    folds = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        folds.append((va_start, va_start + pd.DateOffset(months=2)))

    rows = []
    for va_start, va_end in folds:
        tr = df[df.forecast_kst_dtm < va_start]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        row = {"fold": f"{va_start:%Y-%m}"}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            X_tr, y_tr = tr.loc[trm, feature_cols], tr.loc[trm, tgt] / cap
            vam = va[tgt].notna().to_numpy()
            actual = va.loc[va[tgt].notna(), tgt].to_numpy()

            # 유효시간 평균 발전량 (train에서 추정)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()

            # (a) L1 점예측
            m_l1 = lgb.LGBMRegressor(objective="l1", **BASE_PARAMS)
            m_l1.fit(X_tr, y_tr)
            pred_l1 = np.clip(m_l1.predict(va[feature_cols]) * cap, 0, cap)[vam]

            # 퀀타일 모델들
            qp = np.empty((len(va), len(QUANTILES)))
            for j, q in enumerate(QUANTILES):
                m_q = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
                m_q.fit(X_tr, y_tr)
                qp[:, j] = np.clip(m_q.predict(va[feature_cols]) * cap, 0, cap)
            qp.sort(axis=1)  # 퀀타일 교차 정렬
            pred_med = qp[vam, len(QUANTILES) // 2]
            pred_opt = optimize_submission(qp[vam], cap, a_bar)

            for name, pred in [("l1", pred_l1), ("med", pred_med), ("opt", pred_opt)]:
                s, one_nmae, ficr, n = metric_single(actual, pred, cap)
                row[f"{tgt}_{name}"] = s
                row[f"{tgt}_{name}_nmae"] = one_nmae
                row[f"{tgt}_{name}_ficr"] = ficr
        for name in ("l1", "med", "opt"):
            row[f"score_{name}"] = np.nanmean([row[f"{t}_{name}"] for t in TARGET_COLS])
        rows.append(row)
        print(f"fold {row['fold']}: l1={row['score_l1']:.4f} med={row['score_med']:.4f} "
              f"opt={row['score_opt']:.4f} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    print("\n=== CV mean ===")
    for name in ("l1", "med", "opt"):
        nmae = np.nanmean([res[f"{t}_{name}_nmae"].mean() for t in TARGET_COLS])
        ficr = np.nanmean([res[f"{t}_{name}_ficr"].mean() for t in TARGET_COLS])
        print(f"{name}: score={res[f'score_{name}'].mean():.4f} (std {res[f'score_{name}'].std():.4f}) "
              f"| 1-NMAE={nmae:.4f} FICR={ficr:.4f}")
    out = PROJECT / "experiments"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "quantile_ficr_cv.csv", index=False, encoding="utf-8-sig")
    print(f"saved experiments/quantile_ficr_cv.csv ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
