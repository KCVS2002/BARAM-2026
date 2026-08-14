"""LightGBM 하이퍼파라미터 랜덤 탐색 (야간 배치).

- 12개 설정 × 3 fold (2024-07/09/11 — 테스트와 가장 유사한 후반 fold)
- 평가: v2b 파이프라인 + smooth/interp 의사결정 (현행 최고 구성)
- 기준 설정(BASE_PARAMS)의 3-fold 점수도 함께 산출해 비교
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import QUANTILES, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
FOLDS = [7, 9, 11]

rng = np.random.default_rng(7)
SPACE = dict(
    num_leaves=[31, 63, 127, 255],
    min_child_samples=[20, 40, 80],
    learning_rate=[0.03, 0.05, 0.07],
    n_estimators=[500, 700, 1100],
    colsample_bytree=[0.6, 0.8, 1.0],
    subsample=[0.7, 0.85, 1.0],
    reg_lambda=[0.0, 1.0, 5.0],
)
BASELINE = dict(num_leaves=63, min_child_samples=40, learning_rate=0.05,
                n_estimators=700, colsample_bytree=0.8, subsample=0.8, reg_lambda=0.0)


def sample_config():
    return {k: rng.choice(v).item() for k, v in SPACE.items()}


def evaluate(cfg, df, weights, base_cols, shared_cols) -> float:
    params = dict(objective="quantile", subsample_freq=1, random_state=42, verbose=-1, **cfg)
    scores = []
    for m0 in FOLDS:
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights[tr_idx.to_numpy()]
        X, y, w = stack_groups(tr, base_cols, w_tr)

        models = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(alpha=q, **params)
            m.fit(X[shared_cols], y, sample_weight=w)
            models.append(m)

        fold_scores = []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            sub = va.loc[vam].sort_values("forecast_kst_dtm")
            Xv = group_X(sub, base_cols, tgt)
            qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in models])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(interp_atoms(qp), cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            fold_scores.append(s)
        scores.append(np.nanmean(fold_scores))
    return float(np.mean(scores))


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    shared_cols = base_cols + ["g_rated", "g_rotor", "g_id"]

    results = []
    base_score = evaluate(BASELINE, df, weights, base_cols, shared_cols)
    results.append({"config": "BASELINE", **BASELINE, "score": base_score})
    print(f"BASELINE: {base_score:.4f} ({time.time()-t0:.0f}s)")

    for i in range(12):
        cfg = sample_config()
        s = evaluate(cfg, df, weights, base_cols, shared_cols)
        results.append({"config": f"rand{i}", **cfg, "score": s})
        print(f"rand{i}: {s:.4f} {cfg} ({time.time()-t0:.0f}s)")
        pd.DataFrame(results).to_csv(PROJECT / "experiments" / "tune_lgbm.csv",
                                     index=False, encoding="utf-8-sig")

    res = pd.DataFrame(results).sort_values("score", ascending=False)
    print("\n=== top 5 ===")
    print(res.head(5).to_string(index=False))
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
