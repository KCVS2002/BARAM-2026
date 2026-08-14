"""실험 v3b: 단일 공유 모델 + NaN 허용 외부 NWP 피처 (전 기간 학습).

v3(별도 sister 모델)의 학습량 부족 문제를 피하는 절충안:
LightGBM의 결측 네이티브 처리를 이용해 2022~ 전체로 학습하되,
외부 피처는 존재하는 행(2024-02~)에서만 분기에 활용되게 한다.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_META, QUANTILES, label_weights
from scripts.train_v3_sister import fit_quantiles, group_X, predict_quantiles, stack_groups
from src.decision import optimize_submission
from src.external import build_external_features
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    ext = build_external_features(PROJECT)
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(ext, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ext_cols = [c for c in ext.columns if c != "forecast_kst_dtm"]
    all_cols = base_cols + ext_cols
    shared_cols = all_cols + ["g_rated", "g_rotor", "g_id"]
    print(f"ready ({time.time()-t0:.0f}s)")

    rows = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights[tr_idx.to_numpy()]

        X, y, w = stack_groups(tr, all_cols, w_tr)
        models = fit_quantiles(X, y, w, shared_cols)

        row = {"fold": f"{va_start:%Y-%m}"}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            actual = va.loc[vam, tgt].to_numpy()
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            qp = predict_quantiles(models, group_X(va.loc[vam], all_cols, tgt), shared_cols, cap)
            pred = optimize_submission(qp, cap, a_bar)
            s, _, _, _ = metric_single(actual, pred, cap)
            row[f"{tgt}"] = s
        row["score"] = np.nanmean([row[t] for t in TARGET_COLS])
        rows.append(row)
        print(f"fold {row['fold']}: {row['score']:.4f} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    print(f"\n=== v3b CV mean: {res.score.mean():.4f} (v2b/A 참고: 0.6283) ===")
    late = res[res.fold >= "2024-07"]
    print(f"후반 3개 fold (외부 유효): v3b={late.score.mean():.4f} vs A=0.6336")
    res.to_csv(PROJECT / "experiments" / "v3b_nanfeat_cv.csv", index=False, encoding="utf-8-sig")
    print(f"saved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
