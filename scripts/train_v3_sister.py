"""실험 v3: 다중 NWP sister 모델 2단 블렌딩.

- 모델 A: v2b 파이프라인 (제공 데이터, 전 기간 학습) — 기존 최고 CV 0.6283
- 모델 B: v2b + 외부 NWP 피처 (ECMWF/ICON/GFS day2), 외부데이터 존재 구간(2024-02~)만 학습
- 블렌딩: 퀀타일 평균 (vincentization), B 학습량 부족 fold는 A만 사용

주의: CV 초반 fold(2024-01/03)는 B의 학습 데이터가 거의 없어 외부 효과가 과소평가됨.
     실제 테스트(2025)는 외부 커버 학습이 10.5개월이라 후반 fold가 더 대표적.
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
from src.decision import optimize_submission
from src.external import build_external_features
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
MIN_B_ROWS = 3000  # 그룹당 B 최소 학습 행 수


def stack_groups(df, feature_cols, weights=None):
    xs, ys, ws = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        m = df[tgt].notna()
        X = df.loc[m, feature_cols].copy()
        rated, rotor = GROUP_META[tgt]
        X["g_rated"], X["g_rotor"] = rated, rotor
        X["g_id"] = list(GROUP_META).index(tgt)
        xs.append(X)
        ys.append(df.loc[m, tgt] / cap)
        if weights is not None:
            ws.append(weights.loc[m.to_numpy(), tgt])
    return (pd.concat(xs), pd.concat(ys), pd.concat(ws) if weights is not None else None)


def fit_quantiles(X, y, w, cols):
    models = []
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
        m.fit(X[cols], y, sample_weight=w)
        models.append(m)
    return models


def predict_quantiles(models, X, cols, cap):
    qp = np.column_stack([np.clip(m.predict(X[cols]) * cap, 0, cap) for m in models])
    qp.sort(axis=1)
    return qp


def group_X(base_X, feature_cols, tgt):
    X = base_X[feature_cols].copy()
    rated, rotor = GROUP_META[tgt]
    X["g_rated"], X["g_rotor"] = rated, rotor
    X["g_id"] = list(GROUP_META).index(tgt)
    return X


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
    shared_a = base_cols + ["g_rated", "g_rotor", "g_id"]
    shared_b = base_cols + ext_cols + ["g_rated", "g_rotor", "g_id"]
    ext_avail = df["om_ws100_mean"].notna()
    print(f"ready: ext coverage {ext_avail.mean()*100:.1f}% of train ({time.time()-t0:.0f}s)")

    rows = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights[tr_idx.to_numpy()]

        XA, yA, wA = stack_groups(tr, base_cols, w_tr)
        mA = fit_quantiles(XA, yA, wA, shared_a)

        tr_b = tr[ext_avail[tr_idx.to_numpy()].to_numpy()]
        w_tr_b = w_tr[ext_avail[tr_idx.to_numpy()].to_numpy()]
        use_b = tr_b[TARGET_COLS].notna().sum().min() >= MIN_B_ROWS
        if use_b:
            XB, yB, wB = stack_groups(tr_b, base_cols + ext_cols, w_tr_b)
            mB = fit_quantiles(XB, yB, wB, shared_b)

        row = {"fold": f"{va_start:%Y-%m}", "b_used": bool(use_b)}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            actual = va.loc[vam, tgt].to_numpy()
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()

            qpA = predict_quantiles(mA, group_X(va.loc[vam], base_cols, tgt), shared_a, cap)
            variants = {"A": qpA}
            if use_b:
                qpB = predict_quantiles(mB, group_X(va.loc[vam], base_cols + ext_cols, tgt), shared_b, cap)
                variants["B"] = qpB
                variants["AB"] = 0.5 * qpA + 0.5 * qpB
            for name, qp in variants.items():
                pred = optimize_submission(np.sort(qp, axis=1), cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                row[f"{tgt}_{name}"] = s
        for name in ("A", "B", "AB"):
            vals = [row.get(f"{t}_{name}") for t in TARGET_COLS]
            row[f"score_{name}"] = np.nanmean([v for v in vals if v is not None]) if any(v is not None for v in vals) else np.nan
        rows.append(row)
        print(f"fold {row['fold']}: A={row['score_A']:.4f} "
              f"B={row.get('score_B', float('nan')):.4f} AB={row.get('score_AB', float('nan')):.4f} "
              f"({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    print("\n=== CV mean (v2b 참고: 0.6283) ===")
    for name in ("A", "B", "AB"):
        col = f"score_{name}"
        if col in res:
            print(f"{name}: {res[col].mean():.4f} (유효 fold {res[col].notna().sum()}개)")
    late = res[res.b_used]
    if len(late):
        print(f"외부데이터 유효 fold만: A={late.score_A.mean():.4f} "
              f"B={late.score_B.mean():.4f} AB={late.score_AB.mean():.4f}")
    res.to_csv(PROJECT / "experiments" / "v3_sister_cv.csv", index=False, encoding="utf-8-sig")
    print(f"saved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
