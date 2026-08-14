"""v9: 모델링 개선 실험 배치 (진단 기반, 변형별 독립 평가).

진단 근거 (experiments/oof 분석 2026-07-12):
- cf 0.5~1.0 구간 = FICR 가중 72%, 적중률 최저 → (a) 고출력 가중
- 2024→2025 분포이동 실증 → (b) 최근성 가중, (d) 단조 제약
- 램프 구간 NMAE 0.166 vs 0.114 → (c) 램프/veer 피처

기준: v2b + smooth/interp = 0.6346.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, QUANTILES, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"


def add_ramp_features(feat: pd.DataFrame) -> pd.DataFrame:
    """(c) 램프·veer 피처: 사이클 내 다시차 변화 + 연직 풍향 시어."""
    feat = feat.sort_values("forecast_kst_dtm").reset_index(drop=True)
    block = (feat.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.date
    for c in ["gfs_ws100", "ldaps_ws50max", "gfs_ws850"]:
        g = feat.groupby(block)[c]
        feat[f"{c}_diff2"] = g.transform(lambda s: s.diff(2).fillna(0))
        feat[f"{c}_diff3"] = g.transform(lambda s: s.diff(3).fillna(0))
        feat[f"{c}_range6"] = g.transform(
            lambda s: s.rolling(6, center=True, min_periods=2).max()
            - s.rolling(6, center=True, min_periods=2).min())
    # 연직 veer: 100m vs 850hPa 풍향 차 (cos/sin 내적 기반)
    v_cos = (feat["gfs_ws100_dsin"] * feat["gfs_ws850_dsin"]
             + feat["gfs_ws100_dcos"] * feat["gfs_ws850_dcos"])
    feat["veer_100_850"] = v_cos
    # 풍향 시간 변화 (사이클 내)
    g = feat.groupby(block)["gfs_ws100_dsin"]
    feat["dir_change3"] = feat.groupby(block).apply(
        lambda d: (d["gfs_ws100_dsin"].diff(3).abs() + d["gfs_ws100_dcos"].diff(3).abs()).fillna(0)
    ).reset_index(level=0, drop=True)
    return feat


def run_cv(df, weights_fn, feature_cols, extra_params=None, label=""):
    t0 = time.time()
    shared_cols = feature_cols + ["g_rated", "g_rotor", "g_id"]
    params = dict(BASE_PARAMS)
    if extra_params:
        params.update(extra_params)
    fold_scores = {}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights_fn(tr, va_start)
        X, y, w = stack_groups(tr, feature_cols, w_tr)

        if "monotone_constraints" in params:
            mc = [params["monotone_constraints"].get(c, 0) for c in shared_cols]
            params2 = {**params, "monotone_constraints": mc}
        else:
            params2 = params
        models = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params2)
            m.fit(X[shared_cols], y, sample_weight=w)
            models.append(m)

        fs = []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            sub = va.loc[vam].sort_values("forecast_kst_dtm")
            Xv = group_X(sub, feature_cols, tgt)
            qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in models])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(interp_atoms(qp), cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            fs.append(s)
        fold_scores[f"{va_start:%Y-%m}"] = np.nanmean(fs)
    mean = np.mean(list(fold_scores.values()))
    print(f"{label}: mean={mean:.4f} | " + " ".join(f"{k}:{v:.4f}" for k, v in fold_scores.items())
          + f" ({time.time()-t0:.0f}s)")
    return mean, fold_scores


def main() -> None:
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")

    def w_clean(tr, va_start):
        return clean_w[(df.forecast_kst_dtm < va_start).to_numpy()]

    results = {}

    # (a) 고출력 가중: w *= 1 + 2*cf  (FICR의 A-가중 구조 반영)
    def w_highout(tr, va_start):
        w = w_clean(tr, va_start).copy()
        for tgt in TARGET_COLS:
            cf = (tr[tgt] / CAPACITY_KWH[tgt]).fillna(0).clip(0, 1)
            w[tgt] = w[tgt].to_numpy() * (1 + 2 * cf.to_numpy())
        return w
    results["a_highout"] = run_cv(df, w_highout, base_cols, label="(a) 고출력 가중")[0]

    # (b) 최근성 가중: 반감기 18개월
    def w_recency(tr, va_start):
        w = w_clean(tr, va_start).copy()
        age_days = (va_start - tr.forecast_kst_dtm).dt.days.to_numpy()
        decay = 0.5 ** (age_days / 540.0)
        for tgt in TARGET_COLS:
            w[tgt] = w[tgt].to_numpy() * decay
        return w
    results["b_recency"] = run_cv(df, w_recency, base_cols, label="(b) 최근성 가중")[0]

    # (c) 램프/veer 피처
    feat_c = add_ramp_features(feat.copy())
    cols_c = [c for c in feat_c.columns if c != "forecast_kst_dtm"]
    df_c = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_c, on="forecast_kst_dtm", how="left")

    def w_clean_c(tr, va_start):
        return clean_w[(df_c.forecast_kst_dtm < va_start).to_numpy()]
    results["c_ramp"] = run_cv(df_c, w_clean_c, cols_c, label="(c) 램프/veer 피처")[0]

    # (d) 단조 제약 (핵심 풍속 피처들에 +1)
    mono = {c: 1 for c in ["ldaps_ws50max", "gfs_ws100", "gfs_ws80", "gfs_ws10",
                           "wpd_100m", "wpd_ldaps50", "gfs_ws850",
                           "gfs_surface_0_gust"] if c in base_cols}
    results["d_monotone"] = run_cv(df, w_clean, base_cols,
                                   extra_params={"monotone_constraints": mono},
                                   label="(d) 단조 제약")[0]

    print("\n=== 요약 (기준 0.6346) ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f} ({v-0.6346:+.4f})")


if __name__ == "__main__":
    main()
