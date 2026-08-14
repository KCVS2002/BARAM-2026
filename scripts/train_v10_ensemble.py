"""v10: 구조적 앙상블 실험 (피처 확장 아님 — LB 전이 실적이 있는 유형).

  v10a: 소스별 sister (LDAPS-only / GFS-only / 결합) 퀀타일 평균
  v10b: LightGBM + CatBoost 이종 앙상블 (결합 피처, 퀀타일 평균)

기반: sub_003 구성 (v2b + smooth/interp, 제공 피처만). 기준 CV 0.6346.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, QUANTILES, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"


def fit_predict_lgb(tr, va_subs, cols, w_tr):
    X, y, w = stack_groups(tr, cols, w_tr)
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    models = [lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
        X[shared], y, sample_weight=w) for q in QUANTILES]
    out = {}
    for tgt, sub in va_subs.items():
        cap = CAPACITY_KWH[tgt]
        Xv = group_X(sub, cols, tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared]) * cap, 0, cap) for m in models])
        qp.sort(axis=1)
        out[tgt] = qp
    return out


def fit_predict_cat(tr, va_subs, cols, w_tr):
    X, y, w = stack_groups(tr, cols, w_tr)
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    Xf, Xv_cache = X[shared].fillna(-999), {}
    models = []
    for q in QUANTILES:
        m = CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=500,
                              learning_rate=0.08, depth=6, random_seed=42, verbose=False)
        m.fit(Xf, y, sample_weight=w)
        models.append(m)
    out = {}
    for tgt, sub in va_subs.items():
        cap = CAPACITY_KWH[tgt]
        Xv = group_X(sub, cols, tgt)[shared].fillna(-999)
        qp = np.column_stack([np.clip(m.predict(Xv) * cap, 0, cap) for m in models])
        qp.sort(axis=1)
        out[tgt] = qp
    return out


def evaluate(qp_dict_list, weights_list, va_subs, tr):
    fs = []
    for tgt, sub in va_subs.items():
        cap = CAPACITY_KWH[tgt]
        qp = sum(w * d[tgt] for w, d in zip(weights_list, qp_dict_list))
        qp = np.sort(smooth_quantiles_by_day(np.sort(qp, axis=1), sub["forecast_kst_dtm"]), axis=1)
        a_tr = tr.loc[tr[tgt].notna(), tgt]
        a_bar = a_tr[a_tr >= cap * 0.10].mean()
        pred = optimize_submission(interp_atoms(qp), cap, a_bar)
        s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
        fs.append(s)
    return np.nanmean(fs)


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
    ldaps_cols = [c for c in base_cols if c.startswith("ldaps_")] + \
                 ["hour_sin", "hour_cos", "month_sin", "month_cos", "lead_h"]
    gfs_cols = [c for c in base_cols if c.startswith("gfs_") or c.startswith("wpd_") or c in
                ("air_density", "shear_100_10", "shear_850_100")] + \
               ["hour_sin", "hour_cos", "month_sin", "month_cos", "lead_h"]

    res = {k: {} for k in ["v10a_sister", "v10b_cat", "v10b_mix", "lgb_alone"]}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]
        va_subs = {t: va.loc[va[t].notna()].sort_values("forecast_kst_dtm") for t in TARGET_COLS}
        fold = f"{va_start:%Y-%m}"

        qp_comb = fit_predict_lgb(tr, va_subs, base_cols, w_tr)
        qp_ld = fit_predict_lgb(tr, va_subs, ldaps_cols, w_tr)
        qp_gf = fit_predict_lgb(tr, va_subs, gfs_cols, w_tr)
        qp_cat = fit_predict_cat(tr, va_subs, base_cols, w_tr)

        res["lgb_alone"][fold] = evaluate([qp_comb], [1.0], va_subs, tr)
        res["v10a_sister"][fold] = evaluate([qp_comb, qp_ld, qp_gf], [0.5, 0.25, 0.25], va_subs, tr)
        res["v10b_cat"][fold] = evaluate([qp_cat], [1.0], va_subs, tr)
        res["v10b_mix"][fold] = evaluate([qp_comb, qp_cat], [0.6, 0.4], va_subs, tr)
        print(f"fold {fold}: lgb={res['lgb_alone'][fold]:.4f} sister={res['v10a_sister'][fold]:.4f} "
              f"cat={res['v10b_cat'][fold]:.4f} mix={res['v10b_mix'][fold]:.4f} ({time.time()-t0:.0f}s)")

    print("\n=== 요약 (기준 0.6346) ===")
    for k, d in res.items():
        vals = list(d.values())
        print(f"{k}: {np.mean(vals):.4f} (std {np.std(vals):.4f})")


if __name__ == "__main__":
    main()
