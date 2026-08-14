"""제출 v7: LightGBM(0.6) + CatBoost(0.4) 퀀타일 혼합 앙상블 (v10b_mix, CV 0.6368).

sub_003 구성(제공 피처, 정제 가중, 공유 학습) 위에 이종 알고리즘 혼합만 추가.
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

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_META, QUANTILES, label_weights
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
W_LGB, W_CAT = 0.6, 0.4


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    sub = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"])

    feat_tr = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    feat_te = build_features(
        pd.read_csv(DATA / "test/ldaps_test.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "test/gfs_test.csv", encoding="utf-8-sig"),
    )
    feature_cols = [c for c in feat_tr.columns if c != "forecast_kst_dtm"]
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_tr, on="forecast_kst_dtm", how="left")
    X_te_base = sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")[feature_cols]

    stack_X, stack_y, stack_w = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna()
        Xg = df.loc[trm, feature_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xg["g_rated"], Xg["g_rotor"] = rated, rotor
        Xg["g_id"] = list(GROUP_META).index(tgt)
        stack_X.append(Xg)
        stack_y.append(df.loc[trm, tgt] / cap)
        stack_w.append(weights.loc[trm.to_numpy(), tgt])
    X_all, y_all, w_all = pd.concat(stack_X), pd.concat(stack_y), pd.concat(stack_w)
    shared_cols = feature_cols + ["g_rated", "g_rotor", "g_id"]

    lgb_models, cat_models = [], []
    Xf = X_all[shared_cols].fillna(-999)
    for q in QUANTILES:
        m1 = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
        m1.fit(X_all[shared_cols], y_all, sample_weight=w_all)
        lgb_models.append(m1)
        m2 = CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=500,
                               learning_rate=0.08, depth=6, random_seed=42, verbose=False)
        m2.fit(Xf, y_all, sample_weight=w_all)
        cat_models.append(m2)
        print(f"q{q:.2f} done ({time.time()-t0:.0f}s)")

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        Xv = X_te_base.copy()
        rated, rotor = GROUP_META[tgt]
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        Xvf = Xv[shared_cols].fillna(-999)
        qp_l = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in lgb_models])
        qp_c = np.column_stack([np.clip(m.predict(Xvf) * cap, 0, cap) for m in cat_models])
        qp_l.sort(axis=1)
        qp_c.sort(axis=1)
        qp = W_LGB * qp_l + W_CAT * qp_c
        qp = np.sort(smooth_quantiles_by_day(np.sort(qp, axis=1), sub["forecast_kst_dtm"]), axis=1)
        a = df.loc[df[tgt].notna(), tgt]
        a_bar = a[a >= cap * 0.10].mean()
        out[tgt] = optimize_submission(interp_atoms(qp), cap, a_bar)
        print(f"{tgt}: pred mean={out[tgt].mean():.0f}")

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_007_lgb_cat_mix.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
