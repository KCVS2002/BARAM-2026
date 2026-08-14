"""#90 진단: 2025 테스트 기간의 ā 괴리 측정 (제출 불요).

v57에서 selfbar 이득은 ā 괴리(|ā_실측 − ā_train|/ā_train)에 비례함을 확인
(괴리 25~35% fold에서 +0.002, 15% 이하에서 노이즈). 2025의 괴리를 테스트
원자에서 직접 추정해 프로브 가치 판단. main GBM(q3, 공식 구성) 경로만 사용 —
AnEn/sister는 ā 추정에 영향 미미 (med 중심 재배치라 기간 평균 보존).
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
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"


def a_bar_self(atoms, cap):
    valid = atoms >= 0.1 * cap
    p_valid = valid.mean(axis=1)
    e_valid = np.where(valid, atoms, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    m = p_valid > 0
    return float((e_valid[m] * p_valid[m]).sum() / p_valid[m].sum())


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
    weights_main = weights.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        weights_main[tgt] = weights_main[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_tr, on="forecast_kst_dtm", how="left")
    te = sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
    ifs = load_ifs_features()
    df = df.merge(ifs, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs, on="forecast_kst_dtm", how="left")
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    mu_t, sd_t = df[tri].mean(), df[tri].std()
    for d_ in (df, te):
        z = (d_[tri] - mu_t) / sd_t
        d_["cons3_mean"] = z.mean(axis=1)
        d_["cons3_std"] = z.std(axis=1)
    ifs2 = load_ifs2_features()
    df = df.merge(ifs2, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs2, on="forecast_kst_dtm", how="left")
    for d_ in (df, te):
        d_["ifs_shear"] = d_["ifs_ws700"] - d_["ifs_ws925"]
    ifs_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
                "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]

    stack_X, stack_y, stack_w = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna()
        Xg = df.loc[trm, feature_cols + ifs_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xg["g_rated"], Xg["g_rotor"] = rated, rotor
        Xg["g_id"] = list(GROUP_META).index(tgt)
        stack_X.append(Xg)
        stack_y.append(df.loc[trm, tgt] / cap)
        stack_w.append(weights_main.loc[trm.to_numpy(), tgt])
    X_all, y_all, w_all = pd.concat(stack_X), pd.concat(stack_y), pd.concat(stack_w)
    shared_cols = feature_cols + ifs_cols + ["g_rated", "g_rotor", "g_id"]
    qmodels = [lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
               .fit(X_all[shared_cols], y_all, sample_weight=w_all) for q in QUANTILES]
    print(f"GBM done ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== 2025 테스트 ā 괴리 진단 ===")
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        rated, rotor = GROUP_META[tgt]
        Xv = te[feature_cols + ifs_cols].copy()
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
        atoms = interp_atoms(qp, n=300)
        a_hat = a_bar_self(atoms, cap)
        a = df.loc[df[tgt].notna(), tgt]
        a_tr = float(a[a >= cap * 0.10].mean())
        print(f"{tgt}: ā_train {a_tr:.0f} → ā_hat(2025) {a_hat:.0f} "
              f"(괴리 {(a_hat-a_tr)/a_tr*100:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
