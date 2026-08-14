"""v52: sister 고출력 가중 (#84 후속) — g3 OM sister 학습에도 q3 가중 적용 A/B.

현행 공식(sub_029)은 main GBM만 ×(1+3cf²) — sister는 무가중. 같은 정렬 논리를
sister에 적용하면 g3 고출력 정확도가 추가로 오르는지 검증.
- base: main q3 + sister 무가중 (현행) / sw: main q3 + sister q3
홀드아웃 2024-10~12, g3 전용, sister 조건부 원자수(#78)는 양쪽 동일 적용.
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
from scripts.train_v22_omsister import load_om
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
P_MIX = 0.15
TGT = "kpx_group_3"
NA = {0: 125, 1: 150, 2: 175}  # 가까움/중간/멂


def row_interp(src: np.ndarray, n: int) -> np.ndarray:
    return np.interp(np.linspace(0, len(src) - 1, n), np.arange(len(src)), src)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    w_q3 = clean_w.copy()
    for _t in TARGET_COLS:
        _cf = (lab[_t] / CAPACITY_KWH[_t]).fillna(0).to_numpy()
        w_q3[_t] = w_q3[_t].to_numpy() * (1 + 3 * np.clip(_cf, 0, 1) ** 2)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    om = load_om()
    df = df.merge(om, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    main_cols = base_cols + ifs10
    sis_cols = base_cols + om_cols
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    va_start = pd.Timestamp(2024, 10, 1, 1)
    va_end = pd.Timestamp(2025, 1, 1, 1)
    tr_idx = df.forecast_kst_dtm < va_start
    tr = df[tr_idx]
    va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
    X, y, w = stack_groups(tr, main_cols, w_q3[tr_idx.to_numpy()])
    shared = main_cols + ["g_rated", "g_rotor", "g_id"]
    models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
        X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
    sis_idx = tr_idx & (df.forecast_kst_dtm >= pd.Timestamp(2024, 1, 1, 1))
    tr_s = df[sis_idx]
    shared_s = sis_cols + ["g_rated", "g_rotor", "g_id"]
    sisters = {}
    for v, wsrc in (("base", clean_w), ("sw", w_q3)):
        Xs, ys, ws = stack_groups(tr_s, sis_cols, wsrc[sis_idx.to_numpy()])
        sisters[v] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            Xs[shared_s], ys, sample_weight=ws) for q in QUANTILES_FULL}
        print(f"sister {v} 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

    cap = CAPACITY_KWH[TGT]
    trm = tr[TGT].notna()
    sub = va.loc[va[TGT].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
    Xv = group_X(sub, main_cols, TGT)
    qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                          for q in QUANTILES_FULL])
    qp.sort(axis=1)
    qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
    gbm300 = interp_atoms(qp, n=300)
    gbm150 = gbm300[:, np.linspace(0, 299, 150).astype(int)]
    med = np.median(gbm150, axis=1)

    tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
    mu_a = tr_ok[ANEN_FEATS].mean()
    sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
    knn = NearestNeighbors(n_neighbors=200).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
    dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
    anen200 = np.sort((tr_ok[TGT] / cap).to_numpy()[aidx] * cap, axis=1)
    d_bar = dist[:, :150].mean(axis=1)
    terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)

    Xvs = group_X(sub, sis_cols, TGT)
    sis300s = {}
    for v in sisters:
        sq = np.column_stack([np.clip(sisters[v][q].predict(Xvs[shared_s]) * cap, 0, cap)
                              for q in QUANTILES_FULL])
        sq.sort(axis=1)
        sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
        sis300s[v] = interp_atoms(sq, n=300)

    n8 = max(int(round(300 * P_MIX * 2 / 3)), 1)
    n6 = max(int(round(300 * P_MIX * 1 / 3)), 1)
    mix8 = gbm150[:, np.linspace(0, 149, n8).astype(int)] * 0.8
    mix6 = gbm150[:, np.linspace(0, 149, n6).astype(int)] * 0.6

    a_tr = tr.loc[trm, TGT]
    a_bar = a_tr[a_tr >= cap * 0.10].mean()
    actual = sub[TGT].to_numpy()
    n = len(sub)

    # 공통: 조건부 base 300 (sub_025 채택분)
    base300 = np.empty((n, 300))
    for i in range(n):
        na = NA[terc[i]]
        an = row_interp(anen200[i], na)
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = row_interp(gbm300[i], 300 - na)
        base300[i] = np.sort(np.concatenate([an, gb]))

    res = {}
    for v, cond_sis in (("base", True), ("sw", True)):
        sis300 = sis300s[v]
        atoms = np.empty((n, 300 + n8 + n6 + 175))
        for i in range(n):
            ns = NA[terc[i]] if cond_sis else 150  # 멂=175 / 가까움=125 (inv_soft 동일)
            sis = row_interp(sis300[i], ns)
            pad = np.full(175 - ns, np.nan)
            atoms[i] = np.sort(np.concatenate([base300[i], mix8[i], mix6[i], sis, pad]))
        preds = np.empty(n)
        for t_ in (125, 150, 175):
            m = np.array([(NA[terc[i]] if cond_sis else 150) == t_ for i in range(n)])
            if not m.any():
                continue
            k = 300 + n8 + n6 + t_
            preds[m] = optimize_submission(np.sort(atoms[m][:, :k], axis=1), cap, a_bar)
        s, nm, fi, _ = metric_single(actual, preds, cap)
        res[v] = (s, nm, fi)
        print(f"{v:8s}: g3 {s:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== v52 sister 가중 (홀드아웃 g3) === 차이 {res['sw'][0]-res['base'][0]:+.4f}")


if __name__ == "__main__":
    main()
