"""v50: 결정층·검색층 통합 배터리 (B·C정식·D) — 학습 1회 공유, 변형 6종.

- pred_anen (B): AnEn 검색 좌표를 원시 피처(수동 가중) → GBM 19분위 벡터(조건부
  분포의 충분통계량)로 교체. 라이브러리 좌표는 학습 구간 in-sample 예측.
- dist (C1): #77 inv_soft 재현 (거리 3분위 ±25원자) — 재현 확인용.
- lvl (C2): 수준 3분위 — 고수준 anen175 / 중간 150 / 저수준 125 (캐시 발견 +0.0032).
- dist_lvl (C3): 두 신호 가산 (na = 150 ±25(dist) ±25(lvl), clip 100~200) — 독립성 판정.
- smco (D): 사이클내 평활 창을 램프 3분위로 가변 (램프↑ 1h / 중간 3h / 안정 5h).
기준: 공식 피처셋 feat10, CV 0.6591 (v31b 하네스 base와 동일 구성).
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
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
NA_D = {0: -25, 1: 0, 2: 25}  # 거리: 멂=+25 anen
NA_L = {0: -25, 1: 0, 2: 25}  # 수준: 고수준=+25 anen


def row_interp(src, n):
    return np.interp(np.linspace(0, len(src) - 1, n), np.arange(len(src)), src)


def build_g(gbm300, anen200, med, cap, a_bar, na_arr):
    n = len(med)
    atoms = np.empty((n, 300))
    for i in range(n):
        na = int(na_arr[i])
        an = row_interp(anen200[i], na)
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = row_interp(gbm300[i], 300 - na)
        atoms[i] = np.sort(np.concatenate([an, gb]))
    return optimize_submission(atoms, cap, a_bar)


def smooth_var(qp_raw, dtm, med_raw, cap):
    """램프 3분위 조건부 평활: 창 1/3/5."""
    day = (dtm - pd.Timedelta(hours=1)).dt.date
    med_s = pd.Series(med_raw, index=range(len(med_raw)))
    ramp = med_s.groupby(day.values).transform(lambda s: s.diff().abs().rolling(2).max()).fillna(0).to_numpy() / cap
    terc = np.searchsorted(np.quantile(ramp, [1 / 3, 2 / 3]), ramp)
    sm3 = smooth_quantiles_by_day(qp_raw, dtm)
    # 5h 창: 3h 평활을 한 번 더 (근사)
    sm5 = smooth_quantiles_by_day(sm3, dtm)
    out = qp_raw.copy()
    out[terc == 1] = sm3[terc == 1]
    out[terc == 0] = sm5[terc == 0]  # 안정 → 강평활
    return np.sort(out, axis=1)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols = base_cols + ifs10
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    variants = ["base", "dist", "lvl", "dist_lvl", "smco", "pred_anen"]
    res = {v: {} for v in variants}
    dec = {v: {} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        X, y, w = stack_groups(tr, cols, w_tr)
        shared = cols + ["g_rated", "g_rotor", "g_id"]
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        dc = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qp_raw = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
            qp_raw.sort(axis=1)
            med_raw = np.median(qp_raw, axis=1)
            qp_std = np.sort(smooth_quantiles_by_day(qp_raw, sub["forecast_kst_dtm"]), axis=1)
            qp_cond = smooth_var(qp_raw, sub["forecast_kst_dtm"], med_raw, cap)

            # 피처 공간 AnEn (거리 포함)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            lib = (tr_ok[tgt] / cap).to_numpy()
            anen200 = np.sort(lib[aidx] * cap, axis=1)
            d_bar = dist[:, :150].mean(axis=1)
            terc_d = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)

            # 예측분포 공간 AnEn (B): 라이브러리 = 학습 in-sample 분위 벡터
            Xt = group_X(tr_ok, cols, tgt)
            qp_tr = np.column_stack([models[q].predict(Xt[shared]) for q in QUANTILES_FULL])
            qp_tr.sort(axis=1)
            sdq = qp_tr.std(axis=0)
            sdq[sdq == 0] = 1
            knn_p = NearestNeighbors(n_neighbors=200).fit(qp_tr / sdq)
            _, pidx = knn_p.kneighbors((qp_std / cap) / sdq)
            anen200_p = np.sort(lib[pidx] * cap, axis=1)

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            n = len(sub)

            def decide(qp_use, anen_use, na_arr):
                gbm300 = interp_atoms(qp_use, n=300)
                med = np.median(gbm300, axis=1)
                return build_g(gbm300, anen_use, med, cap, a_bar, na_arr)

            med_lvl = np.median(qp_std, axis=1) / cap
            terc_l = np.searchsorted(np.quantile(med_lvl, [1 / 3, 2 / 3]), med_lvl)
            na_base = np.full(n, 150)
            na_dist = 150 + np.array([NA_D[t] for t in terc_d])
            na_lvl = 150 + np.array([NA_L[t] for t in terc_l])
            na_both = np.clip(na_dist + na_lvl - 150, 100, 200)

            runs = {
                "base": (qp_std, anen200, na_base),
                "dist": (qp_std, anen200, na_dist),
                "lvl": (qp_std, anen200, na_lvl),
                "dist_lvl": (qp_std, anen200, na_both),
                "smco": (qp_cond, anen200, na_base),
                "pred_anen": (qp_std, anen200_p, na_base),
            }
            for v, (qpu, anu, nau) in runs.items():
                g = decide(qpu, anu, nau)
                s_, nm, fi, _ = metric_single(actual, g, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v50 통합 배터리 ===")
    for v in variants:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v:9s}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f}")


if __name__ == "__main__":
    main()
