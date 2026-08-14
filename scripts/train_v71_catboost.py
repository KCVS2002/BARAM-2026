"""#109 v71 (구상 B): 이종 모델 분위 블렌딩 — CatBoost MultiQuantile.

원자 생성기의 함수 클래스 다양화. 현행은 LGBM 단일 가족 19분위 —
동일 정보·동일 가중(q3)·동일 하류에서 CatBoost(대칭 트리, 순서형 부스팅)를
추가해 조건부 오차 자체의 감소(유일하게 남은 경로, #61)를 노린다.
#95(스태킹)와의 차이: 2024 적합 메타 결합기 없음(#54 비저촉), 정보 부분집합
아닌 동일 전체 피처(#68 희석 비저촉), 위치 평균이라 폭 불변(#61 비저촉).

변형 (GBM 쪽 원자만 교체, AnEn·NA_OFF 재배분·결정층 동일):
  base_off : 현행 (LGBM gbm300)                     — 기준선
  cbrep    : CatBoost cb300 전면 교체               — 단독 품질 진단
  qavg     : 분위 곡선 50/50 평균 (같은 분위끼리)    — 위치 평균 결합
  half     : lgb150+cb150 원자 풀링                 — 분포 풀링 결합
진단: fold별 q50 오차 상관 (다양성의 실질 크기).
하네스 v66 동일 (6-fold, q3 가중, 풀링 채점 병기). 판정 #91 기준.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
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
NA_OFF = {2: 175, 1: 150, 0: 125}
VARIANTS = ["base_off", "cbrep", "qavg", "half"]
CB_PARAMS = dict(
    loss_function="MultiQuantile:alpha=" + ",".join(str(q) for q in QUANTILES_FULL),
    iterations=1200, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    random_seed=42, verbose=0, allow_writing_files=False,
)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    w_q3 = clean_w.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        w_q3[tgt] = w_q3[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
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
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    folds = list(range(1, 12, 2))
    for fi_, m0 in enumerate(folds):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: LGBM 완료 ({time.time()-t0:.0f}s)", flush=True)
        cb = CatBoostRegressor(**CB_PARAMS).fit(X[shared], y, sample_weight=w)
        print(f"fold {fold}: CatBoost 완료 ({time.time()-t0:.0f}s) "
              f"[진행 {fi_+1}/{len(folds)} fold, {(fi_+1)/len(folds)*100:.0f}%]", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        err_corr = []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qp_lgb = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
            qp_cb = np.clip(cb.predict(Xv[shared]) * cap, 0, cap)

            actual = sub[tgt].to_numpy()
            e_l = np.sort(qp_lgb, axis=1)[:, 9] - actual
            e_c = np.sort(qp_cb, axis=1)[:, 9] - actual
            err_corr.append(np.corrcoef(e_l, e_c)[0, 1])

            def to300(qp_raw):
                qp_ = np.sort(qp_raw, axis=1)
                qp_ = np.sort(smooth_quantiles_by_day(qp_, sub["forecast_kst_dtm"]), axis=1)
                return interp_atoms(qp_, n=300)

            g300 = {"base_off": to300(qp_lgb), "cbrep": to300(qp_cb),
                    "qavg": to300((np.sort(qp_lgb, axis=1) + np.sort(qp_cb, axis=1)) / 2)}
            lgb150 = g300["base_off"][:, np.linspace(0, 299, 150).astype(int)]
            cb150 = g300["cbrep"][:, np.linspace(0, 299, 150).astype(int)]
            g300["half"] = np.sort(np.concatenate([lgb150, cb150], axis=1), axis=1)

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(
                ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            d_bar = dist[:, :150].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            for v in VARIANTS:
                med = np.median(g300[v], axis=1)
                atoms = np.empty((len(sub), 300))
                for i in range(len(sub)):
                    na = NA_OFF[terc[i]]
                    ng = 300 - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 299, ng), np.arange(300), g300[v][i])
                    atoms[i] = np.sort(np.concatenate([an, gb]))
                pred = optimize_submission(atoms, cap, a_bar)
                s_, nm, fi2, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi2))
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" | q50오차상관 {np.mean(err_corr):.3f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v71 CatBoost 블렌딩 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in VARIANTS:
        fm = np.mean(list(res[v].values()))
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi2 = np.mean([dec[v][f][1] for f in res[v]])
        ps = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_)
        print(f"{v:8s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi2:.4f} | 풀링 {np.mean(ps):.4f}")
    for v in VARIANTS:
        print(f"{v:8s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))


if __name__ == "__main__":
    main()
