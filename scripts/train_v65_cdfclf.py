"""#99 v65 (Phase 3): 초과확률 CDF 분류 — 손실 기하 교체 문법.

분위회귀(pinball: 분위 축을 고정하고 값을 학습)의 쌍대: **값 축을 고정하고
확률을 학습** — 임계 격자 t_k별 이진 분류 P(y > t_k) → 조건부 CDF → 분위 역변환.
교차엔트로피는 pinball과 다른 손실 기하라 CDF의 다른 영역을 정확화하며,
FICR이 요구하는 것이 정확히 '임의 임계 근방의 CDF 정확도'.

공정 비교 원칙(v64 교훈): 후처리 파이프라인을 GBM 경로와 **완전 동일**하게 —
CDF 역변환으로 19분위(QUANTILES_FULL) 생성 → 정렬 → 일 평활 → interp 300.
문법(분위 생성기)만 다름.

구현: 임계 40개 (cf 0.0~0.975, 0.025 간격) × 이진 LGBM (q3 가중).
CDF 단조화 = np.maximum.accumulate. 역변환 시 끝점 (0,0)·(1,cap) 보강.
변형: base_off(앵커) / cdf300(전면 교체) / gbmcdf(gbm150+cdf150, raw) /
      pool3c(gbm150+knn75+cdf75). 하네스 v58 동일. 판정 #91 기준.
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
THRESH = np.arange(0.0, 0.9751, 0.025)  # 40개 (cf 단위)
NA_OFF = {2: 175, 1: 150, 0: 125}
VARIANTS = ["base_off", "cdf300", "gbmcdf", "pool3c"]
CLF_PARAMS = {k: v for k, v in BASE_PARAMS.items()}


def cdf_to_qp(prob_exceed, cap):
    """P(y>t_k) 행렬 (n, K) → 19분위 (n, 19), kWh."""
    F = 1.0 - prob_exceed  # CDF at THRESH
    F = np.maximum.accumulate(np.clip(F, 0, 1), axis=1)  # 단조화
    # 끝점 보강: t=0 이전 F=0 (y>=0), t=1(cap)에서 F=1
    tgrid = np.concatenate([[0.0], THRESH, [1.0]])
    Fx = np.concatenate([np.zeros((len(F), 1)), F, np.ones((len(F), 1))], axis=1)
    qp = np.empty((len(F), len(QUANTILES_FULL)))
    for i in range(len(F)):
        qp[i] = np.interp(QUANTILES_FULL, Fx[i], tgrid)
    return np.clip(qp * cap, 0, cap)


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
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        clfs = []
        yv = y.to_numpy()
        for t_ in THRESH:
            lbl = (yv > t_).astype(int)
            if lbl.min() == lbl.max():  # 단일 클래스 방어
                clfs.append(float(lbl[0]))
                continue
            c = lgb.LGBMClassifier(objective="binary", **CLF_PARAMS)
            c.fit(X[shared], lbl, sample_weight=w)
            clfs.append(c)
        print(f"fold {fold}: 학습 완료 (분위 19 + 분류 {len(THRESH)}) ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm300 = interp_atoms(qp, n=300)
            gbm150 = gbm300[:, np.linspace(0, 299, 150).astype(int)]
            med = np.median(gbm300, axis=1)

            # CDF 문법 — GBM 경로와 동일 후처리
            pe = np.column_stack([
                (np.full(len(sub), c) if isinstance(c, float)
                 else c.predict_proba(Xv[shared])[:, 1]) for c in clfs])
            qp_cdf = cdf_to_qp(pe, cap)
            qp_cdf.sort(axis=1)
            qp_cdf = np.sort(smooth_quantiles_by_day(qp_cdf, sub["forecast_kst_dtm"]), axis=1)
            cdf300 = interp_atoms(qp_cdf, n=300)
            cdf150 = cdf300[:, np.linspace(0, 299, 150).astype(int)]
            cdf75 = cdf300[:, np.linspace(0, 299, 75).astype(int)]

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(
                ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            knn150 = np.sort(anen200[:, np.linspace(0, 199, 150).astype(int)], axis=1)
            knn150c = np.clip(knn150 + (med - np.median(knn150, axis=1))[:, None], 0, cap)
            knn75 = knn150c[:, np.linspace(0, 149, 75).astype(int)]

            d_bar = dist[:, :150].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            base_atoms = np.empty((len(sub), 300))
            for i in range(len(sub)):
                na = NA_OFF[terc[i]]
                ng = 300 - na
                an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                gb = np.interp(np.linspace(0, 299, ng), np.arange(300), gbm300[i])
                base_atoms[i] = np.sort(np.concatenate([an, gb]))

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            atoms_by = {
                "base_off": base_atoms,
                "cdf300": cdf300,
                "gbmcdf": np.sort(np.concatenate([gbm150, cdf150], axis=1), axis=1),
                "pool3c": np.sort(np.concatenate([gbm150, knn75, cdf75], axis=1), axis=1),
            }
            for v in VARIANTS:
                pred = optimize_submission(atoms_by[v], cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v65 초과확률 CDF 분류 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in VARIANTS:
        fm = np.mean(list(res[v].values()))
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        ps = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_)
        print(f"{v:8s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
