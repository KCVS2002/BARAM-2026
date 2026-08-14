"""#98 v64 (Phase 2): 점예측 + 조건부 잔차 dressing — 분포 구성의 분해 문법.

분위회귀(분포 직접 학습)의 대안: **점예측 모델과 오차 분포 모델을 분리**.
- 점예측: L1 LGBM (cf, q3 가중) — 조건부 중앙값의 단일 최강 추정.
- 오차 모델: 학습 구간 내 시간순 3-블록 준-OOF 잔차 수집 → 예측 수준(p̂ 분위
  구간, 자기적응) 조건부 경험 잔차 분포 → p̂에 입혀 원자 생성 (dressing).
근거: ①잔차 풀링은 계절·레짐 횡단의 강한 정칙화 = 2025 전이 방어선(#61 이론)과
정합 ②#79의 이분산(고출력 오차 집중)을 분위회귀와 다른 방식으로 명시 포착.
주의: 조건 구간은 잔차 풀의 p̂ 분위 기준(자기적응) — 절대 수준 기준 금지(#82).

변형 (결정층 고정):
  base_off : 현행 공식 (앵커)
  drs300   : dressing 단독 300원자 (문법 전면 교체)
  gbmdrs   : gbm150 + dress150 (AnEn 역할 교체; dress는 raw — 자체 수준 유지,
             sister 전례: raw > shift)
  pool3d   : gbm150 + knn75 + dress75 (3원 병행)
  drs10    : drs300의 조건 구간 5→10 (용량-반응)
하네스 v58 동일 (6-fold q3, 풀링 병기). 판정 #91 기준.
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
NA_OFF = {2: 175, 1: 150, 0: 125}
VARIANTS = ["base_off", "drs300", "gbmdrs", "pool3d", "drs10"]


def fit_point(tr, cols, w_tr):
    X, y, w = stack_groups(tr, cols, w_tr)
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    m = lgb.LGBMRegressor(objective="regression_l1", **BASE_PARAMS)
    m.fit(X[shared], y, sample_weight=w)
    return m


def collect_residuals(df, cols, w_all, va_start):
    """학습 구간 내 시간순 3-블록 준-OOF 잔차 (그룹 공유 풀).

    반환: p_hat(예측 cf), resid(y-p̂, cf 단위) — 블록별 이전 데이터로만 학습.
    """
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    tr_all = df[df.forecast_kst_dtm < va_start]
    tmin = tr_all.forecast_kst_dtm.min()
    span = va_start - tmin
    edges = [tmin + span * f for f in (0.4, 0.6, 0.8)] + [va_start]
    ps, rs = [], []
    for b in range(3):
        m_idx = df.forecast_kst_dtm < edges[b]
        blk = df[(df.forecast_kst_dtm >= edges[b]) & (df.forecast_kst_dtm < edges[b + 1])]
        model = fit_point(df[m_idx], cols, w_all[m_idx.to_numpy()])
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            sub = blk[blk[tgt].notna()]
            if not len(sub):
                continue
            Xb = group_X(sub, cols, tgt)
            p = np.clip(model.predict(Xb[shared]), 0, 1)
            ps.append(p)
            rs.append(sub[tgt].to_numpy() / cap - p)
    return np.concatenate(ps), np.concatenate(rs)


def dress_atoms(p_va, p_pool, r_pool, cap, n_bins=5, n_atoms=300):
    """p̂ 분위 구간(자기적응) 조건부 잔차 분포를 p̂에 입힘."""
    edges = np.quantile(p_pool, np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_pool = np.searchsorted(edges, p_pool)
    bin_va = np.searchsorted(edges, p_va)
    qs = (np.arange(n_atoms) + 0.5) / n_atoms
    rq = {b: np.quantile(r_pool[bin_pool == b], qs) for b in range(n_bins)}
    out = np.empty((len(p_va), n_atoms))
    for i in range(len(p_va)):
        out[i] = np.clip((p_va[i] + rq[bin_va[i]]) * cap, 0, cap)
    return np.sort(out, axis=1)


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
        point = fit_point(tr, cols, w_q3[tr_idx.to_numpy()])
        p_pool, r_pool = collect_residuals(df, cols, w_q3, va_start)
        print(f"fold {fold}: 학습 완료 (잔차 풀 {len(r_pool)}행) ({time.time()-t0:.0f}s)", flush=True)

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

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(
                ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            knn150 = np.sort(anen200[:, np.linspace(0, 199, 150).astype(int)], axis=1)
            knn150c = np.clip(knn150 + (med - np.median(knn150, axis=1))[:, None], 0, cap)

            p_va = np.clip(point.predict(Xv[shared]), 0, 1)
            drs300 = dress_atoms(p_va, p_pool, r_pool, cap, n_bins=5, n_atoms=300)
            drs10 = dress_atoms(p_va, p_pool, r_pool, cap, n_bins=10, n_atoms=300)
            drs150 = drs300[:, np.linspace(0, 299, 150).astype(int)]
            drs75 = drs300[:, np.linspace(0, 299, 75).astype(int)]
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
                "drs300": drs300,
                "gbmdrs": np.sort(np.concatenate([gbm150, drs150], axis=1), axis=1),
                "pool3d": np.sort(np.concatenate([gbm150, knn75, drs75], axis=1), axis=1),
                "drs10": drs10,
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

    print("\n=== v64 잔차 dressing — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
