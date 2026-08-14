"""#100 v66 (Phase 4): GBM-EMOS — 모수적 분포회귀 (이분산 절단정규).

분포문법 탐색의 반대극(모수형): 조건부 분포를 N(μ(x), σ(x)²)의 [0,1] 절단으로
가정. μ = L2 LGBM (q3 가중). σ = **전체 피처 조건부** 잔차크기 LGBM
(log|r| 회귀, 시간순 3블록 준-OOF 잔차 — v64의 1차원 뭉갬과 달리 조건화 완전).
σ̂ = exp(pred)·√(π/2) (E|r|→σ 해석적 변환 — 적합 상수 0개).
문헌 근거: gradient-boosting EMOS는 Schulz&Lerch 2022 비교에서 EMOS < GB-EMOS.
모수 형태 제약 = 최강 정칙화 (원칙 3의 극단) — 단 무릎 구간 형태 왜곡 리스크.

후처리는 GBM 경로와 동일 (19분위 → 정렬 → 일 평활 → interp300, v64 교훈).
변형: base_off / par300(전면 교체) / gbmpar(gbm150+par150) / pool3p(+knn75+par75).
하네스 v58 동일. 판정 #91 기준. Phase 1~3 전멸 후 반대극 확인용 최종 실험.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
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
VARIANTS = ["base_off", "par300", "gbmpar", "pool3p"]
QARR = np.array(QUANTILES_FULL)


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
        mu_model = lgb.LGBMRegressor(objective="regression_l1", **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w)

        # 시간순 3블록 준-OOF 잔차 → σ 모델 (전체 피처 조건화, 무가중)
        tmin = tr.forecast_kst_dtm.min()
        span = va_start - tmin
        edges = [tmin + span * f for f in (0.4, 0.6, 0.8)] + [va_start]
        Xr, yr = [], []
        for b in range(3):
            m_idx = df.forecast_kst_dtm < edges[b]
            blk = df[(df.forecast_kst_dtm >= edges[b]) & (df.forecast_kst_dtm < edges[b + 1])]
            Xb_all, yb_all, wb_all = stack_groups(df[m_idx], cols, w_q3[m_idx.to_numpy()])
            mub = lgb.LGBMRegressor(objective="regression_l1", **BASE_PARAMS).fit(
                Xb_all[shared], yb_all, sample_weight=wb_all)
            for tgt in TARGET_COLS:
                cap = CAPACITY_KWH[tgt]
                sub_b = blk[blk[tgt].notna()]
                if not len(sub_b):
                    continue
                Xb = group_X(sub_b, cols, tgt)
                p = np.clip(mub.predict(Xb[shared]), 0, 1)
                r = sub_b[tgt].to_numpy() / cap - p
                Xr.append(Xb[shared])
                yr.append(np.log(np.abs(r) + 1e-3))
        Xr = pd.concat(Xr)
        yr = np.concatenate(yr)
        sig_model = lgb.LGBMRegressor(objective="regression_l2", **BASE_PARAMS).fit(Xr, yr)
        print(f"fold {fold}: 학습 완료 (σ풀 {len(yr)}행) ({time.time()-t0:.0f}s)", flush=True)

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

            # 모수 분포: 절단정규 N(μ, σ²)|[0,1]
            mu = np.clip(mu_model.predict(Xv[shared]), 0, 1)
            sig = np.exp(sig_model.predict(Xv[shared])) * np.sqrt(np.pi / 2)
            sig = np.clip(sig, 0.005, 0.6)
            a_ = (0.0 - mu) / sig
            b_ = (1.0 - mu) / sig
            qp_par = np.empty((len(sub), 19))
            for i in range(len(sub)):
                qp_par[i] = truncnorm.ppf(QARR, a_[i], b_[i], loc=mu[i], scale=sig[i])
            qp_par = np.clip(qp_par * cap, 0, cap)
            qp_par.sort(axis=1)
            qp_par = np.sort(smooth_quantiles_by_day(qp_par, sub["forecast_kst_dtm"]), axis=1)
            par300 = interp_atoms(qp_par, n=300)
            par150 = par300[:, np.linspace(0, 299, 150).astype(int)]
            par75 = par300[:, np.linspace(0, 299, 75).astype(int)]

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
                "par300": par300,
                "gbmpar": np.sort(np.concatenate([gbm150, par150], axis=1), axis=1),
                "pool3p": np.sort(np.concatenate([gbm150, knn75, par75], axis=1), axis=1),
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

    print("\n=== v66 GBM-EMOS (절단정규) — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
