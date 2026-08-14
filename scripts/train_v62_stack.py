"""#95 v62: 소스별 독립 모델 스태킹 (티어3b 후반부, HEFTCom2024 우승 구조 번역).

현행 = 3개 NWP를 피처 병합한 단일 모델. HEFTCom 우승 = 소스별 독립 모델 → 메타 결합
(pinball -8% 보고). 우리식 검증:
- ld/gf/if: 단일 소스 모델 (진단 — 각 소스의 단독 실력)
- stack: 분위별 NNLS 메타 (y ~ [p_ld, p_gf, p_if], train 내 적합, 저용량 3계수)
- pool : merged 150원자 + 소스별 50원자×3 (sister 문법 — 정보 부분집합의 관점 다양성)
리스크 인지: #68(같은 정보 풀링=희석)·#37(평균=선명화)와 충돌 가능 — 소스 분리가
'다른 정보'로 기능하는지가 관건. 하네스 v58 동일 (6-fold q3, 풀링 병기).
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
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
CAL = ["hour_sin", "hour_cos", "month_sin", "month_cos", "lead_h"]


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
    cols0 = base_cols + ifs10
    ld_cols = [c for c in base_cols if c.startswith("ldaps_")] + CAL
    gf_cols = [c for c in base_cols if c.startswith("gfs_")] + \
        ["air_density", "wpd_100m", "shear_100_10", "shear_850_100"] + CAL
    if_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "ifs_t925", "ifs_dt",
               "ifs_ws700", "ifs_shear", "ifs_q850"] + CAL
    SETS = {"merged": cols0, "ld": ld_cols, "gf": gf_cols, "if": if_cols}
    variants = ["base", "ld", "gf", "if", "stack", "pool"]
    print(f"피처 수: " + " ".join(f"{k}={len(v)}" for k, v in SETS.items()), flush=True)
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in variants}
    dec = {v: {} for v in variants}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        models = {}
        for name, cols in SETS.items():
            X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[name] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        # 메타 계수: train 내 소스 예측 → NNLS (분위별·전 그룹 공유)
        meta_w = {}
        Xm, ym, _ = stack_groups(tr, cols0, w_q3[tr_idx.to_numpy()])
        for q in QUANTILES_FULL:
            P = np.column_stack([
                models[m][q].predict(Xm[SETS[m] + ["g_rated", "g_rotor", "g_id"]])
                for m in ("ld", "gf", "if")])
            meta_w[q], _ = nnls(P, ym.to_numpy())
        print(f"fold {fold}: 학습 완료, meta(q50)={np.round(meta_w[0.50], 2)} "
              f"({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        dc = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            qp_src = {}
            for name, cols in SETS.items():
                Xv = group_X(sub, cols, tgt)
                shared = cols + ["g_rated", "g_rotor", "g_id"]
                qp = np.column_stack([np.clip(models[name][q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qp_src[name] = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            # stack: 분위별 메타 결합 (평활 전 원예측이 이상적이나 근사로 평활 후 결합)
            qp_stack = np.column_stack([
                np.clip(sum(meta_w[q][k] * qp_src[m][:, i] for k, m in enumerate(("ld", "gf", "if"))),
                        0, cap)
                for i, q in enumerate(QUANTILES_FULL)])
            qp_stack.sort(axis=1)

            def decide(qp_or_atoms, is_atoms=False):
                if not is_atoms:
                    gbm = interp_atoms(qp_or_atoms, n=150)
                else:
                    gbm = qp_or_atoms
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                return optimize_submission(atoms, cap, a_bar)

            preds = {"base": decide(qp_src["merged"]), "ld": decide(qp_src["ld"]),
                     "gf": decide(qp_src["gf"]), "if": decide(qp_src["if"]),
                     "stack": decide(qp_stack)}
            pool_atoms = np.concatenate(
                [interp_atoms(qp_src["merged"], n=150)] +
                [interp_atoms(qp_src[m], n=50) for m in ("ld", "gf", "if")], axis=1)
            preds["pool"] = decide(np.sort(pool_atoms, axis=1), is_atoms=True)

            for v in variants:
                s_, nm, fi, _ = metric_single(actual, preds[v], cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(preds[v])
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v62 소스별 스태킹 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in variants:
        fm = np.mean(list(res[v].values()))
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        ps = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_)
        print(f"{v:6s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
