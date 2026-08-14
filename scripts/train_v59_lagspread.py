"""#92 v59: GFS D-2 사이클 lagged/spread 피처 (티어2, domain_theory.md §2).

수집(collect_gfs_lag.py): NOAA S3 GFS D-2 00UTC 사이클 f040~f063, 100m u/v + gust,
9격자, 2022~2025 전 기간 (실패 7슬롯/35,064). 발표 D-2 오후 — 누수 안전.

피처 (스칼라 우선 원칙: 격자별 ws 계산 후 평균):
- lag: d2_ws100 (D-2 예보 풍속), d2_gust — 다른 사이클의 관점 (lagged ensemble)
- spr: spr_ws100 = gfs_ws100 − d2_ws100 (부호), spr_abs = |·|, spr_gust,
       spr_ddir = D-1·D-2 풍향 각차 — 사이클 간 예보 변동성 = 불확실성 신호
변형: base / lag / spr / both. 하네스 v58 동일 (6-fold q3, 풀링 병기).
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


def load_gfs_d2() -> pd.DataFrame:
    d = pd.read_csv(PROJECT / "external_data/noaa_gfs/gfs_lag_d2_20220101_20251231.csv",
                    encoding="utf-8-sig")
    wide = d.pivot_table(index=["run", "fxx", "lat", "lon"], columns="var",
                         values="value").reset_index()
    wide["ws"] = np.hypot(wide["u100"], wide["v100"])
    grp = wide.groupby(["run", "fxx"]).agg(
        d2_ws100=("ws", "mean"), d2_gust=("gust", "mean"),
        d2_u=("u100", "mean"), d2_v=("v100", "mean")).reset_index()
    run_dt = pd.to_datetime(grp["run"].astype(str), format="%Y%m%d%H")
    grp["forecast_kst_dtm"] = run_dt + pd.to_timedelta(grp["fxx"] + 9, unit="h")
    return grp[["forecast_kst_dtm", "d2_ws100", "d2_gust", "d2_u", "d2_v"]]


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

    d2 = load_gfs_d2()
    df = df.merge(d2, on="forecast_kst_dtm", how="left")
    cov = df[["d2_ws100"]].notna().mean().iloc[0]
    print(f"D-2 커버리지: {cov*100:.2f}%", flush=True)
    df["spr_ws100"] = df["gfs_ws100"] - df["d2_ws100"]
    df["spr_abs"] = df["spr_ws100"].abs()
    df["spr_gust"] = df["gfs_surface_0_gust"] - df["d2_gust"]
    wd1 = np.degrees(np.arctan2(df["gfs_ws100_dsin"], df["gfs_ws100_dcos"])) % 360
    wd2 = (np.degrees(np.arctan2(-df["d2_u"], -df["d2_v"]))) % 360
    dd = np.abs(wd1 - wd2)
    df["spr_ddir"] = np.minimum(dd, 360 - dd)

    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols0 = base_cols + ifs10
    lag_cols = ["d2_ws100", "d2_gust"]
    spr_cols = ["spr_ws100", "spr_abs", "spr_gust", "spr_ddir"]
    VARIANTS = {"base": cols0, "lag": cols0 + lag_cols, "spr": cols0 + spr_cols,
                "both": cols0 + lag_cols + spr_cols}
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    variants = list(VARIANTS)
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
        for v in variants:
            cols = VARIANTS[v]
            X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[v] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

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

            for v in variants:
                cols = VARIANTS[v]
                Xv = group_X(sub, cols, tgt)
                shared = cols + ["g_rated", "g_rotor", "g_id"]
                qp = np.column_stack([np.clip(models[v][q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm = interp_atoms(qp, n=150)
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v59 GFS D-2 lagged/spread — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:5s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
