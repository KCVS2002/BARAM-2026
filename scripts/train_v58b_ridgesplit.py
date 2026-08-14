"""#91b v58b: ridge 6피처 분리 절제 — perp(능선 수직 성분 4) vs fr(Froude 2).

v58에서 ridge가 경계선(+0.0015, fold 이질성 극단) → 두 메커니즘 분리로 신호원 규명.
하네스 v58 동일.
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
RIDGE_AZ = [20.0, 140.0]  # 서다리/동다리 주향 (deg, OSM 좌표 실측)


def wind_dir(dsin, dcos):
    return np.degrees(np.arctan2(dsin, dcos)) % 360


def add_phys_features(df: pd.DataFrame) -> dict:
    """티어1 피처군 추가. 반환: {그룹명: [컬럼들]}."""
    # ── veff: 밀도보정 유효풍속 ──
    rho = df["air_density"]
    fac = (rho / 1.225) ** (1 / 3)
    veff_cols = []
    for c in ("ldaps_ws50max", "gfs_ws100", "ifs_ws925"):
        df[f"veff_{c}"] = df[c] * fac
        veff_cols.append(f"veff_{c}")

    # ── ridge: 능선 수직 성분 + Froude ──
    ridge_cols = []
    for c, pre in (("ldaps_ws50max", "l50"), ("gfs_ws100", "g100"), ("gfs_ws850", "g850")):
        wd = wind_dir(df[f"{c}_dsin"], df[f"{c}_dcos"])
        for az in RIDGE_AZ:
            col = f"perp_{pre}_{int(az)}"
            df[col] = df[c] * np.abs(np.sin(np.radians(wd - az)))
            if pre != "g850":
                ridge_cols.append(col)
    # Froude: N from IFS 온위차 (t925, t850 = t925 − dt), 층후 ~650m, h=1000m
    t925 = df["ifs_t925"].to_numpy()
    t925 = np.where(t925 < 200, t925 + 273.15, t925)
    t850 = t925 - df["ifs_dt"].to_numpy()
    th925 = t925 * (1000 / 925) ** 0.286
    th850 = t850 * (1000 / 850) ** 0.286
    n2 = 9.81 / ((th850 + th925) / 2) * (th850 - th925) / 650.0
    n_bv = np.sqrt(np.clip(n2, 1e-7, None))
    for az in RIDGE_AZ:
        col = f"fr_{int(az)}"
        df[col] = np.clip(df[f"perp_g850_{int(az)}"] / (n_bv * 1000.0), 0, 10)
        ridge_cols.append(col)

    # ── hist: 비대칭(과거만) 롤링 최대 (일 블록 내 — 사이클 혼합 방지) ──
    # 재정렬 금지 (라벨·가중과의 행 정합 유지) — 시간 오름차순만 검증
    assert df["forecast_kst_dtm"].is_monotonic_increasing, "df must be time-sorted"
    hist_cols = []
    block = (df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.date
    for c in ("ldaps_ws50max", "gfs_ws100", "gfs_surface_0_gust"):
        g = df.groupby(block.values)[c]
        for w in (3, 6):
            col = f"prevmax{w}_{c}"
            df[col] = g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).max()
                                  .fillna(s.iloc[0] if len(s) else np.nan))
            # 블록 첫 시간은 이력 없음 → 자기 자신으로 대체
            df[col] = df[col].fillna(df[c])
            hist_cols.append(col)
    return {"veff": veff_cols, "ridge": ridge_cols, "hist": hist_cols}


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

    grp_cols = add_phys_features(df)
    for k, v in grp_cols.items():
        print(f"{k}: {v}", flush=True)
    perp4 = [c for c in grp_cols["ridge"] if c.startswith("perp_")]
    fr2 = [c for c in grp_cols["ridge"] if c.startswith("fr_")]
    VARIANTS = {
        "base": cols0,
        "perp": cols0 + perp4,
        "fr": cols0 + fr2,
        "ridge": cols0 + grp_cols["ridge"],
    }
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

    print("\n=== v58b ridge 분리 절제 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
