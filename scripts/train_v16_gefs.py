"""v16: GEFS 앙상블 스프레드 기반 시나리오 원자 (예측시점 피처 치환 — 학습 불변).

- GEFS 평균·스프레드(10m u/v, 3h) → 시간 보간 → 상대 풍속 불확실성 r_sigma
- 시나리오 z ∈ {-1.5,-1,-0.5,0,+0.5,+1,+1.5}: 모든 풍속류 피처 × (1 + z·r_sigma), wpd × (…)³
- 각 시나리오에서 5개 분위(q.1/.3/.5/.7/.9) 예측 → 35 원자 → 재정렬 후 sub_009 원자에 결합
- 누수 근거: D-1 00Z 사이클 (제공 GFS와 동일 사이클·동일 가용성 규약, 규칙 예시의 기준시점 14:00)

기준: sub_009 구성 0.6470.
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
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
Q_SCEN = [0.1, 0.3, 0.5, 0.7, 0.9]
Z_SCEN = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
WIND_SPEED_COLS_PREFIX = ("ldaps_ws", "gfs_ws")  # 스칼라 풍속 피처
CUBE_COLS = ("wpd_100m", "wpd_ldaps50")
GUST_COLS = ("gfs_surface_0_gust",)


def load_gefs_rsigma(project):
    g = pd.read_csv(project / "external_data" / "gefs" / "gefs_stats_2024_2025.csv", encoding="utf-8-sig")
    g["run_dt"] = pd.to_datetime(g["run"].astype(str), format="%Y%m%d%H")
    g["forecast_kst_dtm"] = g.run_dt + pd.Timedelta(hours=9) + pd.to_timedelta(g.fxx, unit="h")
    piv = g.pivot_table(index="forecast_kst_dtm", columns=["member", "var"], values="value", aggfunc="mean")
    u_m, v_m = piv[("avg", "u10")], piv[("avg", "v10")]
    u_s, v_s = piv[("spr", "u10")], piv[("spr", "v10")]
    ws = np.sqrt(u_m**2 + v_m**2).clip(lower=0.3)
    sigma_ws = np.sqrt((u_m * u_s)**2 + (v_m * v_s)**2) / ws
    r = (sigma_ws / ws).clip(0.02, 0.6).rename("gefs_rsigma")
    out = r.reset_index()
    # 3h → 1h 보간
    full = pd.DataFrame({"forecast_kst_dtm": pd.date_range(out.forecast_kst_dtm.min(),
                                                           out.forecast_kst_dtm.max(), freq="h")})
    out = full.merge(out, on="forecast_kst_dtm", how="left")
    out["gefs_rsigma"] = out["gefs_rsigma"].interpolate(limit=3)
    return out


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    rsig = load_gefs_rsigma(PROJECT)
    df = df.merge(rsig, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ws_cols = [c for c in base_cols if c.startswith(WIND_SPEED_COLS_PREFIX)
               and not c.endswith(("_dsin", "_dcos"))] + list(GUST_COLS)
    print(f"ready: rsigma cov(2024)={df[df.forecast_kst_dtm.dt.year==2024].gefs_rsigma.notna().mean()*100:.0f}% "
          f"({time.time()-t0:.0f}s)")

    res = {}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        X, y, w = stack_groups(tr, base_cols, w_tr)
        shared = base_cols + ["g_rated", "g_rotor", "g_id"]
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}

        fs = []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm = interp_atoms(qp, n=150)
            med = np.median(gbm, axis=1)

            # AnEn
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)

            # GEFS 시나리오 원자
            rs = sub["gefs_rsigma"].fillna(sub["gefs_rsigma"].median()).to_numpy()
            scen_atoms = []
            for z in Z_SCEN:
                factor = np.clip(1 + z * rs, 0.4, 1.8)
                Xs = Xv.copy()
                for c in ws_cols:
                    Xs[c] = Xs[c] * factor
                for c in CUBE_COLS:
                    if c in Xs:
                        Xs[c] = Xs[c] * factor**3
                for q in Q_SCEN:
                    scen_atoms.append(np.clip(models[q].predict(Xs[shared]) * cap, 0, cap))
            scen = np.sort(np.column_stack(scen_atoms), axis=1)  # (n, 35)
            scen = np.clip(scen + (med - np.median(scen, axis=1))[:, None], 0, cap)
            scen = np.repeat(scen, 4, axis=1)  # 140 원자로 증폭 (GBM/AnEn과 균형)

            atoms = np.sort(np.concatenate([gbm, anen, scen], axis=1), axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(atoms, cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            fs.append(s)
        res[fold] = np.nanmean(fs)
        print(f"fold {fold}: {res[fold]:.4f} ({time.time()-t0:.0f}s)")

    print(f"\n=== v16 GEFS 시나리오: {np.mean(list(res.values())):.4f} (기준 0.6470) ===")


if __name__ == "__main__":
    main()
