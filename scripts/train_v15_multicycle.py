"""v15: 멀티사이클(D-2 12Z vs D-1 00Z) 시나리오 원자.

- 두 사이클의 875hPa 풍속 비율 f = ws_d2/ws_d1 = "실물 역학 시나리오" (GEFS 균일배율과 달리
  같은 모델의 독립 초기화가 만든 진짜 대안 흐름)
- 시나리오: f, sqrt(f) 두 단계 → 풍속류 피처 치환 → 5분위 예측 → 원자 결합 (학습 불변)
- D-2는 대상일 01~21시만 커버 (LDAPS +48h 한계) → 미커버 행은 f=1 (시나리오 없음)

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
KMA_DIR = PROJECT / "external_data" / "kma_ldaps"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
Q_SCEN = [0.1, 0.3, 0.5, 0.7, 0.9]
GUST_COLS = ("gfs_surface_0_gust",)
CUBE_COLS = ("wpd_100m", "wpd_ldaps50")


def load_ws875(paths, tmfc_hour_utc):
    frames = []
    for p in paths:
        if Path(p).exists():
            frames.append(pd.read_csv(p, encoding="utf-8-sig", dtype={"tmfc": str}))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["tmfc", "ef", "varn", "level_pa"])
    df = df[df.level_pa == 87500]
    df["forecast_kst_dtm"] = (pd.to_datetime(df.tmfc, format="%Y%m%d%H")
                              + pd.Timedelta(hours=9) + pd.to_timedelta(df.ef, unit="h"))
    piv = df.pivot_table(index="forecast_kst_dtm", columns="varn", values="value")
    ws = np.sqrt(piv[2002] ** 2 + piv[2003] ** 2)
    return ws


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")

    ws_d1 = load_ws875([KMA_DIR / "point_profile.csv"], 0).rename("ws_d1")
    ws_d2 = load_ws875([KMA_DIR / f"point_profile_d2_w{w}.csv" for w in range(3)]
                       + [KMA_DIR / "point_profile_d2.csv"], 12).rename("ws_d2")
    cyc = pd.concat([ws_d1, ws_d2], axis=1).reset_index()
    cyc["cycle_factor"] = (cyc.ws_d2 / cyc.ws_d1.clip(lower=0.5)).clip(0.5, 1.8)
    df = df.merge(cyc[["forecast_kst_dtm", "cycle_factor"]], on="forecast_kst_dtm", how="left")
    df["cycle_factor"] = df["cycle_factor"].fillna(1.0)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ws_cols = [c for c in base_cols if c.startswith(("ldaps_ws", "gfs_ws"))
               and not c.endswith(("_dsin", "_dcos"))] + list(GUST_COLS)
    cov = (df.cycle_factor != 1.0).mean()
    print(f"ready: cycle_factor 유효비율 {cov*100:.0f}% ({time.time()-t0:.0f}s)")

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

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)

            f_full = sub["cycle_factor"].to_numpy()
            scen_atoms = []
            for f_use in (f_full, np.sqrt(f_full)):
                Xs = Xv.copy()
                for c in ws_cols:
                    Xs[c] = Xs[c] * f_use
                for c in CUBE_COLS:
                    if c in Xs:
                        Xs[c] = Xs[c] * f_use**3
                for q in Q_SCEN:
                    scen_atoms.append(np.clip(models[q].predict(Xs[shared]) * cap, 0, cap))
            scen = np.sort(np.column_stack(scen_atoms), axis=1)
            scen = np.clip(scen + (med - np.median(scen, axis=1))[:, None], 0, cap)
            scen = np.repeat(scen, 15, axis=1)  # 150 원자로 균형

            atoms = np.sort(np.concatenate([gbm, anen, scen], axis=1), axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(atoms, cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            fs.append(s)
        res[fold] = np.nanmean(fs)
        print(f"fold {fold}: {res[fold]:.4f} ({time.time()-t0:.0f}s)")

    print(f"\n=== v15 멀티사이클: {np.mean(list(res.values())):.4f} (기준 0.6470) ===")


if __name__ == "__main__":
    main()
