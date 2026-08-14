"""v40: LDAPS 1.5km 안정도 피처 — 기압면 기온 프로파일 (지형 해상 안정도).

동기 (#66): FICR을 움직인 유일한 실증 = 안정도 정보 (IFS t925·dT, 0.25°).
LDAPS는 1.5km로 가덕산 능선을 실제 해상 — 같은 물리의 상위 해상도판.
제공 LDAPS에 기압면 기온 없음(지상 2m뿐) → 미개척 정보. #21(바람 기각)과 구별:
바람은 ldaps_ws50max와 중복이었지만 기온 구조는 신규.

피처 (875hPa ≈ 허브고도층; 900 이하는 지형 채움이라 미사용):
- kma_t875: 허브고도층 기온 (계절·밀도·레짐)
- kma_dt875_850: 얇은 층 기온경사 (IFS ifs_dt의 1.5km판)
- kma_dt875_700: 깊은 층 경사 (자유대기 안정도)
변형: base(공식 feat10) / k2(t875+dt875_850) / k3(k2+dt875_700).
기준: CV 0.6591. 하네스 v31b 동일. 판정 기준: 명확한 CV 이득 (6-fold 방향 일관).
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
KMA_T_CSV = PROJECT / "external_data" / "kma_ldaps" / "point_profile_t.csv"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_kma_t() -> pd.DataFrame:
    d = pd.read_csv(KMA_T_CSV, encoding="utf-8-sig", dtype={"tmfc": str})
    d = d[d.level_pa.isin([87500, 85000, 70000])]
    d = d.drop_duplicates(subset=["tmfc", "ef", "level_pa"])
    d["forecast_kst_dtm"] = (pd.to_datetime(d.tmfc, format="%Y%m%d%H")
                             + pd.Timedelta(hours=9) + pd.to_timedelta(d.ef, unit="h"))
    piv = d.pivot_table(index="forecast_kst_dtm", columns="level_pa", values="value")
    out = pd.DataFrame(index=piv.index)
    out["kma_t875"] = piv[87500]
    out["kma_dt875_850"] = piv[87500] - piv[85000]
    out["kma_dt875_700"] = piv[87500] - piv[70000]
    out.index.name = "forecast_kst_dtm"
    return out.reset_index()


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
    df = df.merge(load_kma_t(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    adds = {
        "base": [],
        "k2": ["kma_t875", "kma_dt875_850"],
        "k3": ["kma_t875", "kma_dt875_850", "kma_dt875_700"],
    }
    cov24 = df.loc[df.forecast_kst_dtm.dt.year == 2024, adds["k3"]].notna().mean()
    print("2024 커버리지:\n" + cov24.to_string(), flush=True)
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    variants = list(adds)
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

        models = {}
        cols_v = {}
        for v in variants:
            cols = base_cols + ifs10 + adds[v]
            X, y, w = stack_groups(tr, cols, w_tr)
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[v] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
            cols_v[v] = (cols, shared)
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
                cols, shared = cols_v[v]
                Xv = group_X(sub, cols, tgt)
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
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v40 LDAPS 안정도 ===")
    for v in variants:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v:4s}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f}")


if __name__ == "__main__":
    main()
