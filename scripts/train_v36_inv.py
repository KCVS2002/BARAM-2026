"""v36: 안정도 3탄 — 접지역전(ifs_inv = t925 − 2t) + 미사용 q925 절제 실험.

#66에서 안정도(t925·dT)가 FICR을 실제로 올린다는 게 LB 실증됨 (전이율 ~1/3.6).
같은 물리 축의 남은 전 기간 가용 변수 2개를 시험:
- ifs_inv: t925(자유대기 ~750m) − 2t(모델 지형 지상) — 접지역전/야간 냉각 풀.
  dT(925-850, 상층 경사)와 상보적인 **하층 경계층 안정도**.
- ifs_q925: wave-2에 수집돼 있으나 v31b에서 미사용 (q850만 채택). 허브고도층 습도.

변형 (base = 공식 피처셋 base124+IFS10, CV 0.6591):
- inv : + ifs_inv
- q925: + ifs_q925
- both: + 둘 다
판정 기준: 명확한 CV 이득 없으면 기각 (피처 절제 원칙). 하네스 v31b 동일.
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
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
IFS2_CSV = PROJECT / "external_data" / "ecmwf_ifs" / "ifs_point2_2022_2025.csv"
IFS3_CSV = PROJECT / "external_data" / "ecmwf_ifs" / "ifs_point3_2022_2025.csv"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_ifs2_features_ext() -> pd.DataFrame:
    """wave-2 물리 피처 (v31b 5종) + q925 추가판."""
    d = pd.read_csv(IFS2_CSV, encoding="utf-8-sig")
    d = d.drop_duplicates(subset=["run", "fxx", "var", "lat", "lon"])
    d["forecast_kst_dtm"] = (pd.to_datetime(d.run) + pd.Timedelta(hours=9)
                             + pd.to_timedelta(d.fxx, unit="h"))
    piv = d.pivot_table(index=["forecast_kst_dtm", "lat", "lon"], columns="var", values="value")
    out = pd.DataFrame(index=piv.index.get_level_values(0).unique().sort_values())
    grp = piv.groupby(level="forecast_kst_dtm")
    out["ifs_t925"] = grp["t925"].mean()
    out["ifs_dt"] = grp["t925"].mean() - grp["t850"].mean()
    ws700 = np.sqrt(piv["u700"] ** 2 + piv["v700"] ** 2)
    out["ifs_ws700"] = ws700.groupby(level="forecast_kst_dtm").mean()
    out["ifs_q850"] = grp["q850"].mean() * 1000.0
    out["ifs_q925"] = grp["q925"].mean() * 1000.0
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full).interpolate(limit=2)
    out.index.name = "forecast_kst_dtm"
    return out.reset_index()


def load_ifs3_t2() -> pd.DataFrame:
    """wave-3 t2 (지상 2m 기온, 전 기간)."""
    d = pd.read_csv(IFS3_CSV, encoding="utf-8-sig")
    d = d[d["var"] == "t2"].drop_duplicates(subset=["run", "fxx", "lat", "lon"])
    d["forecast_kst_dtm"] = (pd.to_datetime(d.run) + pd.Timedelta(hours=9)
                             + pd.to_timedelta(d.fxx, unit="h"))
    out = d.groupby("forecast_kst_dtm")["value"].mean().to_frame("ifs_t2")
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full).interpolate(limit=2)
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
    df = df.merge(load_ifs2_features_ext(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs3_t2(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    df["ifs_inv"] = df["ifs_t925"] - df["ifs_t2"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    adds = {"base": [], "inv": ["ifs_inv"], "q925": ["ifs_q925"],
            "both": ["ifs_inv", "ifs_q925"]}
    cov24 = df.loc[df.forecast_kst_dtm.dt.year == 2024, ["ifs_inv", "ifs_q925"]].notna().mean()
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

    print("\n=== v36 접지역전·q925 절제 ===")
    for v in variants:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v:4s}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f}")


if __name__ == "__main__":
    main()
