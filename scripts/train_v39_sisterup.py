"""v39: OM sister 피처 업그레이드 — LB 검증된 IFS10 ± wave-3 2024+ 전용 변수.

현행 sister(g3 전용, sub_013 채택)는 base124+OM만 사용 — LB에서 검증된 IFS 10피처
(#53 풍속·컨센서스, #66 안정도·시어·습도)와 wave-3의 2024+ 전용 변수(100m 바람,
연직속도 w925/w850)가 미주입 상태. sister는 2024 학습이라 **2024-04+만 있는 변수도
사용 가능** (main GBM은 전 기간 제약으로 불가) — 100m/w의 유일한 주입 경로.

변형 (base GBM은 실전대로 base124+IFS10, <2024-10 학습):
- base : 현행 sister (base124+OM, 2024-01~09 학습) — 실전 구조 재현
- s_ifs: sister + IFS10
- s_w3 : sister + IFS10 + ifs_ws100·ifs_w925·ifs_w850 (2024-04+, 1~3월 NaN 허용)
홀드아웃 2024-10~12, 3그룹 모두 채점 (실전 적용은 g3만이므로 g3가 본선).
주의(#48): 인접 홀드아웃은 2024 학습 요소를 ~20배 과대평가 — 방향 판단용.
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
from scripts.train_v22_omsister import build_pool, load_om
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
IFS3_CSV = PROJECT / "external_data" / "ecmwf_ifs" / "ifs_point3_2022_2025.csv"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_ifs3_w3() -> pd.DataFrame:
    """wave-3 2024-04+ 변수: 100m 풍속(격자 스칼라 평균), w925/w850."""
    d = pd.read_csv(IFS3_CSV, encoding="utf-8-sig")
    d = d[d["var"].isin(["u100", "v100", "w925", "w850"])]
    d = d.drop_duplicates(subset=["run", "fxx", "var", "lat", "lon"])
    d["forecast_kst_dtm"] = (pd.to_datetime(d.run) + pd.Timedelta(hours=9)
                             + pd.to_timedelta(d.fxx, unit="h"))
    piv = d.pivot_table(index=["forecast_kst_dtm", "lat", "lon"], columns="var", values="value")
    out = pd.DataFrame(index=piv.index.get_level_values(0).unique().sort_values())
    ws100 = np.sqrt(piv["u100"] ** 2 + piv["v100"] ** 2)
    out["ifs_ws100"] = ws100.groupby(level="forecast_kst_dtm").mean()
    grp = piv.groupby(level="forecast_kst_dtm")
    out["ifs_w925"] = grp["w925"].mean()
    out["ifs_w850"] = grp["w850"].mean()
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
    om = load_om()
    df = df.merge(om, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs3_w3(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    w3 = ["ifs_ws100", "ifs_w925", "ifs_w850"]
    main_cols = base_cols + ifs10
    sis_sets = {
        "base": base_cols + om_cols,
        "s_ifs": base_cols + om_cols + ifs10,
        "s_w3": base_cols + om_cols + ifs10 + w3,
    }
    y24 = df[df.forecast_kst_dtm.dt.year == 2024]
    print(f"ready: w3 2024 커버리지 {y24[w3].notna().mean().mean()*100:.0f}% "
          f"({time.time()-t0:.0f}s)", flush=True)

    va_start = pd.Timestamp(2024, 10, 1, 1)
    va_end = pd.Timestamp(2025, 1, 1, 1)
    tr_idx = df.forecast_kst_dtm < va_start
    tr = df[tr_idx]
    va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
    w_tr = clean_w[tr_idx.to_numpy()]

    # base GBM: 실전 구조 (base124+IFS10, <2024-10 전체)
    X, y, w = stack_groups(tr, main_cols, w_tr)
    shared = main_cols + ["g_rated", "g_rotor", "g_id"]
    models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
        X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
    # sister 3종: 2024-01~09 학습
    sis_idx = tr_idx & (df.forecast_kst_dtm >= pd.Timestamp(2024, 1, 1, 1))
    tr_s = df[sis_idx]
    sisters = {}
    for v, cols in sis_sets.items():
        Xs, ys, ws = stack_groups(tr_s, cols, clean_w[sis_idx.to_numpy()])
        sh = cols + ["g_rated", "g_rotor", "g_id"]
        sisters[v] = (sh, {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            Xs[sh], ys, sample_weight=ws) for q in QUANTILES_FULL})
        print(f"sister {v} 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in sis_sets}
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = tr[tgt].notna()
        sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
        Xv = group_X(sub, main_cols, tgt)
        qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                              for q in QUANTILES_FULL])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
        tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
        mu_a = tr_ok[ANEN_FEATS].mean()
        sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
        knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
        _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
        anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
        pool, med = build_pool(qp, anen_raw, cap, tgt, sub["forecast_kst_dtm"])
        a_tr = tr.loc[trm, tgt]
        a_bar = a_tr[a_tr >= cap * 0.10].mean()
        actual = sub[tgt].to_numpy()

        line = f"{tgt}:"
        for v, (sh, ms) in sisters.items():
            Xvs = group_X(sub, sis_sets[v], tgt)
            sq = np.column_stack([np.clip(ms[q].predict(Xvs[sh]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            sq.sort(axis=1)
            sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
            sis_atoms = interp_atoms(sq, n=150)  # 실전(raw, sub_013) 동일
            atoms = np.sort(np.concatenate([pool, sis_atoms], axis=1), axis=1)
            pred = optimize_submission(atoms, cap, a_bar)
            s, nm, fi, _ = metric_single(actual, pred, cap)
            res[v][tgt] = (s, nm, fi)
            line += f" {v}={s:.4f}"
        print(line + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v39 sister 피처 업그레이드 (홀드아웃 2024-10~12) ===")
    for v in sis_sets:
        s3 = res[v]["kpx_group_3"]
        tot = np.nanmean([res[v][t][0] for t in TARGET_COLS])
        print(f"{v:5s}: 3그룹 {tot:.4f} | g3 {s3[0]:.4f} (1-NMAE {s3[1]:.4f} / FICR {s3[2]:.4f})")


if __name__ == "__main__":
    main()
