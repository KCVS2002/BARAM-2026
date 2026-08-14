"""v22: Open-Meteo ECMWF/ICON sister 원자 (2024 학습 → 2025 적용, 고리스크 카드).

- 새 정보원: OM previous-runs의 ECMWF IFS025 / ICON / GFS 예보 (previous_day2·3만
  사용 = 항상 D-1 13:00 이전 발표, external_data/README.md 근거 문서화 완료).
- 커버리지가 2024-02+뿐 → 정식 재학습 불가. 대신 2024년만으로 sister GBM 19q를
  학습해 원자로 결합 (#44 기각 시 재개 조건 "정보가 다른 이종 모델" 충족).
- 홀드아웃 설계: base GBM은 <2024-10 전체로 학습(실전과 동일 구조),
  sister는 2024-01~09로 학습 → 2024-10~12 검증. 변형: raw / shift(재정렬).
- 실전 제출 시: sister를 2024 전체로 학습 → 2025 예측.
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
OM_DIR = PROJECT / "external_data" / "openmeteo"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
P_MIX = 0.15


def load_om() -> pd.DataFrame:
    """previous_day2(항상 안전)·day3 컬럼만 사용. 모델별 접두사."""
    out = None
    for tag, fn in (("om_ec", "ecmwf_ifs025_prev_runs.csv"),
                    ("om_ic", "icon_global_prev_runs.csv"),
                    ("om_gf", "gfs_global_prev_runs.csv")):
        d = pd.read_csv(OM_DIR / fn, encoding="utf-8-sig", parse_dates=["kst_dtm"])
        cols = {}
        for base in ("wind_speed_100m", "wind_speed_10m", "wind_gusts_10m"):
            cols[f"{base}_previous_day2"] = f"{tag}_{base.replace('wind_', '').replace('_10m', '10').replace('_100m', '100')}_d2"
        cols["wind_speed_100m_previous_day3"] = f"{tag}_speed100_d3"
        wd = "wind_direction_100m_previous_day2"
        d2 = d[["kst_dtm"] + [c for c in cols if c in d.columns] + ([wd] if wd in d.columns else [])].rename(columns=cols)
        if wd in d2.columns:
            rad = np.deg2rad(d2[wd])
            d2[f"{tag}_dir_sin"] = np.sin(rad)
            d2[f"{tag}_dir_cos"] = np.cos(rad)
            d2 = d2.drop(columns=[wd])
        d2 = d2.rename(columns={"kst_dtm": "forecast_kst_dtm"})
        out = d2 if out is None else out.merge(d2, on="forecast_kst_dtm", how="outer")
    return out


def build_pool(qp, anen_raw, cap, tgt, dtm):
    """sub_012 원자 풀 재현: gbm+anen(재정렬)+g3혼합."""
    gbm = interp_atoms(qp, n=150)
    med = np.median(gbm, axis=1)
    anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
    parts = [gbm, anen]
    if tgt == "kpx_group_3":
        n = gbm.shape[1] + anen.shape[1]
        n8, n6 = max(round(n * P_MIX * 2 / 3), 1), max(round(n * P_MIX / 3), 1)
        parts += [gbm[:, np.linspace(0, 149, n8).astype(int)] * 0.8,
                  gbm[:, np.linspace(0, 149, n6).astype(int)] * 0.6]
    return np.concatenate(parts, axis=1), med


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
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    sis_cols = base_cols + om_cols
    y24 = df[df.forecast_kst_dtm.dt.year == 2024]
    print(f"ready: OM 피처 {len(om_cols)}개 | 2024 커버리지 "
          f"{y24[om_cols].notna().mean().mean()*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    va_start, sis_end = pd.Timestamp(2024, 10, 1, 1), pd.Timestamp(2024, 10, 1, 1)
    va_end = pd.Timestamp(2025, 1, 1, 1)
    tr_idx = df.forecast_kst_dtm < va_start
    tr = df[tr_idx]
    va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
    w_tr = clean_w[tr_idx.to_numpy()]

    # base GBM: 전체 <2024-10 학습 (실전 구조)
    X, y, w = stack_groups(tr, base_cols, w_tr)
    shared = base_cols + ["g_rated", "g_rotor", "g_id"]
    models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
        X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
    # sister GBM: 2024-01~09 + OM 피처
    sis_idx = tr_idx & (df.forecast_kst_dtm >= pd.Timestamp(2024, 1, 1, 1))
    tr_s = df[sis_idx]
    Xs, ys, ws = stack_groups(tr_s, sis_cols, clean_w[sis_idx.to_numpy()])
    shared_s = sis_cols + ["g_rated", "g_rotor", "g_id"]
    sisters = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
        Xs[shared_s], ys, sample_weight=ws) for q in QUANTILES_FULL}
    print(f"학습 완료: base {len(tr)}행 / sister {len(tr_s)}행 ({time.time()-t0:.0f}s)", flush=True)

    res = {v: [] for v in ("base", "om_raw", "om_shift")}
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = tr[tgt].notna()
        sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
        Xv = group_X(sub, base_cols, tgt)
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

        Xvs = group_X(sub, sis_cols, tgt)
        sq = np.column_stack([np.clip(sisters[q].predict(Xvs[shared_s]) * cap, 0, cap)
                              for q in QUANTILES_FULL])
        sq.sort(axis=1)
        sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
        om_raw = interp_atoms(sq, n=150)
        om_shift = np.clip(om_raw + (med - np.median(om_raw, axis=1))[:, None], 0, cap)

        a_tr = tr.loc[trm, tgt]
        a_bar = a_tr[a_tr >= cap * 0.10].mean()
        actual = sub[tgt].to_numpy()
        for v, extra in (("base", None), ("om_raw", om_raw), ("om_shift", om_shift)):
            atoms = np.sort(np.concatenate([pool] + ([extra] if extra is not None else []), axis=1), axis=1)
            pred = optimize_submission(atoms, cap, a_bar)
            s, _, _, _ = metric_single(actual, pred, cap)
            res[v].append(s)
        print(f"{tgt}: base={res['base'][-1]:.4f} raw={res['om_raw'][-1]:.4f} "
              f"shift={res['om_shift'][-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v22 OM sister (홀드아웃 2024-10~12) ===")
    for v in res:
        print(f"{v}: {np.nanmean(res[v]):.4f}")


if __name__ == "__main__":
    main()
