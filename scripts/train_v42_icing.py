"""v42: 착빙(icing) 조건부 하방 혼합 원자 — 물리 트리거 기반 이중모드 재배치.

진단 (2026-07-20): 겨울 중풍속(6~10m/s)에서 착빙 조건(2m기온 -12~0℃ & RH≥85%) 시
이용률이 3그룹 공통 15~20%p 낮음 (n=250~430/빈). 고풍속(12+)에선 소멸 (착빙 물리 정합).
기온·습도는 이미 GBM 피처 — 남은 것은 분포의 이중모드 구조를 원자로 표현하는 것.

현행 (#46): g3만 무조건부 p=0.15 혼합 (0.8×2/3 + 0.6×1/3).
변형 (트리거·p값은 물리 선험 고정 — CV 스윕 금지, #54 상수 함정 회피):
- base   : 현행 구조 재현 (g3 p=0.15 무조건부, g1/g2 없음)
- ice_g3 : g3 조건부 — 착빙 p=0.35 / 비착빙 p=0.10
- ice_all: ice_g3 + g1/g2도 착빙 시간에만 p=0.25 (비착빙 0)
평가: 6-fold rolling, 그룹별 분해 + 겨울 fold(01·11) 주목.
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


def mix_parts(gbm, p):
    """#46 구조: 총 원자수(300) 대비 p 비율을 0.8×(2/3)·0.6×(1/3) 하방 모드로."""
    if p <= 0:
        return []
    n = 2 * gbm.shape[1]
    n8 = max(int(round(n * p * 2 / 3)), 1)
    n6 = max(int(round(n * p * 1 / 3)), 1)
    k = gbm.shape[1] - 1
    return [gbm[:, np.linspace(0, k, n8).astype(int)] * 0.8,
            gbm[:, np.linspace(0, k, n6).astype(int)] * 0.6]


def decide_conditional(gbm, anen, cap, a_bar, ice, p_ice, p_no):
    """착빙/비착빙 행을 분리해 각각의 혼합 비율로 최적화 (행 독립이라 안전)."""
    g = np.empty(gbm.shape[0])
    for flag, p in ((True, p_ice), (False, p_no)):
        m = ice == flag
        if not m.any():
            continue
        parts = [gbm[m], anen[m]] + mix_parts(gbm[m], p)
        atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
        g[m] = optimize_submission(atoms, cap, a_bar)
    return g


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
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    t2 = df["ldaps_heightAboveGround_2_t"] - 273.15
    df["icing"] = ((t2 >= -12) & (t2 <= 0) & (df["ldaps_heightAboveGround_2_r"] >= 85))
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols = base_cols + ifs10
    print(f"ready | 착빙 시간 비율 {df.icing.mean()*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    # (variant, tgt별 (p_ice, p_no)); g3 base=0.15 무조건부
    def plan(v, tgt):
        if v == "base":
            return (0.15, 0.15) if tgt == "kpx_group_3" else (0.0, 0.0)
        if v == "ice_g3":
            return (0.35, 0.10) if tgt == "kpx_group_3" else (0.0, 0.0)
        if v == "ice_all":
            return (0.35, 0.10) if tgt == "kpx_group_3" else (0.25, 0.0)

    variants = ("base", "ice_g3", "ice_all")
    res = {v: {} for v in variants}
    grp = {v: {t: [] for t in TARGET_COLS} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        X, y, w = stack_groups(tr, cols, w_tr)
        shared = cols + ["g_rated", "g_rotor", "g_id"]
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm = interp_atoms(qp, n=150)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            med = np.median(gbm, axis=1)
            anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            ice = sub["icing"].to_numpy()

            for v in variants:
                p_ice, p_no = plan(v, tgt)
                g = decide_conditional(gbm, anen, cap, a_bar, ice, p_ice, p_no)
                s, nm, fi, _ = metric_single(actual, g, cap)
                fs[v].append(s)
                grp[v][tgt].append(s)
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v42 착빙 조건부 하방 혼합 ===")
    for v in variants:
        gs = " / ".join(f"{np.nanmean(grp[v][t]):.4f}" for t in TARGET_COLS)
        print(f"{v:7s}: 총 {np.mean(list(res[v].values())):.4f} | g1/g2/g3 {gs}")


if __name__ == "__main__":
    main()
