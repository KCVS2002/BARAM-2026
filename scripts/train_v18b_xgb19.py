"""v18b: XGBoost 분위별 독립 19모델 이종 원자 (#43 보류 카드의 재시도 조건 소진).

v18(멀티퀀타일 1모델)의 교란 요인이었던 '트리 구조 공유 → 분위 선명도 저하'를 제거:
LGBM과 동일하게 분위별 독립 모델 19개. 추가로 단조 제약 변형(풍속·풍력밀도류 +1)도
동시 평가 — 보류 카드 "XGB 퀀타일 + 단조 제약"까지 한 번에 소진.

변형 4종: xgb19_{raw,shift}, mono_{raw,shift}. GPU(RTX 3070) 학습.
기준: sub_009 구성(gbm+anen) CV 0.6470.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
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

XGB_BASE = dict(
    n_estimators=700, learning_rate=0.05, grow_policy="lossguide", max_depth=0,
    max_leaves=63, min_child_weight=40, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", device="cuda", random_state=42, verbosity=0,
)


def mono_vector(cols: list[str]) -> str:
    up = []
    for c in cols:
        pos = ((c.startswith(("ldaps_ws", "gfs_ws")) and not c.endswith(("_dsin", "_dcos")))
               or c.startswith("wpd_") or c == "gfs_surface_0_gust")
        up.append("1" if pos else "0")
    return "(" + ",".join(up) + ")"


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    variants = ("base", "xgb19_raw", "xgb19_shift", "mono_raw", "mono_shift")
    res = {v: {} for v in variants}
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
        Xs_np = X[shared].to_numpy(dtype=np.float32)
        mono = mono_vector(shared)
        xmods, mmods = {}, {}
        for q in QUANTILES_FULL:
            xmods[q] = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                                        **XGB_BASE).fit(Xs_np, y, sample_weight=w)
            mmods[q] = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                                        monotone_constraints=mono, **XGB_BASE).fit(
                Xs_np, y, sample_weight=w)
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            Xv_np = Xv[shared].to_numpy(dtype=np.float32)
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

            pools = {"base": None}
            for name, mods in (("xgb19", xmods), ("mono", mmods)):
                xq = np.column_stack([np.clip(mods[q].predict(Xv_np) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                xq.sort(axis=1)
                xq = np.sort(smooth_quantiles_by_day(xq, sub["forecast_kst_dtm"]), axis=1)
                raw = interp_atoms(xq, n=150)
                pools[f"{name}_raw"] = raw
                pools[f"{name}_shift"] = np.clip(raw + (med - np.median(raw, axis=1))[:, None], 0, cap)

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            for v in variants:
                parts = [gbm, anen] + ([pools[v]] if pools.get(v) is not None else [])
                atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v18b XGB 독립 19모델 (+단조) ===")
    for v in variants:
        print(f"{v}: {np.mean(list(res[v].values())):.4f}")


if __name__ == "__main__":
    main()
