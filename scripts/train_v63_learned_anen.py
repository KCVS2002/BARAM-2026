"""#97 v63 (Phase 1): 학습된 거리의 AnEn — 조건부 분포 구성의 대안 문법 1.

배경(#96 소거법): 격차의 잔여 용의자는 '분포 구성 문법'. AnEn(표본 기반)은
같은 정보·다른 문법으로 성공한 산 증거인데, 거리가 수동(ANEN_FEATS+FEAT_W).
조사 지목: AnEn predictor 가중 최적화 = 복잡 지형 최대 -20% (단 CV 재적합은
상수 함정) → 합법 대체 = **모델이 학습한 유사도**:
- leaf: median LGBM의 리프 동시점유율(700트리) 유사도 → top-150 실측 원자.
  추가 적합 상수 0개, 지도학습이 피처 가중을 암묵 결정.
- qrf : Quantile Regression Forest — 리프 내 실측 표본이 곧 조건부 분포
  ("지도학습된 AnEn", Meinshausen 2006. quantile-forest 1.4.2, BSD, 로컬).

변형 (결정층 고정, #96 무혐의라 분포층만 교체):
  base_off: 현행 공식 (gbm300 + knn200 d̄3분위 재배분) — 앵커
  knn150  : gbm150 + knnAnEn150 고정 결합 — 문법 비교 공정 기준선
  leaf150 : gbm150 + leafAnEn150
  qrf150  : gbm150 + QRF150 (150분위 직접 예측, med 재중심)
  pool3   : gbm150 + knn75 + leaf75 (표본 문법 2종 병행)
leaf/qrf 원자는 knn과 동일하게 med 재중심 (수준 중복 방지, 형태만 주입).
하네스 v58 동일 (6-fold q3, 풀링 병기). 판정: #91 기준 (≥±0.0016 + fold 일관).
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
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
NA_OFF = {2: 175, 1: 150, 0: 125}
VARIANTS = ["base_off", "knn150", "leaf150", "qrf150", "pool3"]


def leaf_topk(model, X_tr, X_va, k=150):
    """리프 동시점유 count 상위 k 학습행 인덱스 (희소 내적, 전 700트리)."""
    L_tr = model.predict(X_tr, pred_leaf=True).astype(np.int32)
    L_va = model.predict(X_va, pred_leaf=True).astype(np.int32)
    n_tr, n_trees = L_tr.shape
    width = int(max(L_tr.max(), L_va.max())) + 1
    cols_tr = (L_tr + np.arange(n_trees, dtype=np.int32) * width).ravel()
    cols_va = (L_va + np.arange(n_trees, dtype=np.int32) * width).ravel()
    dim = n_trees * width
    A_tr = sparse.csr_matrix(
        (np.ones(cols_tr.size, np.int32),
         cols_tr, np.arange(0, cols_tr.size + 1, n_trees)), shape=(n_tr, dim))
    A_va = sparse.csr_matrix(
        (np.ones(cols_va.size, np.int32),
         cols_va, np.arange(0, cols_va.size + 1, n_trees)), shape=(len(L_va), dim))
    counts = (A_va @ A_tr.T).toarray()  # (n_va, n_tr)
    idx = np.argpartition(-counts, k, axis=1)[:, :k]
    return idx


def main() -> None:
    t0 = time.time()
    from quantile_forest import RandomForestQuantileRegressor
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
    cols = base_cols + ifs10
    Q150 = (np.arange(150) + 0.5) / 150
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
        shared = cols + ["g_rated", "g_rotor", "g_id"]
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        # QRF: 동일 스택·가중 (파라미터는 상식 기본값 — 미튜닝 명시)
        qrf = RandomForestQuantileRegressor(
            n_estimators=300, min_samples_leaf=20, max_features=0.5,
            random_state=42, n_jobs=-1).fit(
            X[shared].fillna(-999).to_numpy(), y.to_numpy(),
            sample_weight=w.to_numpy())
        print(f"fold {fold}: LGBM+QRF 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm300 = interp_atoms(qp, n=300)
            gbm150 = gbm300[:, np.linspace(0, 299, 150).astype(int)]
            med = np.median(gbm300, axis=1)

            # --- kNN AnEn (현행) ---
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(
                ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            knn150 = np.sort(anen200[:, np.linspace(0, 199, 150).astype(int)], axis=1)
            knn150 = np.clip(knn150 + (med - np.median(knn150, axis=1))[:, None], 0, cap)

            # --- leaf-proximity AnEn ---
            Xg_tr = group_X(tr.loc[trm], cols, tgt)
            lidx = leaf_topk(models[0.50], Xg_tr[shared], Xv[shared], k=150)
            acts = tr.loc[trm, tgt].to_numpy()
            leaf150 = np.sort(acts[lidx], axis=1)
            leaf150 = np.clip(leaf150 + (med - np.median(leaf150, axis=1))[:, None], 0, cap)

            # --- QRF 150분위 ---
            qrf150 = qrf.predict(Xv[shared].fillna(-999).to_numpy(), quantiles=list(Q150))
            qrf150 = np.sort(np.clip(qrf150 * cap, 0, cap), axis=1)
            qrf150 = np.clip(qrf150 + (med - np.median(qrf150, axis=1))[:, None], 0, cap)

            # --- base_off: 현행 재배분 ---
            d_bar = dist[:, :150].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            base_atoms = np.empty((len(sub), 300))
            for i in range(len(sub)):
                na = NA_OFF[terc[i]]
                ng = 300 - na
                an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                gb = np.interp(np.linspace(0, 299, ng), np.arange(300), gbm300[i])
                base_atoms[i] = np.sort(np.concatenate([an, gb]))

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            sub75 = np.linspace(0, 149, 75).astype(int)
            atoms_by = {
                "base_off": base_atoms,
                "knn150": np.sort(np.concatenate([gbm150, knn150], axis=1), axis=1),
                "leaf150": np.sort(np.concatenate([gbm150, leaf150], axis=1), axis=1),
                "qrf150": np.sort(np.concatenate([gbm150, qrf150], axis=1), axis=1),
                "pool3": np.sort(np.concatenate(
                    [gbm150, knn150[:, sub75], leaf150[:, sub75]], axis=1), axis=1),
            }
            for v in VARIANTS:
                pred = optimize_submission(atoms_by[v], cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v63 학습된 거리의 AnEn — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in VARIANTS:
        fm = np.mean(list(res[v].values()))
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        ps = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_)
        print(f"{v:8s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
