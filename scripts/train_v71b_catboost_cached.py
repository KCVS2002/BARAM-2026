"""#109 v71b (구상 B, 캐시판): CatBoost 분위 블렌딩 — oof_cache_q3 재사용.

v71과 동일 실험이나 LGBM 기준선을 재학습하지 않는다: qp(정렬·평활 완료)·
anen·dist·actual·a_bar를 oof_cache_q3에서 로드 (#89, sub_029 구성 일치 검증됨).
새 계산은 CatBoost 학습(6 fold × 1 모델)뿐. CatBoost 분위예측(cb_qp)도
같은 캐시 규약으로 저장 → 이후 결합 실험은 재학습 없이 분 단위.

변형 (GBM 쪽 원자만 교체, AnEn·NA_OFF 재배분·결정층 동일):
  base_off : 캐시 qp (LGBM)                        — 기준선
  cbrep    : CatBoost cb300 전면 교체               — 단독 품질 진단
  qavg     : 분위 곡선 50/50 평균 (같은 분위끼리)    — 위치 평균 결합
  half     : lgb150+cb150 원자 풀링                 — 분포 풀링 결합
진단: fold별 q50 오차 상관. 판정 #91 기준 (fold평균 바 ±0.0016, 풀링 병기).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import label_weights
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
CACHE = PROJECT / "experiments" / "oof_cache_q3"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
NA_OFF = {2: 175, 1: 150, 0: 125}
VARIANTS = ["base_off", "cbrep", "qavg", "half"]
CB_PARAMS = dict(
    loss_function="MultiQuantile:alpha=" + ",".join(str(q) for q in QUANTILES_FULL),
    iterations=1200, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    random_seed=42, verbose=0, allow_writing_files=False,
)
IDX150_300 = np.linspace(0, 299, 150).astype(int)


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
    cols = base_cols + ifs10
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    folds = list(range(1, 12, 2))
    for fi_, m0 in enumerate(folds):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        cb_path = CACHE / f"{fold}_cbqp.npz"
        if cb_path.exists():
            cb_store = dict(np.load(cb_path))
            print(f"fold {fold}: CatBoost 캐시 재사용 ({time.time()-t0:.0f}s)", flush=True)
        else:
            X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
            cb = CatBoostRegressor(**CB_PARAMS).fit(X[shared], y, sample_weight=w)
            cb_store = {}
            for tgt in TARGET_COLS:
                cap = CAPACITY_KWH[tgt]
                sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
                if not len(sub):
                    continue
                Xv = group_X(sub, cols, tgt)
                qp_cb = np.sort(np.clip(cb.predict(Xv[shared]) * cap, 0, cap), axis=1)
                qp_cb = np.sort(smooth_quantiles_by_day(qp_cb, sub["forecast_kst_dtm"]), axis=1)
                cb_store[tgt] = qp_cb
                cb_store[tgt + "_dtm"] = (sub["forecast_kst_dtm"].to_numpy()
                                          .astype("datetime64[ns]").astype(np.int64))
            np.savez_compressed(cb_path, **cb_store)
            print(f"fold {fold}: CatBoost 학습·저장 완료 ({time.time()-t0:.0f}s) "
                  f"[진행 {fi_+1}/{len(folds)}, {(fi_+1)/len(folds)*100:.0f}%]", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        err_corr = []
        for tgt in TARGET_COLS:
            z_ = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_lgb, anen, dist = z_["qp"], z_["anen"], z_["dist"]
            actual, a_bar, cap = z_["actual"], float(z_["a_bar"]), float(z_["cap"])
            qp_cb = cb_store[tgt]
            assert np.array_equal(z_["dtm"], cb_store[tgt + "_dtm"]), f"{fold}/{tgt} 행 정렬 불일치"
            assert qp_cb.shape == qp_lgb.shape

            e_l = qp_lgb[:, 9] - actual
            e_c = qp_cb[:, 9] - actual
            err_corr.append(np.corrcoef(e_l, e_c)[0, 1])

            g300 = {"base_off": interp_atoms(qp_lgb, n=300),
                    "cbrep": interp_atoms(qp_cb, n=300),
                    "qavg": interp_atoms((qp_lgb + qp_cb) / 2, n=300)}
            g300["half"] = np.sort(np.concatenate(
                [g300["base_off"][:, IDX150_300], g300["cbrep"][:, IDX150_300]], axis=1), axis=1)

            d_bar = dist[:, :150].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            for v in VARIANTS:
                med = np.median(g300[v], axis=1)
                atoms = np.empty((len(actual), 300))
                for i in range(len(actual)):
                    na = NA_OFF[terc[i]]
                    ng = 300 - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 299, ng), np.arange(300), g300[v][i])
                    atoms[i] = np.sort(np.concatenate([an, gb]))
                pred = optimize_submission(atoms, cap, a_bar)
                s_, nm, fi2, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi2))
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" | q50오차상관 {np.mean(err_corr):.3f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v71b CatBoost 블렌딩 (캐시판) — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in VARIANTS:
        fm = np.mean(list(res[v].values()))
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi2 = np.mean([dec[v][f][1] for f in res[v]])
        ps = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_)
        print(f"{v:8s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi2:.4f} | 풀링 {np.mean(ps):.4f}")
    for v in VARIANTS:
        print(f"{v:8s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))


if __name__ == "__main__":
    main()
