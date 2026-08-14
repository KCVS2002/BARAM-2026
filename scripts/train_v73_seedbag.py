"""#114 예정 v73: 모델 시드 배깅 — LGBM 분위 곡선의 시드 평균.

결정 배깅(+0.0003 LB 실증)과 같은 '노이즈 제거' 가족을 모델 수준으로.
BASE_PARAMS의 subsample/colsample 0.8 무작위성을 시드 5개로 평균해
분위 추정 노이즈를 제거 (위치 평균 — 폭·선명도 봉인축 #61 비저촉).

캐시 규칙: seed 42 qp는 oof_cache_q3 재사용, 신규 학습은 4시드만.
신규 시드 qp도 {fold}_{tgt}_qp_s{seed}.npz로 캐시 (재실행 0비용).
변형: base_off(seed42 단독) / sb3(42+7+123) / sb5(+2024+777).
판정 #91 기준 (fold평균 바 ±0.0016, 풀링 병기).
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from scripts.train_v2_clean_shared import BASE_PARAMS
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
NA = {0: 125, 1: 150, 2: 175}
K = 150
NEW_SEEDS = [7, 123, 2024, 777]
VARIANTS = ["base_off", "sb3", "sb5"]
SEEDSETS = {"sb3": [42, 7, 123], "sb5": [42, 7, 123, 2024, 777]}


def assemble(qp, anen200, dist, cap):
    n_row = len(qp)
    gbm300 = interp_atoms(qp, n=2 * K)
    med = np.median(gbm300, axis=1)
    d_bar = dist[:, :K].mean(axis=1)
    terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
    atoms = np.empty((n_row, 2 * K))
    for i in range(n_row):
        na = NA[terc[i]]
        ng = 2 * K - na
        an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
        atoms[i] = np.sort(np.concatenate([an, gb]))
    return atoms


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, shared = load_base_frame()
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

        need = [s for s in NEW_SEEDS
                if not all((CACHE / f"{fold}_{t}_qp_s{s}.npz").exists() for t in TARGET_COLS)]
        if need:
            X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
            for s in need:
                params = {**BASE_PARAMS, "random_state": s}
                models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(
                    X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
                for tgt in TARGET_COLS:
                    cap = CAPACITY_KWH[tgt]
                    sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
                    Xv = group_X(sub, cols, tgt)
                    qp = np.column_stack([np.clip(m.predict(Xv[shared]) * cap, 0, cap)
                                          for m in models.values()])
                    qp.sort(axis=1)
                    qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                    np.savez_compressed(CACHE / f"{fold}_{tgt}_qp_s{s}.npz", qp=qp)
                print(f"fold {fold}: seed {s} 학습·캐시 완료 ({time.time()-t0:.0f}s)", flush=True)
        print(f"fold {fold}: 시드 준비 완료 [진행 {fi_+1}/{len(folds)}] ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp42, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            qp_by_seed = {42: qp42}
            for s in NEW_SEEDS:
                qp_by_seed[s] = np.load(CACHE / f"{fold}_{tgt}_qp_s{s}.npz")["qp"]
            for v in VARIANTS:
                if v == "base_off":
                    qp_v = qp42
                else:
                    qp_v = np.sort(np.mean([qp_by_seed[s] for s in SEEDSETS[v]], axis=0), axis=1)
                atoms = assemble(qp_v, anen200, dist, cap)
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
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v73 시드 배깅 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
