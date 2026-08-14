"""#116 v74: NA_OFF 재배분 조건 신호에 공간 불확실성(gws_std) 결합.

근거(#115 진단): gws_std가 기존 피처 대비 |잔차| 설명력을 전 그룹 +0.05~0.08
추가 — '언제 불확실한가'의 신규 정보. 소비처는 LB 검증된 재배분 기구(NA_OFF).
평균 원자 폭 불변(재배분만) → #61 비저촉. 순수 캐시 실험 (학습 0회).

변형:
  base_off : 현행 (d̄ 3분위 → 125/150/175)
  sig_rank : rank(d̄)+rank(gws_std) 합산 3분위 → 동일 맵
  sig_2d   : NA(d̄) + gws_std 3분위 오프셋 {저:-25, 중:0, 고:+25} (100~200)
  sig_gws  : gws_std 단독 3분위 (진단용)
판정 #91 기준 (fold평균 바 ±0.0016, 풀링 병기).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
GRIDF = PROJECT / "experiments" / "cache" / "ldaps_grid_features.parquet"
K = 150
NA_MAP = {0: 125, 1: 150, 2: 175}
VARIANTS = ["base_off", "sig_rank", "sig_2d", "sig_gws"]


def terc_of(x: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.quantile(x, [1 / 3, 2 / 3]), x)


def main() -> None:
    t0 = time.time()
    gf = pd.read_parquet(GRIDF)[["forecast_kst_dtm", "gws_std"]]
    # parquet 왕복이 ns→µs 강등 가능 → ns로 정규화 후 정수 키 생성
    gmap = dict(zip(gf.forecast_kst_dtm.astype("datetime64[ns]").astype("int64"), gf.gws_std))
    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        fold = f"2024-{m0:02d}"
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            gws = np.array([gmap.get(t, np.nan) for t in z["dtm"]])
            assert np.isnan(gws).mean() < 0.01, f"{fold} 공간피처 커버리지 부족"
            gws = np.where(np.isnan(gws), np.nanmedian(gws), gws)
            n_row = len(qp)
            gbm300 = interp_atoms(qp, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            t_d = terc_of(d_bar)
            t_g = terc_of(gws)
            t_rank = terc_of(np.argsort(np.argsort(d_bar)) + np.argsort(np.argsort(gws)))
            na_by = {
                "base_off": np.vectorize(NA_MAP.get)(t_d),
                "sig_rank": np.vectorize(NA_MAP.get)(t_rank),
                "sig_2d": np.clip(np.vectorize(NA_MAP.get)(t_d) + (t_g - 1) * 25, 100, 200),
                "sig_gws": np.vectorize(NA_MAP.get)(t_g),
            }
            for v in VARIANTS:
                nas = na_by[v]
                atoms = np.empty((n_row, 2 * K))
                for i in range(n_row):
                    na = int(nas[i])
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    atoms[i] = np.sort(np.concatenate([an, gb]))
                pred = optimize_submission(atoms, cap, a_bar)
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

    print("\n=== v74 공간 불확실성 재배분 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
    for v in VARIANTS:
        print(f"{v:8s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))


if __name__ == "__main__":
    main()
