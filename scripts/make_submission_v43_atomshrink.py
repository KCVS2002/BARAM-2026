# -*- coding: utf-8 -*-
"""sub_043: 중저출력 축소를 원자 분포에 흡수 — 결정층 재최적화 (#111 예정).

sub_038(사후 ×s(p))의 상위호환 시도: 검증된 프로파일 s(p)=0.92(p<=0.45)
->1.0(0.60) 를 원자 값 자체에 단조 워프 x -> x*s(x/cap) 로 적용한 뒤
결정 배깅이 이동된 분포 기준으로 FICR 베팅을 다시 최적화.
캐시 기반 (재학습 0회). 구성은 공식(sub_029)과 동일, 워프만 추가.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import TARGET_COLS

CACHE = PROJECT / "experiments" / "test_cache"
K = 150
P_MIX = 0.15
NA = {0: 125, 1: 150, 2: 175}
KP = np.array([0.00, 0.45, 0.60, 1.00])
KS = np.array([0.92, 0.92, 1.00, 1.00])


def warp(atoms: np.ndarray, cap: float) -> np.ndarray:
    """단조 워프 x -> x*s(x/cap) — 중저출력 원자만 8% 하향, 고출력 불변."""
    return atoms * np.interp(atoms / cap, KP, KS)


def optimize_bagged(atoms, cap, a_bar, B=15, frac=0.6, seed=42):
    rng = np.random.default_rng(seed)
    n = atoms.shape[1]
    k = int(n * frac)
    gs = np.empty((atoms.shape[0], B))
    for b in range(B):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        gs[:, b] = optimize_submission(atoms[:, idx], cap, a_bar)
    return np.clip(gs.mean(axis=1), 0, cap)


def build(tgt: str) -> np.ndarray:
    z = np.load(CACHE / f"test_atoms_{tgt}.npz")
    qp, anen200, dist = z["qp"], z["anen"], z["dist"]
    a_bar, cap = float(z["a_bar"]), float(z["cap"])
    n_row = len(qp)
    gbm300 = interp_atoms(qp, n=2 * K)
    gbm_atoms = gbm300[:, np.linspace(0, 2 * K - 1, K).astype(int)]
    med = np.median(gbm_atoms, axis=1)
    d_bar = dist[:, :K].mean(axis=1)
    terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
    base_atoms = np.empty((n_row, 2 * K))
    for i in range(n_row):
        na = NA[terc[i]]
        ng = 2 * K - na
        an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
        base_atoms[i] = np.sort(np.concatenate([an, gb]))
    parts = [base_atoms]

    if tgt == "kpx_group_3":
        n = 2 * K
        n8 = max(int(round(n * P_MIX * 2 / 3)), 1)
        n6 = max(int(round(n * P_MIX * 1 / 3)), 1)
        parts += [gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
                  gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6]
        sis300 = interp_atoms(z["sisq"], n=2 * K)
        fixed = np.concatenate(parts, axis=1)
        g_out = np.empty(n_row)
        for t_, ns in ((0, 125), (1, 150), (2, 175)):
            m = terc == t_
            sis = np.array([np.interp(np.linspace(0, 2 * K - 1, ns),
                                      np.arange(2 * K), s) for s in sis300[m]])
            atoms_g = np.sort(np.concatenate([fixed[m], sis], axis=1), axis=1)
            g_out[m] = optimize_bagged(warp(atoms_g, cap), cap, a_bar)
        return g_out
    atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
    return optimize_bagged(warp(atoms, cap), cap, a_bar)


def main() -> None:
    t0 = time.time()
    ref = pd.read_csv(PROJECT / "submissions" / "sub_029_q3.csv", encoding="utf-8-sig")
    s38 = pd.read_csv(PROJECT / "submissions" / "sub_038_midshrink092.csv", encoding="utf-8-sig")
    out = ref[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        out[tgt] = build(tgt)
        d38 = np.abs(out[tgt].to_numpy() - s38[tgt].to_numpy())
        print(f"{tgt}: 완료 — sub_038 대비 |Δ| mean {d38.mean():.0f} / max {d38.max():.0f} kWh"
              f" ({time.time()-t0:.0f}s)", flush=True)
    path = PROJECT / "submissions" / "sub_043_atomshrink.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
