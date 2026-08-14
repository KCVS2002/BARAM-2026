"""#89: q3 기반 OOF 캐시(v2)로 하류 결정층 상수 재점검.

배경: 현행 결정층 상수(#77 거리 재배분 125/150/175, 배깅 B=15/frac=0.6)는 전부
무가중 시절 측정치. q3로 원자 분포가 바뀌었으니 최적점 이동 여부를 확인한다.
목적은 스윕 채택이 아니라 **현 구성이 여전히 고원 위인지** — 크게 벗어났을 때만
프로브 후보 (2024 상수 이식 금지 원칙 #54 유지, 구조 변경만 신뢰 가능).

축 1 — AnEn 재배분(NA, d̄ 3분위 멂/중/가까움 → AnEn 원자 수):
  cur {175,150,125} / flat {150,150,150}(재배분 제거) / wide {200,150,100}
  / up {200,175,150}(전체 증량) / dn {150,125,100}(전체 감량)
축 2 — 배깅 (cur 재배분 고정): B15f06(현행) / B15f05 / B15f07 / B30f06
채점: fold 평균 + 풀링(fold 연결 후 그룹별 metric) 병기 (#86).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
FOLDS = ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"]
# NA = {멂(terc2), 중간(terc1), 가까움(terc0)} — make_submission 표기와 정렬
NA_VARIANTS = {
    "cur": {2: 175, 1: 150, 0: 125},
    "flat": {2: 150, 1: 150, 0: 150},
    "wide": {2: 200, 1: 150, 0: 100},
    "up": {2: 200, 1: 175, 0: 150},
    "dn": {2: 150, 1: 125, 0: 100},
}
BAG_VARIANTS = {"B15f06": (15, 0.6), "B15f05": (15, 0.5), "B15f07": (15, 0.7), "B30f06": (30, 0.6)}
N_TOTAL = 300


def optimize_bagged(atoms, cap, a_bar, B=15, frac=0.6, seed=42):
    rng = np.random.default_rng(seed)
    n = atoms.shape[1]
    k = int(n * frac)
    gs = np.empty((atoms.shape[0], B))
    for b in range(B):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        gs[:, b] = optimize_submission(atoms[:, idx], cap, a_bar)
    return np.clip(gs.mean(axis=1), 0, cap)


def build_atoms(qp, anen, dist, cap, na_map):
    gbm300 = interp_atoms(qp, n=N_TOTAL)
    med = np.median(gbm300, axis=1)
    d_bar = dist[:, :150].mean(axis=1)
    terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
    out = np.empty((len(qp), N_TOTAL))
    for i in range(len(qp)):
        na = na_map[terc[i]]
        ng = N_TOTAL - na
        an = np.interp(np.linspace(0, anen.shape[1] - 1, na), np.arange(anen.shape[1]), anen[i])
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = np.interp(np.linspace(0, N_TOTAL - 1, ng), np.arange(N_TOTAL), gbm300[i])
        out[i] = np.sort(np.concatenate([an, gb]))
    return out


def main() -> None:
    t0 = time.time()
    variants = [("na", k) for k in NA_VARIANTS] + [("bag", k) for k in BAG_VARIANTS if k != "B15f06"]
    fold_scores = {v: {} for _, v in variants}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for _, v in variants}

    for fold in FOLDS:
        fs = {v: [] for _, v in variants}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp, anen, dist = z["qp"], z["anen"], z["dist"]
            cap, a_bar, actual = float(z["cap"]), float(z["a_bar"]), z["actual"]
            atoms_by_na = {k: build_atoms(qp, anen, dist, cap, m) for k, m in NA_VARIANTS.items()}
            for kind, v in variants:
                if kind == "na":
                    pred = optimize_bagged(atoms_by_na[v], cap, a_bar)
                else:
                    B, frac = BAG_VARIANTS[v]
                    pred = optimize_bagged(atoms_by_na["cur"], cap, a_bar, B=B, frac=frac)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for _, v in variants:
            fold_scores[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: " + " ".join(f"{v}={fold_scores[v][fold]:.4f}" for _, v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== #89 하류 재점검 (q3 캐시) — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    from src.metric import CAPACITY_KWH
    for kind, v in variants:
        fm = np.mean(list(fold_scores[v].values()))
        ps, pn, pf = [], [], []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, nm, fi, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_); pn.append(nm); pf.append(fi)
        print(f"{kind}:{v:7s} fold평균 {fm:.4f} | 풀링 {np.mean(ps):.4f} "
              f"(1-NMAE {np.mean(pn):.4f} / FICR {np.mean(pf):.4f})")


if __name__ == "__main__":
    main()
