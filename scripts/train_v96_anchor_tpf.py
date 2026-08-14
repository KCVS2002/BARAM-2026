"""#145 v96: AnEn 재정렬 앵커에 TabPFN q50 혼합 — v84(amix, CNN 앵커) 실패의 정밀 재시도.

v84 실패 원인 = CNN q50 품질 열세(1.09)로 앵커 노이즈 주입. TabPFN q50은 품질
동급~우위(pinball ~0.96, v95 실측) + 상관 0.93 → 혼합 앵커가 앵커 오차를 실제
축소 가능. AnEn 150원자 전체가 앵커를 따라가는 큰 지렛대. 무조건부 구조.

변형:
  base   : 앵커 = GBM q50 (공식)
  mix50  : 앵커 = 0.5·GBM + 0.5·TabPFN (g1/g2만, g3 불변)
  mix33  : 앵커 = 2/3·GBM + 1/3·TabPFN (g1/g2만)
  mix50a : mix50을 g3까지 (g3 tpfqp는 v86 캐시)
판정 #91, 판정은 사용자. 순수 캐시.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
VARIANTS = ["base", "mix50", "mix33", "mix50a"]
G12 = ["kpx_group_1", "kpx_group_2"]


def main() -> None:
    t0 = time.time()
    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        fold = f"2024-{m0:02d}"
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            is_g12 = tgt in G12
            qtp = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
            tpf_med = qtp[:, 9]
            if is_g12:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tp75 = interp_atoms(qtp, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            for v in VARIANTS:
                if is_g12:
                    if v == "base":
                        anc = med
                    elif v == "mix33":
                        anc = (2 * med + tpf_med) / 3
                    else:
                        anc = 0.5 * (med + tpf_med)
                else:
                    anc = 0.5 * (med + tpf_med) if v == "mix50a" else med
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (anc[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + ([cnn75[i], tp75[i]] if is_g12 else [])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                pred = optimize_submission(np.array(atoms_l), cap, a_bar)
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

    print("\n=== v96 TabPFN 혼합 앵커 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:6s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")
    for v in VARIANTS:
        print(f"{v:6s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
