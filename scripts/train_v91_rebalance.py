"""#140 v91: 원자 구성 재균형 — 4자 풀(AnEn+GBM 300 / CNN 75 / TabPFN 75) 지분 재점검.

현 지분은 sister 0~1개 시절 최적. 품질 동급 TabPFN 합류 후 미측정 영역 (순수 캐시).
변형 (g1/g2, g3 불변):
  base     : 현 공식 (base 300 [NA 125/150/175] + cnn75 + tpf75)
  b250     : base 250 (NA ×5/6) + cnn75 + tpf75 — 베이스 축소 1단
  b200     : base 200 (NA ×2/3) + cnn75 + tpf75 — 베이스 축소 2단
  s100     : base 300 + cnn100 + tpf100 — sister 증량
  anen_lite: AnEn만 -25 (NA 100/125/150, GBM으로 충전) + cnn75 + tpf75
판정 #91, 판정은 사용자.
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
VARIANTS = ["base", "b250", "b200", "s100", "anen_lite"]
G12 = ["kpx_group_1", "kpx_group_2"]


def cfg_of(v, t):
    """(base_total, na, n_cnn, n_tpf)"""
    if v == "base":
        return 300, NA[t], 75, 75
    if v == "b250":
        return 250, round(NA[t] * 250 / 300), 75, 75
    if v == "b200":
        return 200, round(NA[t] * 200 / 300), 75, 75
    if v == "s100":
        return 300, NA[t], 100, 100
    return 300, NA[t] - 25, 75, 75  # anen_lite


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
            if is_g12:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                qtp = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
                cnn300 = interp_atoms(qcur, n=2 * K)
                tp300 = interp_atoms(qtp, n=2 * K)
            g3_pred = None
            for v in VARIANTS:
                if not is_g12:
                    if g3_pred is None:
                        atoms_l = []
                        for i in range(n_row):
                            na = NA[terc[i]]
                            ng = 2 * K - na
                            an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                            an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                            gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                            atoms_l.append(np.sort(np.concatenate([an, gb])))
                        g3_pred = optimize_submission(np.array(atoms_l), cap, a_bar)
                    pred = g3_pred
                else:
                    atoms_l = []
                    for i in range(n_row):
                        btot, na, nc, nt = cfg_of(v, terc[i])
                        ng = btot - na
                        an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                        gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                        cn = cnn300[i][np.linspace(0, 2 * K - 1, nc).astype(int)]
                        tp = tp300[i][np.linspace(0, 2 * K - 1, nt).astype(int)]
                        atoms_l.append(np.sort(np.concatenate([an, gb, cn, tp])))
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

    print("\n=== v91 원자 재균형 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:9s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")
    for v in VARIANTS:
        print(f"{v:9s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
