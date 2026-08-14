"""#125 v78: 이중 CNN 풀링 — 현 CNN(75) + v77 변형 CNN 추가 (순수 캐시, 학습 0회).

v77 자산 재활용: 성격이 다른 CNN들의 fold별 qp 캐시 (cnnqp=현공식 / _a/_b/_c).
변형 (전부 g1/g2 한정, g3 불변):
  base_off  : 현행 (CNN 없음)
  cur75     : 현 공식 (cnnqp 75) — 기준
  dual_b50  : cnnqp 75 + v77b(시각 임베딩) 50
  dual_c50  : cnnqp 75 + v77c(3시드 평균) 50
  dual_b25  : cnnqp 75 + v77b 25 (약결합)
진단: 현 CNN vs v77b의 q50 오차 상관 (CNN 간 다양성 크기).
판정 #91 + cur75(0.6626) 대비.
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
VARIANTS = ["base_off", "cur75", "dual_b50", "dual_c50", "dual_b25"]
EXTRA = {"dual_b50": ("b", 50), "dual_c50": ("c", 50), "dual_b25": ("b", 25)}


def main() -> None:
    t0 = time.time()
    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        fold = f"2024-{m0:02d}"
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        corrs = []
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
            qb = np.load(CACHE / f"{fold}_{tgt}_cnnqp_b.npz")["qp"]
            qc = np.load(CACHE / f"{fold}_{tgt}_cnnqp_c.npz")["qp"]
            if tgt != "kpx_group_3":
                corrs.append(np.corrcoef(qcur[:, 9] - actual, qb[:, 9] - actual)[0, 1])
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            cur300 = interp_atoms(qcur, n=2 * K)
            ex300 = {"b": interp_atoms(qb, n=2 * K), "c": interp_atoms(qc, n=2 * K)}
            for v in VARIANTS:
                add = []
                if v != "base_off" and tgt != "kpx_group_3":
                    add.append(cur300[:, np.linspace(0, 2 * K - 1, 75).astype(int)])
                    if v in EXTRA:
                        s, n_ = EXTRA[v]
                        add.append(ex300[s][:, np.linspace(0, 2 * K - 1, n_).astype(int)])
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + [a_[i] for a_ in add]
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
              + f" | CNN간 상관 {np.mean(corrs):.3f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v78 이중 CNN (캐시) — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
