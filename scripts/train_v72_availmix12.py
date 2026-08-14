"""#113 예정 v72: g1/g2 가용률 혼합 원자 — #102가 남긴 명시 카드.

g3 공식 구성(가용률 하방 모드 0.8/0.6, p=0.15)의 g1/g2 이식. 근거: #101에서
g1 정지 ≥1기 시간 22.8% / g2 12.0% (g3 20.5%와 동급인데 혼합은 g3만 적용 중).
순수 캐시 실험 (oof_cache_q3, 학습 0회). 변형:
  base_off / am12_p15(g1·g2에 p=0.15) / am12_p075(절반 강도) / am1_p15(g1만)
g3는 전 변형에서 불변(공식 구성에 이미 포함돼 CV 캐시 기준선에 없음 — 주의:
캐시 base는 g3 혼합 미포함 순수 구성이므로 여기선 g1/g2 추가 효과만 측정).
판정 #91 기준 (fold평균 바 ±0.0016, 풀링 병기).
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
NA = {0: 125, 1: 150, 2: 175}
K = 150
VARIANTS = ["base_off", "am12_p15", "am12_p075", "am1_p15"]
P_BY_VAR = {"am12_p15": 0.15, "am12_p075": 0.075, "am1_p15": 0.15}
TGTS_BY_VAR = {"am12_p15": ("kpx_group_1", "kpx_group_2"),
               "am12_p075": ("kpx_group_1", "kpx_group_2"),
               "am1_p15": ("kpx_group_1",)}


def mix_atoms(gbm_atoms: np.ndarray, p_mix: float) -> list:
    n = 2 * K
    n8 = max(int(round(n * p_mix * 2 / 3)), 1)
    n6 = max(int(round(n * p_mix * 1 / 3)), 1)
    return [gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
            gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6]


def main() -> None:
    t0 = time.time()
    folds = [f"2024-{m:02d}" for m in range(1, 12, 2)]
    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for fold in folds:
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            n_row = len(qp)
            gbm300 = interp_atoms(qp, n=2 * K)
            gbm_atoms = gbm300[:, np.linspace(0, 2 * K - 1, K).astype(int)]
            med = np.median(gbm300, axis=1)
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

            for v in VARIANTS:
                if v != "base_off" and tgt in TGTS_BY_VAR[v]:
                    parts = [base_atoms] + mix_atoms(gbm_atoms, P_BY_VAR[v])
                    atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
                else:
                    atoms = base_atoms
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

    print("\n=== v72 g1/g2 가용률 혼합 (캐시) — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
    # 그룹별 풀링 (g1/g2 국소 효과 직접 확인)
    for v in VARIANTS:
        gs = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            gs.append(f"{tgt[-1]}:{s_:.4f}")
        print(f"{v:9s} 그룹별 풀링: " + " ".join(gs))


if __name__ == "__main__":
    main()
