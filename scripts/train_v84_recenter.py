"""#131 v84: AnEn 재정렬 문법 개선 — 앵커 혼합 + 곱셈 재정렬 (순수 캐시, 학습 0회).

재정렬(#26, 대회 최대 단일 승리)의 문법이 초판 그대로라는 사각지대 점검.
무조건부 구조 변경 클래스 (상수 무의존, 조건화 없음 — 전이 성적 최상 클래스).

변형:
  base : 공식 cur75 — an + (gbm_med − median(an)), 덧셈, 앵커 GBM 단독
  amix : 앵커를 (gbm_med + cnn_med)/2 로 (g1/g2만, g3는 CNN 없어 불변)
  mult : 곱셈 재정렬 an × (gbm_med / median(an)) — [0,cap] 유계 변수의 문법 교정
         (median(an) < 1% cap 시 덧셈 폴백)
  mboth: mult + amix 결합
판정 #91 (바 0.0016), 판정은 사용자.
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
VARIANTS = ["base", "amix", "mult", "mboth"]


def recenter(an, anchor, cap, mult):
    m = np.median(an)
    if mult and m >= 0.01 * cap:
        return np.clip(an * (anchor / m), 0, cap)
    return np.clip(an + (anchor - m), 0, cap)


def main() -> None:
    t0 = time.time()
    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    gsc = {v: {t: [] for t in TARGET_COLS} for v in VARIANTS}
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
            has_cnn = tgt != "kpx_group_3"
            if has_cnn:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                cnn_med = qcur[:, 9]
                mix_med = 0.5 * (med + cnn_med)
            else:
                mix_med = med
            for v in VARIANTS:
                anchor = mix_med if v in ("amix", "mboth") else med
                mult = v in ("mult", "mboth")
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = recenter(an, anchor[i], cap, mult)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + ([cnn75[i]] if has_cnn else [])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                pred = optimize_submission(np.array(atoms_l), cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                gsc[v][tgt].append(s_)
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v84 재정렬 문법 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:5s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")
    print("-- 그룹별 fold평균 --")
    for v in VARIANTS:
        print(f"{v:5s}: " + " ".join(f"{t[-1]}={np.nanmean(gsc[v][t]):.4f}" for t in TARGET_COLS))
    for v in VARIANTS:
        print(f"{v:5s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
