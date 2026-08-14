"""FICR 격차(우리 0.412 vs 상위 0.453) 해부 진단 — 재탐색의 출발점.

질문:
1. 우리 오차 분포에서 에너지 가중 질량이 어디에 사는가 (밴드 경계 부근 구조)
2. 상위 FICR 0.4534가 되려면 오차가 얼마나 줄어야 하는가 (균일 축소 s* 역산)
3. '아깝게 놓친'(6~10%) 시간의 규모와 정체 — 작은 개선으로 닿는 FICR 천장 (오라클 κ)
4. g3 실측의 가용률 양자화 구조 — 양자화 스냅 의사결정의 잠재력
캐시(sub_012 시절 qp) 기반 — 분포 '모양' 진단 목적이라 충분.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.decision import interp_atoms, optimize_submission

CACHE = PROJECT / "experiments" / "oof_cache"
FOLDS = ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"]
GROUPS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
TOP_FICR = 0.4534


def ficr_of(err, a):
    price = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    return (a * price).sum() / (a * 4.0).sum()


def main() -> None:
    data = {}
    for tgt in GROUPS:
        errs, acts, meds, gs, dtms = [], [], [], [], []
        for fold in FOLDS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            cap, a_bar, a = float(z["cap"]), float(z["a_bar"]), z["actual"]
            gbm = interp_atoms(z["qp"], n=150)
            med = np.median(gbm, axis=1)
            anen = np.clip(z["anen"] + (med - np.median(z["anen"], axis=1))[:, None], 0, cap)
            atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
            g = optimize_submission(atoms, cap, a_bar)
            m = a >= 0.1 * cap
            errs.append(np.abs(g - a)[m] / cap)
            acts.append(a[m])
            meds.append(med[m] / cap)
            gs.append(g[m] / cap)
            dtms.append(pd.to_datetime(z["dtm"])[m])
        data[tgt] = (np.concatenate(errs), np.concatenate(acts), np.concatenate(meds),
                     np.concatenate(gs), pd.DatetimeIndex(np.concatenate(dtms)), cap)

    print("=" * 90)
    print("[1] 에너지 가중 오차 질량 분포 (%, 유효시간)")
    bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 1.0]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-14", "14-20", ">20"]
    for tgt in GROUPS:
        err, a = data[tgt][0], data[tgt][1]
        w = a / a.sum()
        mass = [w[(err > lo) & (err <= hi)].sum() * 100 for lo, hi in zip(bins[:-1], bins[1:])]
        w6 = w[err <= 0.06].sum()
        w8 = w[err <= 0.08].sum()
        print(f"{tgt}: " + " ".join(f"{l}:{m:.1f}" for l, m in zip(labels, mass))
              + f" | W6={w6:.3f} W8={w8:.3f} FICR={ficr_of(err, a):.4f}")

    print("\n[2] 균일 오차 축소 s에 따른 FICR (상위 0.4534 도달점 역산)")
    for tgt in GROUPS:
        err, a = data[tgt][0], data[tgt][1]
        line = f"{tgt}: "
        s_star = None
        for s in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
            f = ficr_of(err * s, a)
            line += f"s={s}:{f:.3f} "
            if s_star is None and f >= TOP_FICR:
                s_star = s
        print(line + f"→ 0.4534 도달 s*≈{s_star}")

    print("\n[3] 오라클 κ (|err|≤κ면 적중 처리) — 작은 이동으로 닿는 천장")
    for tgt in GROUPS:
        err, a = data[tgt][0], data[tgt][1]
        line = f"{tgt}: 현재 {ficr_of(err, a):.4f}"
        for kap in (0.01, 0.02, 0.04):
            e2 = np.where(err <= 0.06 + kap, np.minimum(err, 0.06), err)
            line += f" | κ={kap*100:.0f}%: {ficr_of(e2, a):.4f}"
        print(line)

    print("\n[4] 아깝게 놓친 구간(6~10%)의 정체 (에너지 가중 %)")
    for tgt in GROUPS:
        err, a, med, g, dtm, cap = data[tgt]
        w = a / a.sum()
        nm = (err > 0.06) & (err <= 0.10)
        share = w[nm].sum() * 100
        over = np.mean(g[nm] > (a[nm] / cap))  # 과대예측 비율 (건수 기준)
        mon = pd.Series(w[nm], index=dtm[nm].month).groupby(level=0).sum().nlargest(3)
        lvl = pd.Series(w[nm], index=pd.cut(a[nm] / cap, [0.1, 0.3, 0.5, 0.7, 1.0])).groupby(level=0, observed=True).sum()
        print(f"{tgt}: 질량 {share:.1f}% | 과대예측 {over*100:.0f}% | 월 상위 {dict((int(k), round(v*100,1)) for k,v in mon.items())} | cf대별 {dict((str(k), round(v*100,1)) for k,v in lvl.items())}")

    print("\n[5] g3 실측 양자화 구조 (실측/GBM중앙값 비율 분포, med 0.3~0.8 구간)")
    err, a, med, g, dtm, cap = data["kpx_group_3"]
    m = (med > 0.3) & (med < 0.8)
    ratio = (a[m] / cap) / med[m]
    hist, edges = np.histogram(ratio, bins=np.arange(0.0, 1.65, 0.1))
    for h, lo in zip(hist, edges[:-1]):
        print(f"  {lo:.1f}-{lo+0.1:.1f}: {'#' * int(h / max(hist) * 50)} {h}")
    for c, lab in ((1.0, "1.0(정상)"), (0.8, "0.8(1기 정지)"), (0.6, "0.6(2기 정지)")):
        frac = np.mean(np.abs(ratio - c) <= 0.05)
        print(f"  {lab} ±0.05 질량: {frac*100:.1f}%")
    # g1 대조
    err1, a1, med1, g1_, dtm1, cap1 = data["kpx_group_1"]
    m1 = (med1 > 0.3) & (med1 < 0.8)
    ratio1 = (a1[m1] / cap1) / med1[m1]
    for c in (1.0, 0.8, 0.6):
        print(f"  [g1 대조] {c} ±0.05 질량: {np.mean(np.abs(ratio1 - c) <= 0.05)*100:.1f}%")


if __name__ == "__main__":
    main()
