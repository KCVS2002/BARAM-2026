"""g3 오차 해부 (#79 후속) — g3 오차가 g2 대비 ~30% 큰 원인 분해.

질문:
1. g3 오차 중 '기상 공통'(g2와 상관) 성분과 'g3 고유' 성분의 비중은?
2. 풍속 레짐별로 균일하게 나쁜가, 특정 레짐인가?
3. 학습량 효과: fold가 진행(학습 1년→2년)되며 격차가 줄어드는가?
   → 줄어든다면 2025 적용 시(2년 학습) CV보다 나을 것 — 격차의 자기 치유 여부
4. g3 단독 결손(부분정지 추정: g2는 정상인데 g3만 크게 미달)의 오차 기여는?
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


def load(fold, tgt):
    z = np.load(CACHE / f"{fold}_{tgt}.npz")
    cap, a_bar, a = float(z["cap"]), float(z["a_bar"]), z["actual"]
    gbm = interp_atoms(z["qp"], n=150)
    med = np.median(gbm, axis=1)
    anen = np.clip(z["anen"] + (med - np.median(z["anen"], axis=1))[:, None], 0, cap)
    atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
    g = optimize_submission(atoms, cap, a_bar)
    return pd.DataFrame({
        "dtm": pd.to_datetime(z["dtm"]), "a": a / cap, "g": g / cap, "med": med / cap,
    }).set_index("dtm")


def main() -> None:
    rows = []
    for fold in FOLDS:
        d2 = load(fold, "kpx_group_2").add_suffix("_2")
        d3 = load(fold, "kpx_group_3").add_suffix("_3")
        m = d2.join(d3, how="inner")
        m["fold"] = fold
        rows.append(m)
    m = pd.concat(rows)
    m["e2"] = m.g_2 - m.a_2
    m["e3"] = m.g_3 - m.a_3
    v2 = m.a_2 >= 0.1
    v3 = m.a_3 >= 0.1
    both = v2 & v3

    print("[1] 오차 분해 (유효시간 교집합, cf 단위)")
    e2, e3 = m.loc[both, "e2"], m.loc[both, "e3"]
    r = np.corrcoef(e2, e3)[0, 1]
    beta = np.cov(e3, e2)[0, 1] / e2.var()
    shared = beta * e2
    resid = e3 - shared
    print(f"  |e2| 평균 {e2.abs().mean():.4f} | |e3| 평균 {e3.abs().mean():.4f} (비율 {e3.abs().mean()/e2.abs().mean():.2f})")
    print(f"  corr(e2,e3)={r:.3f} | e3 분산 중 공유성분 {r**2*100:.0f}% / 고유성분 {(1-r**2)*100:.0f}%")
    print(f"  고유성분 |resid| 평균 {resid.abs().mean():.4f}")

    print("\n[2] 풍속(GBM med) 레짐별 |오차| (g2 vs g3)")
    bins = pd.cut(m.loc[both, "med_3"], [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    tab = m.loc[both].groupby(bins, observed=True).apply(
        lambda d: pd.Series({"|e2|": d.e2.abs().mean(), "|e3|": d.e3.abs().mean(),
                             "비율": d.e3.abs().mean() / max(d.e2.abs().mean(), 1e-9),
                             "n": len(d)}))
    print(tab.round(3).to_string())

    print("\n[3] fold별 |e3|/|e2| 추세 (학습량 1→2년)")
    for fold in FOLDS:
        d = m[(m.fold == fold) & both]
        if len(d) < 50:
            continue
        print(f"  {fold}: |e2| {d.e2.abs().mean():.4f} |e3| {d.e3.abs().mean():.4f} "
              f"비율 {d.e3.abs().mean()/d.e2.abs().mean():.2f} (n={len(d)})")

    print("\n[4] g3 단독 결손 에피소드 (g2 정상·g3 미달)")
    short3 = (m.a_3 < 0.7 * m.med_3) & (m.a_2 > 0.85 * m.med_2) & both
    frac = short3[both].mean()
    contrib = m.loc[short3, "e3"].abs().sum() / m.loc[both, "e3"].abs().sum()
    print(f"  해당 시간 비율 {frac*100:.1f}% | g3 총 |오차| 기여 {contrib*100:.1f}%")
    over = (m.loc[short3, "e3"] > 0).mean()
    print(f"  그중 과대예측(발전 미달) 방향 {over*100:.0f}%")
    # 반대 방향(g3 초과)도
    up3 = (m.a_3 > 1.3 * m.med_3) & (m.a_2 < 1.15 * m.med_2) & both
    print(f"  [대조] g3 단독 초과 시간 {up3[both].mean()*100:.1f}% | 기여 "
          f"{m.loc[up3, 'e3'].abs().sum() / m.loc[both, 'e3'].abs().sum()*100:.1f}%")


if __name__ == "__main__":
    main()
