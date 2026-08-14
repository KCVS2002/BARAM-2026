# -*- coding: utf-8 -*-
"""구상 A 2차: 조건부 축소 프로브 — 중저출력만 ×0.96, 고출력(FICR 밀집) 보호.

p = pred/용량 기준
  s(p) = 0.96              (p <= 0.45)
       = 0.96 + 0.04*(p-0.45)/0.15   (0.45 < p < 0.60)
       = 1.0               (p >= 0.60)
근거: α 프로브 실측 (2025 LB) — NMAE는 α<1 선호(중저출력 과대예측),
FICR은 α=1 고정(고출력 캘리브레이션 정상). 출력 sub_037_midshrink.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
S_LO, P_LO, P_HI = 0.88, 0.45, 0.60

base = pd.read_csv(PROJECT / "submissions" / "sub_029_q3.csv", encoding="utf-8-sig")
assert len(base) == 8760

out = base.copy()
for g, cap in CAP.items():
    p = base[g].to_numpy() / cap
    s = np.where(p <= P_LO, S_LO,
                 np.where(p >= P_HI, 1.0, S_LO + (1 - S_LO) * (p - P_LO) / (P_HI - P_LO)))
    out[g] = np.clip(base[g] * s, 0.0, cap)
    n_sh = (p <= P_LO).sum()
    n_rm = ((p > P_LO) & (p < P_HI)).sum()
    print(f"{g}: 축소 {n_sh}행({n_sh/87.60:.1f}%) 램프 {n_rm}행 보호 {(p >= P_HI).sum()}행"
          f"  총합비 {out[g].sum()/base[g].sum():.4f}")

path = PROJECT / "submissions" / "sub_039_midshrink088.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
print(f"저장: {path.name}  전체 총합비 {out[list(CAP)].to_numpy().sum()/base[list(CAP)].to_numpy().sum():.4f}")
