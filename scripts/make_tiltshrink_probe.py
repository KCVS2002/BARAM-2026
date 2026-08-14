# -*- coding: utf-8 -*-
"""구상 A 4차: 기울인 프로파일 프로브 — 저출력 깊게, 상부 중간 완만하게.

p = pred/용량 기준
  s(p) = 0.88                     (p <= 0.30)
       = 0.88 + 0.08*(p-0.30)/0.15  (0.30~0.45)  -> 0.96
       = 0.96 + 0.04*(p-0.45)/0.15  (0.45~0.60)  -> 1.0
       = 1.0                      (p >= 0.60)
근거: LB 4점 실측 — FICR 절벽은 상부 중간에서 발생, 저출력은 밴드 넓고
에너지 가중 작아 깊은 축소에 무해. 출력 sub_040_tiltshrink.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
KNOTS_P = np.array([0.00, 0.30, 0.45, 0.60, 1.00])
KNOTS_S = np.array([0.88, 0.88, 0.96, 1.00, 1.00])

base = pd.read_csv(PROJECT / "submissions" / "sub_029_q3.csv", encoding="utf-8-sig")
assert len(base) == 8760

out = base.copy()
for g, cap in CAP.items():
    p = base[g].to_numpy() / cap
    s = np.interp(p, KNOTS_P, KNOTS_S)
    out[g] = np.clip(base[g] * s, 0.0, cap)
    seg = [((p <= .30).sum()), (((p > .30) & (p < .45)).sum()),
           (((p >= .45) & (p < .60)).sum()), ((p >= .60).sum())]
    print(f"{g}: 깊은축소 {seg[0]}행 램프1 {seg[1]}행 램프2 {seg[2]}행 보호 {seg[3]}행"
          f"  총합비 {out[g].sum()/base[g].sum():.4f}")

path = PROJECT / "submissions" / "sub_040_tiltshrink.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")
print(f"저장: {path.name}  전체 총합비 {out[list(CAP)].to_numpy().sum()/base[list(CAP)].to_numpy().sum():.4f}")
