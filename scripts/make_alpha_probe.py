# -*- coding: utf-8 -*-
"""구상 A: LB 캘리브레이션 프로브 — sub_029 예측 × α, 용량 클리핑.

sub_035_a097.csv (α=0.97), sub_036_a103.csv (α=1.03).
forecast_id/forecast_kst_dtm 불변, 예측 열만 스케일. utf-8-sig.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}

base = pd.read_csv(PROJECT / "submissions" / "sub_029_q3.csv", encoding="utf-8-sig")
assert list(base.columns) == ["forecast_id", "forecast_kst_dtm",
                              "kpx_group_1", "kpx_group_2", "kpx_group_3"]
assert len(base) == 8760 and base[list(CAP)].notna().all().all()

for alpha, name in [(0.97, "sub_035_a097.csv"), (1.03, "sub_036_a103.csv")]:
    out = base.copy()
    for g, cap in CAP.items():
        out[g] = np.clip(out[g] * alpha, 0.0, cap)
    out.to_csv(PROJECT / "submissions" / name, index=False, encoding="utf-8-sig")
    n_cap = sum((out[g] >= CAP[g] - 1e-9).sum() for g in CAP)
    print(f"{name}: α={alpha}  총합비 {out[list(CAP)].to_numpy().sum() / base[list(CAP)].to_numpy().sum():.4f}"
          f"  용량도달 {n_cap}행  max {max(out[g].max() for g in CAP):,.0f}")
print("완료 — 두 파일 모두 8760행, 열 구조 불변.")
