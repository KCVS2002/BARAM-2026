# -*- coding: utf-8 -*-
"""LB 직접 측정 후속 프로브 2종 (#110 예정) — 기준 sub_038 (평평 s=0.92).

sub_041_g3deep.csv : g3만 축소 강도 0.88 (g1/g2는 sub_038과 동일 0.92)
                     → 총점 변화가 순수하게 g3 강도 기울기 (그룹 축 측정)
sub_042_hitilt.csv : 전 그룹, 고출력 구간에 국소 상향 —
                     p<0.60은 sub_038과 동일, 0.60~0.70 램프 1.0→1.02, 이상 1.02
                     → 총점 변화가 순수하게 고출력 국소 기울기
둘 다 sub_029 원본에 프로파일 적용. 용량 클리핑.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}

base = pd.read_csv(PROJECT / "submissions" / "sub_029_q3.csv", encoding="utf-8-sig")
s38 = pd.read_csv(PROJECT / "submissions" / "sub_038_midshrink092.csv", encoding="utf-8-sig")
assert len(base) == 8760


def profile(p: np.ndarray, s_lo: float, hi_tilt: bool) -> np.ndarray:
    """s(p): p<=0.45 s_lo, 0.45~0.60 램프->1.0, (hi_tilt) 0.60~0.70 램프->1.02, 이후 1.02."""
    if hi_tilt:
        kp = np.array([0.00, 0.45, 0.60, 0.70, 1.00])
        ks = np.array([s_lo, s_lo, 1.00, 1.02, 1.02])
    else:
        kp = np.array([0.00, 0.45, 0.60, 1.00])
        ks = np.array([s_lo, s_lo, 1.00, 1.00])
    return np.interp(p, kp, ks)


# ── sub_041: g3만 0.88, 나머지 0.92 ──
out = base.copy()
for g, cap in CAP.items():
    p = base[g].to_numpy() / cap
    s = profile(p, 0.88 if g == "kpx_group_3" else 0.92, hi_tilt=False)
    out[g] = np.clip(base[g] * s, 0.0, cap)
for g in ("kpx_group_1", "kpx_group_2"):
    d = np.abs(out[g] - s38[g]).max()
    assert d < 1e-9, f"{g}가 sub_038과 달라짐 (max {d})"
d3 = np.abs(out["kpx_group_3"] - s38["kpx_group_3"])
out.to_csv(PROJECT / "submissions" / "sub_041_g3deep.csv", index=False, encoding="utf-8-sig")
print(f"sub_041_g3deep: g1/g2 = sub_038과 동일 (검증됨), g3 변경 {int((d3 > 1e-9).sum())}행"
      f" |Δ|mean {d3.mean():.0f} kWh")

# ── sub_042: 전 그룹 고출력 틸트 ──
out2 = base.copy()
n_clip = 0
for g, cap in CAP.items():
    p = base[g].to_numpy() / cap
    s = profile(p, 0.92, hi_tilt=True)
    raw = base[g] * s
    n_clip += int((raw > cap).sum())
    out2[g] = np.clip(raw, 0.0, cap)
    dd = np.abs(out2[g] - s38[g])
    print(f"sub_042 {g}: 고출력 변경 {int((dd > 1e-9).sum())}행, 중저출력 일치 "
          f"{int((p < 0.599).sum())}행 중 {int(((dd < 1e-9) & (p < 0.599)).sum())}")
out2.to_csv(PROJECT / "submissions" / "sub_042_hitilt.csv", index=False, encoding="utf-8-sig")
print(f"sub_042_hitilt: 클리핑 발동 {n_clip}행 (1.02배가 용량 초과된 행)")
