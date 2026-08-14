# -*- coding: utf-8 -*-
"""#112 예정: 계절별 축소 깊이 LB 분해 프로브 — 기준 sub_038 (전 계절 0.92).

해당 계절 행만 중저출력 축소 0.88 (sub_039와 동일 프로파일), 나머지 0.92.
sub_039(전 계절 0.88) 델타 -0.00054 = 4계절 성분 합 → 3회 제출로 전 분해:
  sub_045_wdeep: 겨울(12,1,2) / sub_046_sdeep: 봄(3,4,5) / sub_047_mdeep: 여름(6,7,8)
  가을 성분 = sub_039델타 - (겨울+봄+여름 델타).
행의 월 귀속은 (forecast_kst_dtm - 1h), kst_dtm이 구간 종료 시각이므로.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
SEASONS = {"sub_045_wdeep": (12, 1, 2), "sub_046_sdeep": (3, 4, 5), "sub_047_mdeep": (6, 7, 8)}


def profile(p: np.ndarray, s_lo: float) -> np.ndarray:
    return np.interp(p, [0.00, 0.45, 0.60, 1.00], [s_lo, s_lo, 1.00, 1.00])


base = pd.read_csv(PROJECT / "submissions" / "sub_029_q3.csv", encoding="utf-8-sig",
                   parse_dates=["forecast_kst_dtm"])
s38 = pd.read_csv(PROJECT / "submissions" / "sub_038_midshrink092.csv", encoding="utf-8-sig")
month = (base["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.month.to_numpy()

for name, months in SEASONS.items():
    m_sea = np.isin(month, months)
    out = base.copy()
    for g, cap in CAP.items():
        p = base[g].to_numpy() / cap
        s = np.where(m_sea, profile(p, 0.88), profile(p, 0.92))
        out[g] = np.clip(base[g] * s, 0.0, cap)
        d_off = np.abs(out[g].to_numpy()[~m_sea] - s38[g].to_numpy()[~m_sea]).max()
        assert d_off < 1e-9, f"{name}/{g}: 비대상 계절이 sub_038과 달라짐"
    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(PROJECT / "submissions" / f"{name}.csv", index=False, encoding="utf-8-sig")
    n_ch = sum(int((np.abs(out[g] - s38[g]) > 1e-9).sum()) for g in CAP)
    print(f"{name}: 대상 {int(m_sea.sum())}행/그룹, 변경 {n_ch}행(3그룹 합), 비대상 = sub_038 일치 검증")
