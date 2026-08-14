"""KMA LDAPS(UMKR 1.5km) 지점 연직 프로파일 피처.

수집: scripts/collect_kma_point.py → external_data/kma_ldaps/point_profile.csv
검증: 875hPa 풍속-발전량 상관 0.76 (파일럿), 지도 방식과 값 일치 확인.
레벨 선택 근거: 900hPa 이하(고기압)는 지형 채움값 (std 2.2 vs 유효 레벨 8+)
→ 875(허브고도 상당), 850, 800, 750, 700hPa 사용.
"""

from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = [87500, 85000, 80000, 75000, 70000]  # Pa


def build_kma_features(project_root: Path) -> pd.DataFrame:
    path = project_root / "external_data" / "kma_ldaps" / "point_profile.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"tmfc": str})
    df = df[df.level_pa.isin(LEVELS)]
    df["forecast_kst_dtm"] = (pd.to_datetime(df.tmfc, format="%Y%m%d%H")
                              + pd.Timedelta(hours=9) + pd.to_timedelta(df.ef, unit="h"))

    u = df[df.varn == 2002].pivot_table(index="forecast_kst_dtm", columns="level_pa", values="value")
    v = df[df.varn == 2003].pivot_table(index="forecast_kst_dtm", columns="level_pa", values="value")

    out = pd.DataFrame(index=u.index)
    for lv in LEVELS:
        hpa = lv // 100
        out[f"kma_ws{hpa}"] = np.sqrt(u[lv] ** 2 + v[lv] ** 2)
    # 875hPa 풍향 (허브고도 상당 — 복잡지형 방향 의존성)
    wd = (np.degrees(np.arctan2(-u[87500], -v[87500]))) % 360
    out["kma_wd875_sin"] = np.sin(np.radians(wd))
    out["kma_wd875_cos"] = np.cos(np.radians(wd))
    # 연직 시어 / veer
    out["kma_shear_850_875"] = out["kma_ws850"] / out["kma_ws875"].clip(lower=0.1)
    out["kma_shear_700_875"] = out["kma_ws700"] / out["kma_ws875"].clip(lower=0.1)
    dot = u[87500] * u[70000] + v[87500] * v[70000]
    mag = np.sqrt((u[87500] ** 2 + v[87500] ** 2) * (u[70000] ** 2 + v[70000] ** 2))
    out["kma_veer_875_700"] = dot / mag.clip(lower=0.1)
    # 사이클 내 시간 문맥 (같은 대상일 블록 — 누수 없음)
    out = out.reset_index()
    block = (out.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.date
    g = out.groupby(block)["kma_ws875"]
    out["kma_ws875_roll3"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    out["kma_ws875_diff1"] = g.transform(lambda s: s.diff().fillna(0))
    out["kma_ws875_dmean"] = g.transform("mean")
    return out
