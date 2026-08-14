"""NOAA GFS 보조변수 피처 (전 기간 커버 — v3 보류 건의 재시도용).

변수: w(VVEL 850hPa 연직속도), hpbl(경계층높이), shtfl(현열플럭스).
파일럿 검증: w는 예측 절대오차와 상관 0.15~0.32 (불확실성 신호).
"""

from pathlib import Path

import pandas as pd

VAR_RENAME = {"w": "vvel850", "unknown": "hpbl", "avg_ishf": "shtfl"}


def build_noaa_features(project_root: Path) -> pd.DataFrame:
    path = project_root / "external_data" / "noaa_gfs" / "gfs_aux_20220101_20251231.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["run_dt"] = pd.to_datetime(df["run"].astype(str), format="%Y%m%d%H")
    df["forecast_kst_dtm"] = df.run_dt + pd.Timedelta(hours=9) + pd.to_timedelta(df.fxx, unit="h")
    df["var"] = df["var"].map(VAR_RENAME)

    # 9격자 평균 + 격자간 표준편차 (공간 불확실성)
    mean = df.pivot_table(index="forecast_kst_dtm", columns="var", values="value", aggfunc="mean")
    mean.columns = [f"noaa_{c}" for c in mean.columns]
    std = df.pivot_table(index="forecast_kst_dtm", columns="var", values="value", aggfunc="std")
    std.columns = [f"noaa_{c}_gstd" for c in std.columns]

    out = pd.concat([mean, std], axis=1)
    # 연직속도 절대값 (오차 크기와의 상관이 방향 무관)
    out["noaa_vvel850_abs"] = out["noaa_vvel850"].abs()
    return out.reset_index()
