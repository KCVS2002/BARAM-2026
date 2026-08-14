"""Open-Meteo 외부 NWP(ECMWF/ICON/GFS) 피처 생성.

누수 규칙 (external_data/README.md 근거):
- previous_day2/day3: 전 시간대 안전.
- previous_day1: 대상시각이 01~13시(KST)인 행만 안전 → 그 외는 NaN 마스킹.
  (LightGBM은 NaN을 네이티브 처리하므로 마스킹된 채로 학습 가능)
"""

from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["ecmwf_ifs025", "icon_global", "gfs_global"]


def build_external_features(project_root: Path) -> pd.DataFrame:
    """kst_dtm(=forecast_kst_dtm) 키의 외부 NWP 피처 테이블."""
    frames = []
    for model in MODELS:
        path = project_root / "external_data" / "openmeteo" / f"{model}_prev_runs.csv"
        df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["kst_dtm"])
        short = model.split("_")[0]  # ecmwf / icon / gfs
        out = pd.DataFrame({"forecast_kst_dtm": df["kst_dtm"]})

        # day2 (안전): 핵심 변수
        out[f"om_{short}_ws100"] = df["wind_speed_100m_previous_day2"] / 3.6  # km/h -> m/s
        wd = np.radians(df["wind_direction_100m_previous_day2"])
        out[f"om_{short}_wd_sin"] = np.sin(wd)
        out[f"om_{short}_wd_cos"] = np.cos(wd)
        out[f"om_{short}_ws10"] = df["wind_speed_10m_previous_day2"] / 3.6
        out[f"om_{short}_gust"] = df["wind_gusts_10m_previous_day2"] / 3.6
        out[f"om_{short}_t2m"] = df["temperature_2m_previous_day2"]
        out[f"om_{short}_sp"] = df["surface_pressure_previous_day2"]

        # day3 및 run-to-run 추세 (예보 안정성 프록시)
        ws_d3 = df["wind_speed_100m_previous_day3"] / 3.6
        out[f"om_{short}_ws100_trend"] = out[f"om_{short}_ws100"] - ws_d3

        # day1 (부분 안전): 01~13시 KST만
        safe = df["kst_dtm"].dt.hour.between(1, 13)
        ws_d1 = df["wind_speed_100m_previous_day1"] / 3.6
        out[f"om_{short}_ws100_d1"] = ws_d1.where(safe)

        frames.append(out.set_index("forecast_kst_dtm"))

    ext = pd.concat(frames, axis=1)

    # 모델 간 불일치 = 예보 불확실성 (FICR 최적화에 특히 중요)
    ws_cols = [f"om_{m.split('_')[0]}_ws100" for m in MODELS]
    ext["om_ws100_mean"] = ext[ws_cols].mean(axis=1)
    ext["om_ws100_std"] = ext[ws_cols].std(axis=1)
    ext["om_ws100_range"] = ext[ws_cols].max(axis=1) - ext[ws_cols].min(axis=1)

    return ext.reset_index()
