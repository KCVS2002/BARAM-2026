"""Open-Meteo Previous Runs API 수집 스크립트.

ECMWF/ICON/GFS의 과거 운영예보를 리드타임별(previous_day1~3)로 수집한다.

[누수 안전 근거 — external_data/README.md에도 기록]
- 대회 예측기준: D-1 13:00 KST(= D-1 04:00 UTC) 이전에 발표된 예보만 사용 가능.
- previous_day2 = 대상시각 48h 전에 예측된 값. 대상시각이 가장 늦은 D+1 00:00 KST여도
  초기화 시각 <= D-1 00:00 KST(= D-2 15:00 UTC)이고 발표는 그 후 수 시간 내
  → 항상 기준시점보다 하루 이상 이전. 안전.
- previous_day1 = 대상시각 24h 전 예측값. 대상시각 <= D 13:00 KST 인 행만 안전.
  (그 이후 시각은 초기화가 기준시점 이후일 수 있음 → 사용 시 시간대 마스킹 필수)
- previous_day3 = 72h 전. 항상 안전. 예보 추세(run-to-run trend) 피처용.

라이선스: CC-BY 4.0 (Open-Meteo). 출처표기 필요.
"""

import time
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "openmeteo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT, LON = 37.28, 128.95  # 태백 가덕산 풍력단지 터빈 중심
API = "https://previous-runs-api.open-meteo.com/v1/forecast"

MODELS = ["ecmwf_ifs025", "icon_global", "gfs_global"]
BASE_VARS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "surface_pressure",
]
LEADS = ["previous_day1", "previous_day2", "previous_day3"]

# 분기 단위 청크 (2024-01 ~ 2025-12)
CHUNKS = []
for year in (2024, 2025):
    for q in range(4):
        start = pd.Timestamp(year=year, month=3 * q + 1, day=1)
        end = (start + pd.offsets.QuarterEnd()).normalize()
        CHUNKS.append((start.date().isoformat(), end.date().isoformat()))


def fetch(model: str, start: str, end: str) -> pd.DataFrame:
    hourly = ",".join(f"{v}_{lead}" for v in BASE_VARS for lead in LEADS)
    r = requests.get(
        API,
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": hourly,
            "start_date": start,
            "end_date": end,
            "models": model,
            "timezone": "Asia/Seoul",
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    if "hourly" not in d:
        raise RuntimeError(f"no hourly in response: {str(d)[:200]}")
    df = pd.DataFrame(d["hourly"])
    df = df.rename(columns={"time": "kst_dtm"})
    return df


def main() -> None:
    for model in MODELS:
        frames = []
        for start, end in CHUNKS:
            for attempt in range(3):
                try:
                    df = fetch(model, start, end)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  retry {attempt + 1} {model} {start}: {e}")
                    time.sleep(10)
            else:
                raise SystemExit(f"FAILED: {model} {start}~{end}")
            n_valid = df.drop(columns="kst_dtm").notna().any(axis=1).sum()
            print(f"{model} {start}~{end}: rows={len(df)} valid={n_valid}")
            frames.append(df)
            time.sleep(2)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(OUT_DIR / f"{model}_prev_runs.csv", index=False, encoding="utf-8-sig")
        print(f"saved {model}: {out.shape}")


if __name__ == "__main__":
    main()
