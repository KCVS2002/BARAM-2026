"""KIM 지역모델(r030, 3km) 지점 수집기 — nwp_vars_down GRIB 전체격자 → 지점 추출.

배경: FICR 격차 진단(#79)이 "오차 스케일 축소 = 새 모델 관점"을 지목. KIM은 미사용
대형 정보원 (한국형 역학 코어, UM/IFS/GFS와 상이, 3km). 지점 API는 NE57(2025말~)만
서빙, NC API는 180일 제한 → 과거 4년은 vars_down GRIB(2.6MB/콜)이 유일 경로.

- wave-1: ugrd/vgrd @875hPa (허브고도 상당), ef 16~39 3h(9개), D-1 00Z.
  1,461일 × 9 ef × 2 var = 26,298콜 ≈ 68GB → 일일 트래픽 한도(키당 ~5GB) 관리,
  3키 순환, 연도 우선순위 2024→2025→2023→2022.
- 추출: 최근접 3×3 격자 (rank 0~8) 값 저장.
- 산출: external_data/kim/kim_point_2022_2025.csv (tmfc,ef,var,pres,rank,lat,lon,value)
- 누수 근거: KIM r030 D-1 00Z 런은 D-1 오전 공개 (KMA 수치모델 생산 주기) < 13:00 cutoff.
"""

import csv
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "kim"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "kim_point_2022_2025.csv"
TMP = OUT_DIR / "_tmp.gb2"
COLS = ["tmfc", "ef", "var", "pres", "rank", "lat", "lon", "value"]

KEYS = []
for name in ("KMA_APIHUB_KEY", "KMA_APIHUB_KEY2", "KMA_APIHUB_KEY3"):
    for line in (PROJECT / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            KEYS.append(line.split("=", 1)[1].strip())
assert KEYS, "no API keys in .env"

API = "https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php"
HEADERS = {"User-Agent": "BARAM2026-data-collector/1.0 (research; contact: ljse237@gmail.com)"}
LAT, LON = 37.2888, 128.9610
FIELDS = [("ugrd", "875"), ("vgrd", "875")]
EFS = [16, 19, 22, 25, 28, 31, 34, 37, 39]
PRIORITY_YEARS = [2024, 2025, 2023, 2022]
# "일 5GB" 추정 한도 제거 (2026-07-21): Range 미지원 확인 후 실제 서버 신호에 위임 —
# 쿼터 응답(QuotaError)이 오면 해당 키 당일 제외, 전 키 소진 시 자정 대기 (핸들러 내장).
DAILY_QUOTA_PER_KEY = 10 ** 9


class QuotaError(RuntimeError):
    pass


def sleep_until_next_day() -> None:
    now = pd.Timestamp.now()
    nxt = now.normalize() + pd.Timedelta(days=1, minutes=5)
    print(f"전 키 할당량 소진 → {nxt}까지 {(nxt-now).total_seconds()/3600:.1f}h 대기", flush=True)
    time.sleep((nxt - now).total_seconds())


_grid_cache = {}


def extract_points(grib_bytes: bytes):
    warnings.filterwarnings("ignore")
    import xarray as xr
    i = grib_bytes.find(b"GRIB")
    if i < 0:
        raise RuntimeError("no GRIB magic")
    TMP.write_bytes(grib_bytes[i:])
    ds = xr.open_dataset(TMP, engine="cfgrib", decode_timedelta=True,
                         backend_kwargs={"indexpath": ""})
    try:
        dv = list(ds.data_vars)[0]
        if "idx" not in _grid_cache:
            lat, lon = ds.latitude.values, ds.longitude.values
            dist = (lat - LAT) ** 2 + (lon - LON) ** 2
            flat = np.argsort(dist, axis=None)[:9]
            _grid_cache["idx"] = np.unravel_index(flat, dist.shape)
            _grid_cache["lat"] = lat
            _grid_cache["lon"] = lon
        iy, ix = _grid_cache["idx"]
        vals = ds[dv].values[iy, ix]
        la = _grid_cache["lat"][iy, ix]
        lo = _grid_cache["lon"][iy, ix]
    finally:
        ds.close()
    return [(k, round(float(la[k]), 4), round(float(lo[k]), 4), round(float(vals[k]), 4))
            for k in range(9)]


def fetch(tmfc: str, var: str, pres: str, ef: int, key: str):
    r = requests.get(API, params=dict(
        nwp="r030", sub="pres", vars=var, pres=pres, tmfc=tmfc, ef=str(ef),
        dataType="GRIB", authKey=key), headers=HEADERS, timeout=300)
    if r.status_code == 403 or (len(r.content) < 3000 and ("제한" in r.text or "한도" in r.text)):
        raise QuotaError(key)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:80]}")
    if len(r.content) < 10000:
        raise RuntimeError(f"small response: {r.text[:100]}")
    return extract_points(r.content)


def main() -> None:
    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV, encoding="utf-8-sig",
                           usecols=["tmfc", "ef", "var"], dtype={"tmfc": str})
        done = set(zip(prev.tmfc, prev.ef.astype(int), prev["var"]))
        print(f"resume: {len(done)} field-slots", flush=True)
        f = open(OUT_CSV, "a", newline="", encoding="utf-8-sig")
        w = csv.writer(f)
    else:
        f = open(OUT_CSV, "w", newline="", encoding="utf-8-sig")
        w = csv.writer(f)
        w.writerow(COLS)

    slots = []
    for yr in PRIORITY_YEARS:
        for day in pd.date_range(f"{yr}-01-01", f"{yr}-12-31"):
            tmfc = (day - pd.Timedelta(days=1)).strftime("%Y%m%d") + "00"
            for var, pres in FIELDS:
                for ef in EFS:
                    if (tmfc, ef, var) not in done:
                        slots.append((tmfc, var, pres, ef))
    total = len(slots)
    print(f"target: {total} calls (~{total*2.6/1024:.0f}GB) | 키 {len(KEYS)}개", flush=True)

    t0, n, n_fail, streak = time.time(), 0, 0, 0
    used = {k: 0 for k in KEYS}
    day0 = pd.Timestamp.now().date()
    for tmfc, var, pres, ef in slots:
        while True:
            if pd.Timestamp.now().date() != day0:
                day0 = pd.Timestamp.now().date()
                used = {k: 0 for k in KEYS}
            live = [k for k in KEYS if used[k] < DAILY_QUOTA_PER_KEY]
            if not live:
                f.flush()
                sleep_until_next_day()
                continue
            key = live[n % len(live)]
            try:
                pts = fetch(tmfc, var, pres, ef, key)
                used[key] += 1
                for rank, la, lo, val in pts:
                    w.writerow([tmfc, ef, var, pres, rank, la, lo, val])
                streak = 0
                break
            except QuotaError:
                print(f"key ...{key[-4:]} 할당량 소진 신호", flush=True)
                used[key] = DAILY_QUOTA_PER_KEY
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                streak += 1
                used[key] += 1
                print(f"FAIL {tmfc} {var}{pres} ef{ef}: {str(e)[:90]}", flush=True)
                if streak >= 10:
                    f.flush()
                    print("연속 실패 10 — 30분 대기", flush=True)
                    time.sleep(1800)
                    streak = 0
                break
        n += 1
        if n % 100 == 0:
            f.flush()
            rate = n / (time.time() - t0)
            eta_h = (total - n) / max(rate, 1e-9) / 3600
            print(f"[{n}/{total}] {n/total*100:.1f}% | {rate:.2f}/s | ETA {eta_h:.1f}h "
                  f"(할당량 대기 제외) | fails={n_fail}", flush=True)
        time.sleep(0.3)
    f.close()
    print(f"done: {n} calls, fails={n_fail} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
