"""기상청 API허브 — UM 국지(LDAPS, l015) 등압면 바람 수집.

목적: 허브고도(지형 ~1000m + 117m ≈ 875~900hPa) 부근 바람을 1.5km 해상도로 확보.
누수 기준: tmfc = D-1 00 UTC 사이클만, ef = 16~39 (제공 데이터와 동일).

파일럿 모드: 기간·변수를 좁혀 신호 품질부터 확인 후 전 기간 확대.
사용법: python scripts/collect_kma_ldaps.py 2024-10-01 2024-12-31
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "kma_ldaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY = None
env = PROJECT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if line.startswith("KMA_APIHUB_KEY="):
            KEY = line.split("=", 1)[1].strip()
KEY = os.environ.get("KMA_APIHUB_KEY", KEY)
assert KEY, "KMA_APIHUB_KEY not found"

API = "https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php"
# 기본 python-requests UA가 서버 봇차단에 걸림 → 연구 목적 식별 UA 사용
HEADERS = {"User-Agent": "BARAM2026-data-collector/1.0 (research; contact: ljse237@gmail.com)"}
SITE_LAT, SITE_LON = 37.284, 128.958

# (sub, vars, pres) — 파일럿 신호검증 결과 875hPa만 채택 (900은 상관 열세)
FIELDS = [
    ("pres", "ugrd", "875"),
    ("pres", "vgrd", "875"),
]
EFS = list(range(16, 40))  # 대상 D일 01시 ~ D+1일 00시 KST

# LDAPS 격자 → 위경도: TEXT 응답은 값만 주므로, GRIB으로 받아 cfgrib으로 좌표 포함 파싱
def fetch_field(tmfc: str, sub: str, var: str, pres: str, ef: int, tmp: Path):
    r = requests.get(API, params=dict(
        nwp="l015", sub=sub, vars=var, pres=pres, tmfc=tmfc, ef=str(ef),
        dataType="GRIB", authKey=KEY), headers=HEADERS, timeout=180)
    if r.status_code != 200 or len(r.content) < 10000:
        raise RuntimeError(f"bad response {r.status_code} len={len(r.content)}: {r.text[:120]}")
    tmp.write_bytes(r.content)
    import xarray as xr
    ds = xr.open_dataset(tmp, engine="cfgrib", decode_timedelta=True,
                         backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    lat, lon = ds.latitude.values, ds.longitude.values
    dist = (lat - SITE_LAT) ** 2 + (lon - SITE_LON) ** 2
    iy, ix = np.unravel_index(np.argsort(dist, axis=None)[:16], dist.shape)
    vals = da.values[iy, ix]
    ds.close()
    return [
        {"grid_rank": k, "lat": float(lat[iy[k], ix[k]]), "lon": float(lon[iy[k], ix[k]]),
         "value": float(vals[k])}
        for k in range(16)
    ]


DAILY_QUOTA_REQ = 4600  # 일 5GB 할당량 (GRIB ~1MB/건) 대비 여유분


def sleep_until_next_day() -> None:
    now = pd.Timestamp.now()
    nxt = (now.normalize() + pd.Timedelta(days=1, minutes=5))
    wait = (nxt - now).total_seconds()
    print(f"일일 할당량 소진 추정 → {nxt}까지 {wait/3600:.1f}h 대기")
    time.sleep(wait)


def main() -> None:
    # 연도 우선순위: 2024(CV실험) → 2025(테스트) → 2023 → 2022
    priority_years = [2024, 2025, 2023, 2022]
    tmp = OUT_DIR / "_tmp.gb2"
    rows = []
    out_path = OUT_DIR / "ldaps_pres_20220101_20251231.csv"
    done_keys = set()
    if out_path.exists():  # 이어받기
        prev = pd.read_csv(out_path, encoding="utf-8-sig")
        rows = prev.to_dict("records")
        done_keys = set(zip(prev["tmfc"], prev["var"], prev["pres"], prev["ef"]))
        print(f"resume: {len(done_keys)} field-slots done")

    all_days = []
    for yr in priority_years:
        all_days += list(pd.date_range(f"{yr}-01-01", f"{yr}-12-31"))

    total = len(all_days) * len(FIELDS) * len(EFS) - len(done_keys)
    print(f"target: {total} requests remain (연도 순서: {priority_years})")
    t0, n_req, n_fail, streak = time.time(), 0, 0, 0
    quota_day, quota_used = pd.Timestamp.now().date(), 0
    for day in all_days:  # day = 예보 대상일 D → tmfc = D-1 00UTC
        tmfc = (day - pd.Timedelta(days=1)).strftime("%Y%m%d") + "00"
        for sub, var, pres in FIELDS:
            for ef in EFS:
                if (int(tmfc), var, int(pres), ef) in done_keys or (tmfc, var, pres, ef) in done_keys:
                    continue
                # 일일 할당량 선제 관리
                if pd.Timestamp.now().date() != quota_day:
                    quota_day, quota_used = pd.Timestamp.now().date(), 0
                if quota_used >= DAILY_QUOTA_REQ:
                    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                    sleep_until_next_day()
                    quota_day, quota_used = pd.Timestamp.now().date(), 0
                try:
                    pts = fetch_field(tmfc, sub, var, pres, ef, tmp)
                    for p in pts:
                        rows.append({"tmfc": tmfc, "var": var, "pres": pres, "ef": ef, **p})
                    streak = 0
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    streak += 1
                    print(f"FAIL {tmfc} {var}{pres} ef{ef}: {str(e)[:100]}")
                    # 연속 실패 = 할당량 소진 추정 → 자정까지 대기
                    if streak >= 5:
                        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                        sleep_until_next_day()
                        streak = 0
                        quota_day, quota_used = pd.Timestamp.now().date(), 0
                n_req += 1
                quota_used += 1
                if n_req % 100 == 0:
                    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                    rate = n_req / (time.time() - t0)
                    print(f"[{n_req}/{total}] {n_req/total*100:.1f}% | {rate:.2f} req/s | "
                          f"오늘 {quota_used}/{DAILY_QUOTA_REQ} | fails={n_fail} | day={day.date()}")
                time.sleep(0.8)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"done: {len(rows)} rows, fails={n_fail} -> {out_path.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
