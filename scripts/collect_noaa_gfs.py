"""NOAA GFS 아카이브 (AWS S3, public domain) — 제공 GFS에 없는 보조 변수 선별 수집.

변수: HPBL(경계층높이), SHTFL(현열플럭스, 대기안정도 프록시), VVEL 850hPa(연직속도).
누수 기준: D-1 00 UTC 사이클, f016~f039 (제공 데이터와 동일).
지점: 제공 GFS와 같은 9개 격자 (37.0~37.5N, 128.75~129.25E).

사용법: python scripts/collect_noaa_gfs.py 2024-10-01 2024-12-31
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "noaa_gfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH = ":(HPBL:surface|SHTFL:surface|VVEL:850 mb):"
LATS = [37.0, 37.25, 37.5]
LONS = [128.75, 129.0, 129.25]


def fetch_day(run_date: pd.Timestamp, fxx: int):
    from herbie import Herbie
    H = Herbie(f"{run_date:%Y-%m-%d} 00:00", model="gfs", product="pgrb2.0p25",
               fxx=fxx, verbose=False)
    ds_list = H.xarray(SEARCH, remove_grib=True)
    if not isinstance(ds_list, list):
        ds_list = [ds_list]
    recs = []
    for ds in ds_list:
        for vn in ds.data_vars:
            da = ds[vn]
            sel = da.sel(latitude=LATS, longitude=LONS)
            for la in LATS:
                for lo in LONS:
                    recs.append({
                        "run": f"{run_date:%Y%m%d}00", "fxx": fxx, "var": vn,
                        "lat": la, "lon": lo,
                        "value": float(sel.sel(latitude=la, longitude=lo).values),
                    })
        ds.close()
    return recs


def main() -> None:
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-10-01")
    end = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2024-12-31")
    out_path = OUT_DIR / f"gfs_aux_{start:%Y%m%d}_{end:%Y%m%d}.csv"

    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path, encoding="utf-8-sig")
        rows = prev.to_dict("records")
        done = set(zip(prev["run"].astype(str), prev.fxx))
        print(f"resume: {len(done)} slots")

    total = len(pd.date_range(start, end)) * 24 - len(done)
    print(f"target: {total} slots remain")
    t0, n, n_fail = time.time(), 0, 0
    for day in pd.date_range(start, end):  # day = 대상일 D → run = D-1 00UTC
        run = day - pd.Timedelta(days=1)
        for fxx in range(16, 40):
            if (f"{run:%Y%m%d}00", fxx) in done:
                continue
            for attempt in range(3):
                try:
                    rows.extend(fetch_day(run, fxx))
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        n_fail += 1
                        print(f"FAIL {run:%Y%m%d} f{fxx}: {str(e)[:100]}")
                    time.sleep(3)
            n += 1
            if n % 50 == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                rate = n / (time.time() - t0)
                eta_h = (total - n) / rate / 3600 if rate > 0 else float("inf")
                print(f"[{n}/{total}] {n/total*100:.1f}% | {rate:.2f} slot/s | "
                      f"ETA {eta_h:.1f}h | fails={n_fail} | day={day.date()}")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"done: {len(rows)} rows, fails={n_fail} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
