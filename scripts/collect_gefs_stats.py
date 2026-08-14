"""GEFS 앙상블 평균(geavg)·스프레드(gespr) 10m 바람 수집 (NOAA AWS, public domain).

용도: 예측시점 피처 치환용 시나리오 생성 (학습 불변 → 커버리지 문제 없음)
     → 2024(검증) + 2025(테스트)만 수집.
누수: D-1 00 UTC 사이클, f015~f039 3시간 간격 (GEFS는 3h 산출).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "gefs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "gefs_stats_2024_2025.csv"

LATS = [37.0, 37.25, 37.5]
LONS = [128.75, 129.0, 129.25]
FXX = list(range(15, 40, 3))
N_WORKERS = 6


def fetch_slot(run: pd.Timestamp, fxx: int, member: str):
    from herbie import Herbie
    try:
        H = Herbie(f"{run:%Y-%m-%d} 00:00", model="gefs", product="atmos.25",
                   member=member, fxx=fxx, verbose=False)
        ds = H.xarray(":(UGRD|VGRD):10 m above ground:", remove_grib=True)
        recs = []
        for vn in ds.data_vars:
            sel = ds[vn].sel(latitude=LATS, longitude=LONS)
            for la in LATS:
                for lo in LONS:
                    recs.append({"run": f"{run:%Y%m%d}00", "fxx": fxx, "member": member,
                                 "var": vn, "lat": la, "lon": lo,
                                 "value": float(sel.sel(latitude=la, longitude=lo).values)})
        ds.close()
        return run, fxx, member, recs, None
    except Exception as e:  # noqa: BLE001
        return run, fxx, member, [], str(e)[:100]


def main() -> None:
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-01-01")
    end = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2025-12-31")

    rows, done = [], set()
    if OUT_PATH.exists():
        prev = pd.read_csv(OUT_PATH, encoding="utf-8-sig")
        rows = prev.to_dict("records")
        done = set(zip(prev["run"].astype(str), prev["fxx"], prev["member"]))
        print(f"resume: {len(done)} slots")

    slots = []
    for day in pd.date_range(start, end):
        run = day - pd.Timedelta(days=1)
        for fxx in FXX:
            for member in ("avg", "spr"):
                if (f"{run:%Y%m%d}00", fxx, member) not in done:
                    slots.append((run, fxx, member))
    total = len(slots)
    print(f"target: {total} slots ({N_WORKERS} workers)")

    t0, n, n_fail = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(fetch_slot, r, f, m) for r, f, m in slots]
        for fut in as_completed(futures):
            run, fxx, member, recs, err = fut.result()
            n += 1
            if err:
                n_fail += 1
                if n_fail < 30 or n_fail % 50 == 0:
                    print(f"FAIL {run:%Y%m%d} f{fxx} {member}: {err}")
            else:
                rows.extend(recs)
            if n % 200 == 0:
                pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
                rate = n / (time.time() - t0)
                print(f"[{n}/{total}] {n/total*100:.1f}% | {rate:.2f}/s | "
                      f"ETA {(total-n)/rate/3600:.1f}h | fails={n_fail}")
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"done: {len(rows)} rows, fails={n_fail} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
