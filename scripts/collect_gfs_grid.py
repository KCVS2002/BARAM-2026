"""NOAA GFS 광역 격자 수집 — 공간 통계 피처용 (#93, 티어3a 1단계).

배경(research/domain_theory.md §2): 단일 지점 대신 NWP 격자 전체의 공간 통계
(평균·분산·구배·PC)가 풍력 MAE -12.85% 보고 (Andrade & Bessa 2017). 제공 데이터는
9격자(0.5°/~50km)뿐 → 0.25° 11×11(36.0~38.5N, 127.5~130.0E, ~275km)로 확장.

수집: UGRD/VGRD 100m / D-1 00UTC f016~f039 (제공 GFS와 동일 누수 기준) / 슬롯당
1행(11×11 풍속 121컬럼 + u·v 평균). 재현: python scripts/collect_gfs_grid.py 2022-01-01 2025-12-31
"""

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "noaa_gfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH = ":(UGRD|VGRD):100 m above ground:"
LATS = np.arange(36.0, 38.51, 0.25)   # 11
LONS = np.arange(127.5, 130.01, 0.25)  # 11
N_WORKERS = 6


def fetch_slot_inner(run: pd.Timestamp, fxx: int):
    warnings.filterwarnings("ignore")
    from herbie import Herbie
    H = Herbie(f"{run:%Y-%m-%d} 00:00", model="gfs", product="pgrb2.0p25",
               fxx=fxx, verbose=False)
    ds_list = H.xarray(SEARCH, remove_grib=True)
    if not isinstance(ds_list, list):
        ds_list = [ds_list]
    u = v = None
    for ds in ds_list:
        for vn in ds.data_vars:
            arr = ds[vn].sel(latitude=LATS, longitude=LONS)
            # latitude 내림차순 저장 대비 정렬 고정 (남→북, 서→동)
            arr = arr.sortby("latitude").sortby("longitude")
            if vn.lower().startswith("u"):
                u = arr.values
            else:
                v = arr.values
        ds.close()
    ws = np.hypot(u, v)
    row = {"run": f"{run:%Y%m%d}00", "fxx": fxx,
           "u_mean": float(np.mean(u)), "v_mean": float(np.mean(v))}
    for i in range(ws.shape[0]):
        for j in range(ws.shape[1]):
            row[f"ws_{i}_{j}"] = round(float(ws[i, j]), 3)
    return row


def fetch_slot(run: pd.Timestamp, fxx: int):
    for attempt in range(3):
        try:
            return run, fxx, fetch_slot_inner(run, fxx), None
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return run, fxx, None, str(e)[:120]
            time.sleep(3)


def main() -> None:
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
    end = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2025-12-31")
    out_path = OUT_DIR / f"gfs_grid100_{start:%Y%m%d}_{end:%Y%m%d}.csv"

    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path, encoding="utf-8-sig")
        rows = prev.to_dict("records")
        done = set(zip(prev["run"].astype(str), prev.fxx))
        print(f"resume: {len(done)} slots", flush=True)

    slots = []
    for day in pd.date_range(start, end):  # day = 대상일 D → run = D-1 00UTC
        run = day - pd.Timedelta(days=1)
        for fxx in range(16, 40):
            if (f"{run:%Y%m%d}00", fxx) not in done:
                slots.append((run, fxx))
    total = len(slots)
    print(f"target: {total} slots ({N_WORKERS}-way)", flush=True)

    t0, n, n_fail = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fetch_slot, run, fxx) for run, fxx in slots]
        for fut in as_completed(futs):
            run, fxx, row, err = fut.result()
            n += 1
            if err is not None:
                n_fail += 1
                print(f"FAIL {run:%Y%m%d} f{fxx}: {err}", flush=True)
            else:
                rows.append(row)
            if n % 200 == 0 or n == total:
                pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                el = time.time() - t0
                eta = el / n * (total - n) / 3600
                print(f"[{n}/{total}] {n/total*100:.1f}% fail={n_fail} "
                      f"elapsed={el/3600:.1f}h ETA={eta:.1f}h", flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"완료: {len(rows)}행, 실패 {n_fail} → {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
