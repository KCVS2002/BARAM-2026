"""NOAA GFS D-2 사이클 수집 — lagged ensemble·사이클 간 spread 피처용 (#92, 티어2).

배경(research/domain_theory.md): 이전 발표 사이클과의 차이(spread)는 예보 불확실성
신호 (HEFTCom 등에서 검증). OM Previous Runs는 2024-02부터라 main GBM 투입 불가
→ NOAA S3 원본(전 기간)에서 직접 수집.

수집: UGRD/VGRD 100m + GUST / **D-2 00UTC 사이클 f040~f063** / 9개 격자 (제공 GFS 동일).
매핑: 대상일 D의 KST H시 = D-2 00UTC + (39+H)h → f040(01시)~f063(24시).
누수: D-2 00UTC 런 발표 ≈ D-2 04~07UTC — 예측기준시점(D-1 04UTC)보다 하루 전. 안전.
재현: python scripts/collect_gfs_lag.py 2022-01-01 2025-12-31 (체크포인트 이어받기 지원)
"""

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# 라이브러리 내부 경고(herbie UserWarning, cfgrib FutureWarning 등 전부 무해) 전체 억제
# — 수집 로그의 진행도 가독성 확보
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "noaa_gfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH = ":((UGRD|VGRD):100 m above ground|GUST:surface):"
LATS = [37.0, 37.25, 37.5]
LONS = [128.75, 129.0, 129.25]
N_WORKERS = 6


def fetch_slot_inner(run: pd.Timestamp, fxx: int):
    warnings.filterwarnings("ignore")  # 워커 프로세스에도 전체 억제
    from herbie import Herbie
    H = Herbie(f"{run:%Y-%m-%d} 00:00", model="gfs", product="pgrb2.0p25",
               fxx=fxx, verbose=False)
    ds_list = H.xarray(SEARCH, remove_grib=True)
    if not isinstance(ds_list, list):
        ds_list = [ds_list]
    recs = []
    for ds in ds_list:
        for vn in ds.data_vars:
            sel = ds[vn].sel(latitude=LATS, longitude=LONS)
            for la in LATS:
                for lo in LONS:
                    recs.append({
                        "run": f"{run:%Y%m%d}00", "fxx": fxx, "var": vn,
                        "lat": la, "lon": lo,
                        "value": float(sel.sel(latitude=la, longitude=lo).values),
                    })
        ds.close()
    return recs


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
    out_path = OUT_DIR / f"gfs_lag_d2_{start:%Y%m%d}_{end:%Y%m%d}.csv"

    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path, encoding="utf-8-sig")
        rows = prev.to_dict("records")
        done = set(zip(prev["run"].astype(str), prev.fxx))
        print(f"resume: {len(done)} slots", flush=True)

    slots = []
    for day in pd.date_range(start, end):  # day = 대상일 D → run = D-2 00UTC
        run = day - pd.Timedelta(days=2)
        for fxx in range(40, 64):
            if (f"{run:%Y%m%d}00", fxx) not in done:
                slots.append((run, fxx))
    total = len(slots)
    print(f"target: {total} slots ({N_WORKERS}-way)", flush=True)

    t0, n, n_fail = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fetch_slot, run, fxx) for run, fxx in slots]
        for fut in as_completed(futs):
            run, fxx, recs, err = fut.result()
            n += 1
            if err is not None:
                n_fail += 1
                print(f"FAIL {run:%Y%m%d} f{fxx}: {err}", flush=True)
            else:
                rows.extend(recs)
            if n % 100 == 0 or n == total:
                pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                el = time.time() - t0
                eta = el / n * (total - n) / 3600
                print(f"[{n}/{total}] {n/total*100:.1f}% fail={n_fail} "
                      f"elapsed={el/3600:.1f}h ETA={eta:.1f}h", flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"완료: {len(rows)}행, 실패 {n_fail} → {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
