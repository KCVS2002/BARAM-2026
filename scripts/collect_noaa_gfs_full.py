"""NOAA GFS 보조변수 전 기간 병렬 수집 (파일럿 신호검증 통과 → 확대).

파일럿 대비 변경: ProcessPoolExecutor 6-way 병렬 (S3는 동시요청 허용).
※ 스레드 병렬은 ecCodes(GRIB 파서)가 thread-unsafe라 크래시 → 프로세스 격리 필수.
수집: HPBL, SHTFL, VVEL850 / D-1 00UTC 사이클 f016~f039 / 9개 격자.

사용법: python scripts/collect_noaa_gfs_full.py 2022-01-01 2025-12-31
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.collect_noaa_gfs import OUT_DIR, fetch_day

N_WORKERS = 6


def fetch_slot(run: pd.Timestamp, fxx: int):
    for attempt in range(3):
        try:
            return run, fxx, fetch_day(run, fxx), None
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return run, fxx, [], str(e)[:120]
            time.sleep(3)
    return run, fxx, [], "unreachable"


def main() -> None:
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
    end = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2025-12-31")
    out_path = OUT_DIR / f"gfs_aux_{start:%Y%m%d}_{end:%Y%m%d}.csv"

    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path, encoding="utf-8-sig")
        rows = prev.to_dict("records")
        done = set(zip(prev["run"].astype(str), prev["fxx"]))
    # 파일럿 결과도 시드로 재활용
    pilot = OUT_DIR / "gfs_aux_20241001_20241231.csv"
    if pilot.exists() and not out_path.exists():
        prev = pd.read_csv(pilot, encoding="utf-8-sig")
        rows = prev.to_dict("records")
        done = set(zip(prev["run"].astype(str), prev["fxx"]))
        print(f"seeded from pilot: {len(done)} slots")

    slots = []
    for day in pd.date_range(start, end):
        run = day - pd.Timedelta(days=1)
        for fxx in range(16, 40):
            if (f"{run:%Y%m%d}00", fxx) not in done:
                slots.append((run, fxx))
    total = len(slots)
    print(f"target: {total} slots remain ({N_WORKERS} workers)")

    t0, n, n_fail = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(fetch_slot, run, fxx) for run, fxx in slots]
        for fut in as_completed(futures):
            run, fxx, recs, err = fut.result()
            n += 1
            if err:
                n_fail += 1
                print(f"FAIL {run:%Y%m%d} f{fxx}: {err}")
            else:
                rows.extend(recs)
            if n % 200 == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                rate = n / (time.time() - t0)
                eta_h = (total - n) / rate / 3600
                print(f"[{n}/{total}] {n/total*100:.1f}% | {rate:.2f} slot/s | "
                      f"ETA {eta_h:.1f}h | fails={n_fail}")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"done: {len(rows)} rows, fails={n_fail} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
