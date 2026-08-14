"""D-2 12Z 수집 병렬 워커: 날짜를 N등분, 워커별 전용 키·전용 출력 파일.

사용법: python collect_kma_d2_worker.py <worker_id 0~2>
완료 후 merge_d2.py로 병합.
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "kma_ldaps"

WORKER = int(sys.argv[1])
N_WORKERS = 3
OUT_PATH = OUT_DIR / f"point_profile_d2_w{WORKER}.csv"
MAIN_PATH = OUT_DIR / "point_profile_d2.csv"  # 단일 수집기가 이미 받은 슬롯 재활용

KEYS = []
for line in (PROJECT / ".env").read_text().splitlines():
    if line.startswith("KMA_APIHUB_KEY"):
        KEYS.append(line.split("=", 1)[1].strip())
KEY = KEYS[WORKER % len(KEYS)]

API = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-um_grib_pt_txt1"
HEADERS = {"User-Agent": "BARAM2026-data-collector/1.0 (research; contact: ljse237@gmail.com)"}
LAT, LON = 37.2888, 128.9610
EFS = list(range(28, 49))
PRIORITY_YEARS = [2024, 2025, 2023, 2022]
ROW_RE = re.compile(r"^\d{10} ")


def sleep_until_next_day():
    now = pd.Timestamp.now()
    nxt = now.normalize() + pd.Timedelta(days=1, minutes=5)
    print(f"w{WORKER}: 할당량 대기 {(nxt-now).total_seconds()/3600:.1f}h")
    time.sleep((nxt - now).total_seconds())


def fetch(tmfc, ef):
    r = requests.get(API, params=dict(
        group="UMKR", nwp="N512", data="P", varn="2002,2003", tmfc=tmfc, hf=str(ef),
        lon=str(LON), lat=str(LAT), disp="A", authKey=KEY), headers=HEADERS, timeout=60)
    if r.status_code == 403 or "제한" in r.text[:200]:
        raise PermissionError("quota")
    if r.status_code != 200 or "ERROR" in r.text[:300] or '"status"' in r.text[:200]:
        raise RuntimeError(r.text[:100])
    recs = []
    for line in r.text.splitlines():
        if ROW_RE.match(line):
            p = line.split()
            recs.append({"tmfc": p[0], "ef": ef, "varn": int(p[2]),
                         "level_pa": int(p[3]), "value": float(p[4])})
    if not recs:
        raise RuntimeError("no rows")
    return recs


def main():
    done = set()
    rows = []
    for path in (OUT_PATH, MAIN_PATH):
        if path.exists():
            prev = pd.read_csv(path, encoding="utf-8-sig", dtype={"tmfc": str})
            done |= set(zip(prev["tmfc"], prev["ef"]))
            if path == OUT_PATH:
                rows = prev.to_dict("records")
    print(f"w{WORKER}: resume {len(done)} slots known")

    slots = []
    all_days = []
    for yr in PRIORITY_YEARS:
        all_days += list(pd.date_range(f"{yr}-01-01", f"{yr}-12-31"))
    for i, day in enumerate(all_days):
        if i % N_WORKERS != WORKER:
            continue
        tmfc = (day - pd.Timedelta(days=2)).strftime("%Y%m%d") + "12"
        for ef in EFS:
            if (tmfc, ef) not in done:
                slots.append((tmfc, ef))
    total = len(slots)
    print(f"w{WORKER}: target {total}")

    t0, n, n_fail, streak = time.time(), 0, 0, 0
    for tmfc, ef in slots:
        try:
            rows.extend(fetch(tmfc, ef))
            streak = 0
        except PermissionError:
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
            sleep_until_next_day()
            continue
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            streak += 1
            print(f"w{WORKER} FAIL {tmfc} ef{ef}: {str(e)[:70]}")
            if streak >= 10:
                pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
                time.sleep(900)
                streak = 0
        n += 1
        if n % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
            rate = n / (time.time() - t0)
            print(f"w{WORKER} [{n}/{total}] {n/total*100:.1f}% | {rate:.2f}/s | ETA {(total-n)/rate/3600:.1f}h | fails={n_fail}")
        time.sleep(0.15)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"w{WORKER} done: fails={n_fail} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
