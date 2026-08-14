"""KMA UM(LDAPS=UMKR 1.5km) 지점 기온 프로파일 수집기 — nph-um_grib_pt_txt1.

동기 (#66): FICR을 움직인 건 바람이 아니라 기온 구조(안정도)였다. 제공 LDAPS에는
기압면 기온이 없고(지상 2m뿐), 기존 KMA 수집(#21 기각)은 u/v만이었음 → LDAPS 1.5km
지형 해상 안정도(t875−t850 경사, 허브고도층)는 미개척 정보.

- varn=0 = 기온(K), 검증: 2024-06-01 00Z ef24 t1000=290.9K/t975=288.9K (물리적 정상).
- 한 콜에 전 기압면 프로파일 (레벨 생략), D-1 00Z 사이클(제공 데이터 동일, 누수 안전).
- 1,461일 × ef16~39 = 35,064콜 × ~3.5KB. 3키 순환, 체크포인트 이어받기.
- 산출: external_data/kma_ldaps/point_profile_t.csv (tmfc,ef,varn,level_pa,value)
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "kma_ldaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "point_profile_t.csv"

KEYS = []
for name in ("KMA_APIHUB_KEY", "KMA_APIHUB_KEY2", "KMA_APIHUB_KEY3"):
    for line in (PROJECT / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            KEYS.append(line.split("=", 1)[1].strip())
assert KEYS, "no API keys in .env"

API = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-um_grib_pt_txt1"
HEADERS = {"User-Agent": "BARAM2026-data-collector/1.0 (research; contact: ljse237@gmail.com)"}
LAT, LON = 37.2888, 128.9610  # 기존 수집과 동일 격자 (rank 0)
EFS = list(range(16, 40))
PRIORITY_YEARS = [2024, 2025, 2023, 2022]
ROW_RE = re.compile(r"^\d{10} ")


class QuotaError(RuntimeError):
    pass


def sleep_until_next_day() -> None:
    now = pd.Timestamp.now()
    nxt = now.normalize() + pd.Timedelta(days=1, minutes=5)
    print(f"모든 키 할당량 소진 → {nxt}까지 {(nxt-now).total_seconds()/3600:.1f}h 대기", flush=True)
    time.sleep((nxt - now).total_seconds())


def fetch(tmfc: str, ef: int, key: str) -> list[dict]:
    r = requests.get(API, params=dict(
        group="UMKR", nwp="N512", data="P", varn="0", tmfc=tmfc, hf=str(ef),
        lon=str(LON), lat=str(LAT), disp="A", authKey=key), headers=HEADERS, timeout=60)
    if r.status_code == 403 or "제한" in r.text[:200]:
        raise QuotaError(key)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:80]}")
    if '"status"' in r.text[:200] or "ERROR" in r.text[:300]:
        raise RuntimeError(f"API error: {r.text[:120]}")
    recs = []
    for line in r.text.splitlines():
        if not ROW_RE.match(line):
            continue
        p = line.split()
        recs.append({"tmfc": p[0], "ef": ef, "varn": int(p[2]),
                     "level_pa": int(p[3]), "value": float(p[4])})
    if not recs:
        raise RuntimeError(f"no data rows: {r.text[:120]}")
    return recs


def self_check() -> None:
    """기온 물리 범위 검증 (200~330K, 상층일수록 저온 경향)."""
    for key in KEYS:
        try:
            recs = fetch("2024093000", 16, key)
            break
        except QuotaError:
            print(f"key ...{key[-4:]} 할당량 소진 — 다음 키로", flush=True)
    else:
        sleep_until_next_day()
        recs = fetch("2024093000", 16, KEYS[0])
    d = {r["level_pa"]: r["value"] for r in recs}
    t875, t500 = d.get(87500), d.get(50000)
    assert t875 and 240 < t875 < 320, f"t875 이상: {t875}"
    assert t500 and t500 < t875, f"연직 감률 이상: t875={t875} t500={t500}"
    print(f"self-check OK: t875={t875:.2f}K t500={t500:.2f}K (레벨 {len(d)}개)", flush=True)


def main() -> None:
    done = set()
    rows = []
    if OUT_PATH.exists():
        prev = pd.read_csv(OUT_PATH, encoding="utf-8-sig", dtype={"tmfc": str})
        rows = prev.to_dict("records")
        done = set(zip(prev["tmfc"], prev["ef"]))
        print(f"resume: {len(done)} slots", flush=True)

    self_check()

    slots = []
    for yr in PRIORITY_YEARS:
        for day in pd.date_range(f"{yr}-01-01", f"{yr}-12-31"):
            tmfc = (day - pd.Timedelta(days=1)).strftime("%Y%m%d") + "00"
            for ef in EFS:
                if (tmfc, ef) not in done:
                    slots.append((tmfc, ef))
    total = len(slots)
    print(f"target: {total} requests (키 {len(KEYS)}개 순환)", flush=True)

    t0, n, n_fail, streak = time.time(), 0, 0, 0
    live_keys = list(KEYS)
    for tmfc, ef in slots:
        while True:
            if not live_keys:
                pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
                sleep_until_next_day()
                live_keys = list(KEYS)
            key = live_keys[n % len(live_keys)]
            try:
                rows.extend(fetch(tmfc, ef, key))
                streak = 0
                break
            except QuotaError:
                print(f"key ...{key[-4:]} 할당량 소진 (잔여 키 {len(live_keys)-1})", flush=True)
                live_keys.remove(key)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                streak += 1
                print(f"FAIL {tmfc} ef{ef}: {str(e)[:90]}", flush=True)
                if streak >= 10:
                    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
                    print("연속 실패 10 — 30분 대기", flush=True)
                    time.sleep(1800)
                    streak = 0
                break
        n += 1
        if n % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
            rate = n / (time.time() - t0)
            eta_h = (total - n) / rate / 3600
            print(f"[{n}/{total}] {n/total*100:.1f}% | {rate:.2f} req/s | ETA {eta_h:.1f}h | fails={n_fail}",
                  flush=True)
        time.sleep(0.15)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"done: {n} req, fails={n_fail} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
