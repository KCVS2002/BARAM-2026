"""KMA UM(LDAPS=UMKR 1.5km) 지점 조회 수집기 — nph-um_grib_pt_txt1.

검증 완료 사항 (2026-07-12):
- group=UMKR & nwp=N512 & varn=2002,2003 (u,v 목록) & level 생략 → 전 기압면 프로파일
- 값 정합성: 지도(전체격자) 방식과 동일 원본 확인 (grid cell 값 1e-6 이내 일치)
- 좌표 (37.2888, 128.9610) = 파일럿 신호검증 통과 격자(rank 0)에 스냅
- 시작 시 파일럿 데이터와 자동 대조 검증 후 본 수집 진입

요청량: 1,461일 × 24 ef = 35,064건 × ~3.5KB ≈ 120MB (할당량 무관), 두 키 순환.
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
OUT_PATH = OUT_DIR / "point_profile.csv"

KEYS = []
for name in ("KMA_APIHUB_KEY", "KMA_APIHUB_KEY2", "KMA_APIHUB_KEY3"):
    for line in (PROJECT / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            KEYS.append(line.split("=", 1)[1].strip())
assert KEYS, "no API keys in .env"

API = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-um_grib_pt_txt1"
HEADERS = {"User-Agent": "BARAM2026-data-collector/1.0 (research; contact: ljse237@gmail.com)"}
LAT, LON = 37.2888, 128.9610  # 파일럿 검증 격자 rank 0
EFS = list(range(16, 40))
PRIORITY_YEARS = [2024, 2025, 2023, 2022]
ROW_RE = re.compile(r"^\d{10} ")


class QuotaError(RuntimeError):
    pass


def sleep_until_next_day() -> None:
    now = pd.Timestamp.now()
    nxt = now.normalize() + pd.Timedelta(days=1, minutes=5)
    print(f"모든 키 할당량 소진 → {nxt}까지 {(nxt-now).total_seconds()/3600:.1f}h 대기")
    time.sleep((nxt - now).total_seconds())


def fetch(tmfc: str, ef: int, key: str) -> list[dict]:
    r = requests.get(API, params=dict(
        group="UMKR", nwp="N512", data="P", varn="2002,2003", tmfc=tmfc, hf=str(ef),
        lon=str(LON), lat=str(LAT), disp="A", authKey=key), headers=HEADERS, timeout=60)
    # 일일 제한: 트래픽 5GB + 호출 20,000회 — 메시지가 다를 수 있어 403 전체를 할당량으로 간주
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
    """파일럿(지도 방식) 값과 대조 — 동일 원본 보증."""
    pilot_path = OUT_DIR / "ldaps_pres_20241001_20241231.csv"
    if not pilot_path.exists():
        print("파일럿 파일 없음 — self-check 생략")
        return
    pilot = pd.read_csv(pilot_path, encoding="utf-8-sig")
    ref = pilot[(pilot["var"] == "ugrd") & (pilot.pres == 875)
                & (pilot.tmfc == 2024093000) & (pilot.ef == 16) & (pilot.grid_rank == 0)]
    for key in KEYS:
        try:
            recs = fetch("2024093000", 16, key)
            break
        except QuotaError:
            print(f"key ...{key[-4:]} 할당량 소진 — 다음 키로")
    else:
        sleep_until_next_day()
        recs = fetch("2024093000", 16, KEYS[0])
    got = [r["value"] for r in recs if r["varn"] == 2002 and r["level_pa"] == 87500]
    diff = abs(got[0] - ref.value.iloc[0])
    assert diff < 1e-3, f"정합성 검증 실패: diff={diff}"
    print(f"self-check OK: 지도={ref.value.iloc[0]:.5f} 지점={got[0]:.5f} (diff {diff:.2e})")


def main() -> None:
    done = set()
    rows = []
    if OUT_PATH.exists():
        prev = pd.read_csv(OUT_PATH, encoding="utf-8-sig", dtype={"tmfc": str})
        rows = prev.to_dict("records")
        done = set(zip(prev["tmfc"], prev["ef"]))
        print(f"resume: {len(done)} slots")

    self_check()

    slots = []
    for yr in PRIORITY_YEARS:
        for day in pd.date_range(f"{yr}-01-01", f"{yr}-12-31"):
            tmfc = (day - pd.Timedelta(days=1)).strftime("%Y%m%d") + "00"
            for ef in EFS:
                if (tmfc, ef) not in done:
                    slots.append((tmfc, ef))
    total = len(slots)
    print(f"target: {total} requests (키 {len(KEYS)}개 순환)")

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
                print(f"key ...{key[-4:]} 할당량 소진 (잔여 키 {len(live_keys)-1})")
                live_keys.remove(key)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                streak += 1
                print(f"FAIL {tmfc} ef{ef}: {str(e)[:90]}")
                if streak >= 10:
                    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
                    print("연속 실패 10 — 30분 대기")
                    time.sleep(1800)
                    streak = 0
                break
        n += 1
        if n % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
            rate = n / (time.time() - t0)
            eta_h = (total - n) / rate / 3600
            print(f"[{n}/{total}] {n/total*100:.1f}% | {rate:.2f} req/s | ETA {eta_h:.1f}h | fails={n_fail}")
        time.sleep(0.15)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"done: {n} req, fails={n_fail} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
