"""#115 예정: LDAPS 1.5km 공간장 수집 — 단지 중심 28×28 크롭 (u/v 875hPa).

대형 스윙(④): 16격자 통계를 넘는 중규모 흐름 패턴 학습용 원료.
누수 기준: tmfc = D-1 00UTC 사이클, ef=16~39 (제공 데이터와 동일 규약).
키별 전담 워커 병렬 (일 5GB/키 ≈ 4500요청). 실측 교훈(07-29): 할당량 초과는
'초과' 텍스트가 아니라 403/연결거부로 온다 → 키 사망 처리 후 자정+5분까지 대기.
저장: external_data/kma_ldaps_grid/{YYYY-MM-DD}.npz — u,v (24,28,28) float32.
순서: 2024 → 2025 → 2023 → 2022.
사용법: python collect_ldaps_grid.py --worker 0 --nworkers 3  (워커 k = 키 k 전담,
        전체 일자 목록의 index % nworkers == k 파티션 담당)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "kma_ldaps_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = []
for line in (PROJECT / ".env").read_text().splitlines():
    for name in ("KMA_APIHUB_KEY", "KMA_APIHUB_KEY2", "KMA_APIHUB_KEY3"):
        if line.startswith(name + "="):
            v = line.split("=", 1)[1].strip()
            if v:
                KEYS.append(v)
assert KEYS, "KMA API 키 없음"
print(f"키 {len(KEYS)}개 로드", flush=True)

API = "https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php"
HEADERS = {"User-Agent": "BARAM2026-data-collector/1.0 (research; contact: ljse237@gmail.com)"}
IY, IX, HALF = 363, 429, 14  # 사이트 최근접 인덱스 (프로브 실측), 28x28
EFS = list(range(16, 40))
YEARS_ORDER = [2024, 2025, 2023, 2022]

WORKER = 0
NWORKERS = 1
req_count = 0
# 실측 보정(07-30): 4500 자체 정지 시점에 마이페이지 잔여 확인됨 → 백스톱만 남김.
# 실제 한도 판정은 403 신호로 (15분 백오프 3연속 → 자정 대기)
REQ_BUDGET = 12000


def sleep_to_midnight() -> None:
    now = pd.Timestamp.now()
    nxt = now.normalize() + pd.Timedelta(days=1, minutes=5)
    global req_count
    print(f"[w{WORKER}] 키 할당량 소진(요청 {req_count}) → {nxt}까지 "
          f"{(nxt-now).total_seconds()/3600:.1f}h 대기", flush=True)
    time.sleep((nxt - now).total_seconds())
    req_count = 0


def fetch(tmfc: str, var: str, ef: int, tmp: Path) -> np.ndarray:
    """전담 키로 1필드 수집. 403/연결거부 연쇄 = 할당량 소진 → 자정까지 대기 후 재시도."""
    import xarray as xr
    global req_count
    key = KEYS[WORKER % len(KEYS)]
    conn_fail = 0
    f403 = 0
    while True:
        if req_count >= REQ_BUDGET:
            sleep_to_midnight()
        try:
            time.sleep(0.15)
            req_count += 1
            r = requests.get(API, params=dict(
                nwp="l015", sub="pres", vars=var, pres="875", tmfc=tmfc, ef=str(ef),
                dataType="GRIB", authKey=key), headers=HEADERS, timeout=180)
            if r.status_code == 200 and len(r.content) > 100000:
                tmp.write_bytes(r.content)
                ds = xr.open_dataset(tmp, engine="cfgrib", decode_timedelta=True,
                                     backend_kwargs={"indexpath": ""})
                da = ds[list(ds.data_vars)[0]]
                crop = da.values[IY - HALF:IY + HALF, IX - HALF:IX + HALF].astype(np.float32)
                ds.close()
                if crop.shape == (2 * HALF, 2 * HALF) and not np.isnan(crop).any():
                    f403 = conn_fail = 0
                    return crop
                raise RuntimeError(f"crop bad {crop.shape}")
            if r.status_code == 403:  # 일시 레이트리밋 vs 진짜 소진 구분: 15분 백오프 ×3
                f403 += 1
                if f403 >= 3:
                    sleep_to_midnight()
                    f403 = 0
                else:
                    print(f"[w{WORKER}] 403 → 15분 백오프 ({f403}/3)", flush=True)
                    time.sleep(900)
                continue
            if r.status_code == 200:  # 정상 응답인데 GRIB이 아님 (자료 없는 날 등)
                raise RuntimeError(f"non-grib response len={len(r.content)}: {r.text[:80]}")
            raise RuntimeError(f"bad status {r.status_code}")
        except requests.exceptions.RequestException:
            conn_fail += 1
            if conn_fail >= 5:  # 연결거부 연쇄 = IP/키 차단 → 자정까지 대기
                sleep_to_midnight()
                conn_fail = 0
            else:
                time.sleep(10 * conn_fail)


def main() -> None:
    global WORKER, NWORKERS
    if "--worker" in sys.argv:
        WORKER = int(sys.argv[sys.argv.index("--worker") + 1])
    if "--nworkers" in sys.argv:
        NWORKERS = int(sys.argv[sys.argv.index("--nworkers") + 1])
    t0 = time.time()
    days = []
    for y in YEARS_ORDER:
        end = pd.Timestamp(f"{y}-12-31")
        if y == 2025:
            end = min(end, pd.Timestamp.now().normalize() - pd.Timedelta(days=2))
        days += list(pd.date_range(f"{y}-01-01", end, freq="D"))
    part = [d for i, d in enumerate(days) if i % NWORKERS == WORKER]
    todo = [d for d in part if not (OUT_DIR / f"{d:%Y-%m-%d}.npz").exists()]
    print(f"[w{WORKER}/{NWORKERS}] 파티션 {len(part)}일 중 미수집 {len(todo)}일 "
          f"(키#{WORKER % len(KEYS)} 전담)", flush=True)
    tmp = OUT_DIR / f"_tmp_w{WORKER}.gb2"
    done0, fail = len(part) - len(todo), []
    for n, d in enumerate(todo):
        tmfc = f"{d - pd.Timedelta(days=1):%Y%m%d}00"  # D-1 00UTC
        try:
            u = np.stack([fetch(tmfc, "ugrd", ef, tmp) for ef in EFS])
            v = np.stack([fetch(tmfc, "vgrd", ef, tmp) for ef in EFS])
            np.savez_compressed(OUT_DIR / f"{d:%Y-%m-%d}.npz", u=u, v=v)
        except Exception as e:
            fail.append(str(d.date()))
            print(f"[w{WORKER}] {d.date()} 실패: {str(e)[:100]}", flush=True)
        if (n + 1) % 10 == 0 or n == len(todo) - 1:
            el = time.time() - t0
            rate = (n + 1) / el * 86400
            eta_d = (len(todo) - n - 1) / max(rate, 1e-9)
            print(f"[w{WORKER}] 진행 {done0+n+1}/{len(part)} ({(done0+n+1)/len(part)*100:.1f}%) "
                  f"| {rate:.0f}일/일 페이스 | 잔여 ETA {eta_d:.1f}일 | 실패 {len(fail)} "
                  f"({el/60:.0f}m)", flush=True)
    if fail:
        (OUT_DIR / f"_failed_days_w{WORKER}.txt").write_text("\n".join(fail), encoding="utf-8")
    print(f"[w{WORKER}] === 수집 종료 ===", flush=True)


if __name__ == "__main__":
    main()
