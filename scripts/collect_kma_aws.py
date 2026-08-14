"""KMA AWS/ASOS 시간별 실측 수집기 (태백 가덕산 인근 지점) — 2키 병렬 워커판.

- API: apihub typ01/url/awsh.php (매시 정시, 전 지점 응답).
- 실행: python collect_kma_aws.py --worker 0  (KEY3 전용, 시간축 짝수 슬롯)
        python collect_kma_aws.py --worker 1  (KEY2 전용, 시간축 홀수 슬롯)
  워커별 전용 키·전용 출력 파일(aws_hourly_w{N}.csv)로 충돌 없음. KEY1은 소진되어 제외.
- 지점 화이트리스트(풍력단지 37.2888,128.9610 인근 산악/기준 지점):
    216 태백(ASOS)  100  116  314(~1500m)  315 매봉산(1088m)  316 함백산(912m)  320 백운산(1264m)  45
- 산출 병합: aws_hourly.csv(1차 수집분) + aws_hourly_w0.csv + aws_hourly_w1.csv
- 체크포인트: 세 파일의 tm 합집합은 건너뜀 (이어받기).
- 키 불가(403 쿼터/미승인, 연결 홀딩 타임아웃) 시 30분 대기 후 재시도 (자정 리셋 자동 포착).
- 누수 근거: AWS/ASOS 실측은 관측 즉시 공개되는 실시간 자료. 피처 단계에서 예측기준시점
  (D-1 13:00) 이전 관측만 사용하도록 lag 처리한다 (여기서는 원자료만 수집).
"""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "kma_aws"
ALL_CSVS = [OUT_DIR / "aws_hourly.csv", OUT_DIR / "aws_hourly_w0.csv", OUT_DIR / "aws_hourly_w1.csv"]
STATIONS = {45, 100, 116, 216, 314, 315, 316, 320}
COLS = ["tm", "stn", "ta", "wd", "ws", "rn_hr1", "hm", "pa"]
URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"
KEY_BY_WORKER = {0: "KMA_APIHUB_KEY3", 1: "KMA_APIHUB_KEY2"}


def load_key(name: str) -> str:
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f".env에 {name} 없음")


class QuotaError(Exception):
    pass


def fetch_hour(session: requests.Session, key: str, tm: str) -> list[list]:
    try:
        r = session.get(URL, params={"tm": tm, "stn": 0, "help": 0, "authKey": key}, timeout=25)
    except (requests.Timeout, requests.ConnectionError) as e:
        # 쿼터 소진 키는 서버가 연결을 홀딩함 → 키 수준 불가로 취급
        raise QuotaError(f"timeout/refused: {e}") from e
    txt = r.text
    if r.status_code == 403 or "제한" in txt or "활용신청" in txt:
        raise QuotaError(txt[:200])
    rows = []
    for line in txt.splitlines():
        if not line[:1].isdigit():
            continue
        p = line.split()
        if len(p) < 10:
            continue
        stn = int(p[1])
        if stn in STATIONS:
            # YYMMDDHHMI STN TA WD WS RN_DAY RN_HR1 HM PA PS
            rows.append([p[0], stn, p[2], p[3], p[4], p[6], p[7], p[8]])
    return rows


def wait_retry(worker: int) -> None:
    print(f"[w{worker}] 키 사용 불가 → 30분 대기 후 재시도 ({datetime.now():%H:%M})", flush=True)
    time.sleep(1800)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True, choices=[0, 1])
    args = ap.parse_args()
    wk = args.worker
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = load_key(KEY_BY_WORKER[wk])
    out_csv = OUT_DIR / f"aws_hourly_w{wk}.csv"

    hours = pd.date_range("2022-01-01 00:00", "2025-12-31 23:00", freq="h")
    done: set[str] = set()
    for p in ALL_CSVS:
        if p.exists():
            done |= set(pd.read_csv(p, encoding="utf-8-sig", usecols=["tm"], dtype=str)["tm"].unique())
    if not out_csv.exists():
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(COLS)
    todo = [f"{h:%Y%m%d%H%M}" for h in hours if f"{h:%Y%m%d%H%M}" not in done][wk::2]
    total = len(todo)
    print(f"[w{wk}] 키={KEY_BY_WORKER[wk]} | 담당 {total}", flush=True)

    session = requests.Session()
    t0 = time.time()
    fails = 0
    MIN_INTERVAL = 0.15  # 워커당 ≤6.7콜/s → 2워커 합 ≤13콜/s (밴 임계 23콜/s 미만)
    FLUSH_EVERY = 100
    i = 0
    buf = []
    fout = open(out_csv, "a", newline="", encoding="utf-8-sig")
    fw = csv.writer(fout)
    while i < len(todo):
        tm = todo[i]
        t_call = time.time()
        try:
            rows = fetch_hour(session, key, tm)
        except QuotaError:
            if buf:  # 버퍼 저장 후 대기
                fw.writerows(buf); fout.flush(); buf = []
            wait_retry(wk)
            t0 = time.time()
            todo, i = todo[i:], 0
            total = len(todo)
            continue
        except Exception as e:
            fails += 1
            print(f"[w{wk}] 오류({fails}): {type(e).__name__} {str(e)[:80]} → 30s 후 재시도", flush=True)
            time.sleep(30)
            continue
        buf.extend(rows)
        i += 1
        if i % FLUSH_EVERY == 0:
            fw.writerows(buf); fout.flush(); buf = []
            rate = i / max(time.time() - t0, 1)
            eta = (len(todo) - i) / max(rate, 1e-9) / 3600
            print(f"[w{wk}] [{i}/{total}] {100*i/total:.1f}% | {rate:.1f}/s | ETA {eta:.1f}h | fails={fails}",
                  flush=True)
        dt = time.time() - t_call
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)
    if buf:
        fw.writerows(buf)
    fout.flush(); fout.close()
    print(f"[w{wk}] 완료", flush=True)


if __name__ == "__main__":
    main()
