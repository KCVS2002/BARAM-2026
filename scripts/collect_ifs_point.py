"""ECMWF IFS 오픈데이터 지점 수집기 (제3의 NWP 소스).

- 소스: ECMWF open data (CC-BY-4.0, 실시간 공개) — aws/google/azure 아카이브,
  Herbie(model='ifs', product='oper')로 접근. 커버리지 2022-01-25 ~ 현재.
- 누수 근거: D-1 00Z 런은 D-1 아침(~08시)에 공개 → 예측기준시점(D-1 13:00) 이전 ✓.
  제공 LDAPS/GFS와 동일한 D-1 00Z 사이클만 수집.
- 변수: 10u/10v(sfc), u/v @925/850hPa (100u/v는 2025 사이클에서 제외되어 사용 불가).
  산악 단지(~1,100m ≈ 880hPa)에는 925/850이 적합.
- 스텝: f15~f39 3h 간격 (대상일 KST 00~24시) → 피처 단계에서 1h 보간.
- 산출: external_data/ecmwf_ifs/ifs_point_2022_2025.csv
  (run, fxx, var, lat, lon, value; var ∈ u10,v10,u925,v925,u850,v850)
- 체크포인트: (run,fxx) 단위 이어받기. 503 Slow Down은 지수 백오프 재시도.
- eccodes는 스레드-비안전 → ProcessPoolExecutor.
"""

import csv
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "ecmwf_ifs"
OUT_CSV = OUT_DIR / "ifs_point_2022_2025.csv"
COLS = ["run", "fxx", "var", "lat", "lon", "value"]
FXX = list(range(15, 40, 3))
SEARCH = r"(:10u:|:10v:|:[uv]:.*(925|850))"
LAT0, LAT1, LON0, LON1 = 37.6, 37.0, 128.7, 129.3  # slice(내림차순 lat); 0.25°: 3×3, 0.4°: 2×2
RUNS = pd.date_range("2022-01-25", "2025-12-30", freq="D")


def fetch_one(args):
    run_s, fxx = args
    warnings.filterwarnings("ignore")
    from herbie import Herbie
    import random
    for attempt in range(7):
        rows = []
        try:
            H = Herbie(run_s, model="ifs", product="oper", fxx=fxx, verbose=False,
                       priority=["aws", "azure", "google"])
            if not H.grib:
                return run_s, fxx, "nofile", rows
            # 지표면(10u,10v)과 기압면(u,v@925,850)을 분리 검색 → cfgrib 멀티레벨 병합
            # 실패(구 0.4° 격자의 isobaricInhPa 인덱싱 오류)를 서로 격리
            parsed_any = False
            level_err = False
            for search in (":(10u|10v):", ":(u|v):(925|850):"):
                try:
                    dss = H.xarray(search, remove_grib=True)
                except Exception as pe:
                    if "503" in str(pe) or "Slow Down" in str(pe) or "429" in str(pe):
                        raise
                    level_err = True
                    continue
                if not isinstance(dss, list):
                    dss = [dss]
                try:  # sel/좌표 파싱 오류(구 0.4° 격자의 isobaricInhPa 인덱스)도 격리
                    for ds in dss:
                        sub = ds.sel(latitude=slice(LAT0, LAT1), longitude=slice(LON0, LON1))
                        levs = ([None] if "isobaricInhPa" not in sub.coords
                                else np.atleast_1d(sub.isobaricInhPa.values).tolist())
                        for lev in levs:
                            s2 = sub if lev is None else sub.sel(isobaricInhPa=lev)
                            for dv in s2.data_vars:
                                if dv in ("u10", "v10"):
                                    name = dv
                                elif dv in ("u", "v") and int(lev or 0) in (925, 850):
                                    name = f"{dv}{int(lev)}"
                                else:
                                    continue
                                arr = s2[dv]
                                las, los = np.meshgrid(arr.latitude.values, arr.longitude.values,
                                                       indexing="ij")
                                for la, lo, val in zip(las.ravel(), los.ravel(), arr.values.ravel()):
                                    rows.append([run_s, fxx, name, round(float(la), 2),
                                                 round(float(lo), 2), round(float(val), 3)])
                                parsed_any = True
                except Exception:
                    level_err = True
                    continue
            if parsed_any:
                return run_s, fxx, ("ok" if not level_err else "ok-nolevel"), rows
            return run_s, fxx, "err:parse-empty", rows
        except Exception as e:
            msg = str(e)
            if "503" in msg or "Slow Down" in msg or "429" in msg:
                # 지터로 워커 동기화(thundering herd) 방지: 15,30,60,120,180,180... + rand
                time.sleep(min(15 * (2 ** attempt), 180) + random.uniform(0, 15))
                continue
            if attempt < 2:
                time.sleep(5)
                continue
            return run_s, fxx, f"err:{type(e).__name__}:{msg[:60]}", rows
    return run_s, fxx, "err:503-exhausted", rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_CSV.exists():
        d = pd.read_csv(OUT_CSV, encoding="utf-8-sig", usecols=["run", "fxx"])
        done = set(zip(d.run.astype(str), d.fxx.astype(int)))
    else:
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(COLS)
    tasks = [(f"{r:%Y-%m-%d}", fx) for r in RUNS for fx in FXX
             if (f"{r:%Y-%m-%d}", fx) not in done]
    total = len(RUNS) * len(FXX)
    n0 = total - len(tasks)
    print(f"전체 {total} | 완료 {n0} | 남음 {len(tasks)}", flush=True)

    t0 = time.time()
    fails = 0
    ndone = 0
    with ProcessPoolExecutor(max_workers=2) as ex, \
            open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        futs = [ex.submit(fetch_one, t) for t in tasks]
        for fut in as_completed(futs):
            run_s, fxx, status, rows = fut.result()
            if status == "ok":
                w.writerows(rows)
            else:
                fails += 1
                if fails <= 30 or fails % 50 == 0:
                    print(f"실패 {run_s} f{fxx}: {status}", flush=True)
            ndone += 1
            if ndone % 100 == 0:
                f.flush()
                rate = ndone / max(time.time() - t0, 1)
                eta = (len(tasks) - ndone) / max(rate, 1e-9) / 3600
                print(f"[{n0+ndone}/{total}] {100*(n0+ndone)/total:.1f}% | {rate:.2f}/s | "
                      f"ETA {eta:.1f}h | fails={fails}", flush=True)
    print(f"완료: fails={fails}", flush=True)


if __name__ == "__main__":
    main()
