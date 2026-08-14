"""ECMWF IFS 오픈데이터 wave-3 수집기 (2t 전 기간 + 100m 바람·연직속도 2024-04+).

- 가용성 실측 (2026-07-18): 2t는 2022-01-25부터 전 기간 존재.
  100u/100v(sfc)·w(925/850hPa)는 2024-04~05 공개 확대 이후에만 존재
  (wave-1 주석 "100u/v 사용 불가"는 2022~23 구간 관찰의 오기).
- 용도: ifs_inv = t925(wave-2) − 2t 접지역전(안정도 3탄, main GBM 후보).
  100m/w는 전 기간 불가 → main GBM 금지, 2024+ sister류 실험 재료로 비축.
- 스텝 f15~f39 3h. 산출: external_data/ecmwf_ifs/ifs_point3_2022_2025.csv
  (run,fxx,var,lat,lon,value; var ∈ t2,u100,v100,w925,w850)
- 누수 근거: D-1 00Z 런, 공개 ~D-1 08시 < 13:00 cutoff. CC-BY-4.0.
- ENS 수집기(worker 3)와 병행 → worker 2, 503 지수 백오프+지터.
"""

import csv
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "external_data" / "ecmwf_ifs"
OUT_CSV = OUT_DIR / "ifs_point3_2022_2025.csv"
COLS = ["run", "fxx", "var", "lat", "lon", "value"]
FXX = list(range(15, 40, 3))
LAT0, LAT1, LON0, LON1 = 37.6, 37.0, 128.7, 129.3
RUNS = pd.date_range("2022-01-25", "2025-12-30", freq="D")
EXT_FROM = "2024-04-01"  # 100u/v·w 공개 확대 경계 (이전 슬롯은 2t만 시도)


def fetch_one(args):
    run_s, fxx = args
    warnings.filterwarnings("ignore")
    from herbie import Herbie
    import random
    searches = [":2t:sfc:"]
    if run_s >= EXT_FROM:
        searches += [":(100u|100v):sfc:", ":w:(925|850):"]
    for attempt in range(7):
        rows = []
        try:
            H = Herbie(run_s, model="ifs", product="oper", fxx=fxx, verbose=False,
                       priority=["aws", "azure", "google"])
            if not H.grib:
                return run_s, fxx, "nofile", rows
            parsed_any = False
            for search in searches:
                try:
                    H2 = Herbie(run_s, model="ifs", product="oper", fxx=fxx, verbose=False,
                                priority=["aws", "azure", "google"])
                    dss = H2.xarray(search, remove_grib=True)
                except Exception as pe:
                    if "503" in str(pe) or "Slow Down" in str(pe) or "429" in str(pe):
                        raise
                    continue  # 검색 단위 격리 (변수 부재 등)
                if not isinstance(dss, list):
                    dss = [dss]
                try:
                    for ds in dss:
                        sub = ds.sel(latitude=slice(LAT0, LAT1), longitude=slice(LON0, LON1))
                        levc = sub.coords.get("isobaricInhPa")
                        if levc is None:  # 지표면
                            lev_list = [(None, False)]
                        elif levc.ndim == 0:  # 스칼라 좌표: sel 불가·불필요
                            lev_list = [(float(levc.values), False)]
                        else:  # 다중 레벨 차원 (w 925+850이 한 ds로 병합됨) → 전 레벨 순회
                            lev_list = [(float(v), True) for v in np.atleast_1d(levc.values)]
                        for lev, need_sel in lev_list:
                            s2 = sub.sel(isobaricInhPa=lev) if need_sel else sub
                            for dv in s2.data_vars:
                                if dv == "t2m":
                                    name = "t2"
                                elif dv in ("u100", "v100"):
                                    name = dv
                                elif dv == "w" and int(lev or 0) in (925, 850):
                                    name = f"w{int(lev)}"
                                else:
                                    continue
                                arr = s2[dv]
                                las, los = np.meshgrid(arr.latitude.values, arr.longitude.values,
                                                       indexing="ij")
                                for la, lo, val in zip(las.ravel(), los.ravel(), arr.values.ravel()):
                                    rows.append([run_s, fxx, name, round(float(la), 2),
                                                 round(float(lo), 2), round(float(val), 6)])
                                parsed_any = True
                except Exception:
                    continue
            if parsed_any:
                return run_s, fxx, "ok", rows
            return run_s, fxx, "err:parse-empty", rows
        except Exception as e:
            msg = str(e)
            if "503" in msg or "Slow Down" in msg or "429" in msg:
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
