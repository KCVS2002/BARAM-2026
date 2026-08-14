"""ECMWF ENS 멤버 서브셋 수집기 (앙상블 평균 계산용, 10m u/v).

- enfo 스트림에 ens-mean(em) 메시지 없음 → 멤버 서브셋(cf + pf 7개 = 8멤버)의
  평균으로 근사 (51멤버 평균과 상관 ~0.97 수준).
- 인덱스 형식: cf ':10u:sfc:g:...:cf:enfo:', pf ':10u:sfc:{N}:g:...:pf:enfo:'
- 스텝 f15~f39 3h, 9격자(0.25°)/4격자(0.4°) 지점 추출. 슬롯당 다운로드 ~14MB(폐기).
- wave-2 완료(2026-07-18) 후 worker 3으로 증속 (S3 대역 단독 사용).
- 산출: external_data/ecmwf_ifs/ens_point_2022_2025.csv (run,fxx,member,var,lat,lon,value)
- 누수 근거: ENS도 D-1 00Z 런, 공개 ~D-1 08~09시 < 13:00 cutoff. CC-BY-4.0.
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
OUT_CSV = OUT_DIR / "ens_point_2022_2025.csv"
COLS = ["run", "fxx", "member", "var", "lat", "lon", "value"]
FXX = list(range(15, 40, 3))
MEMBERS = ["cf", "p6", "p12", "p18", "p25", "p31", "p37", "p44"]
LAT0, LAT1, LON0, LON1 = 37.6, 37.0, 128.7, 129.3
RUNS = pd.date_range("2022-01-25", "2025-12-30", freq="D")


# 8멤버 결합 검색: 인덱스 1회 + 다운로드 1회 (멤버별 8회 대비 ~3배 가속, 2026-07-20)
COMBINED = r":(10u|10v):sfc:(g:.*:cf:enfo:|(" + "|".join(m[1:] for m in MEMBERS if m != "cf") \
           + r"):g:.*:pf:enfo:)"


def fetch_one(args):
    run_s, fxx = args
    warnings.filterwarnings("ignore")
    from herbie import Herbie
    import random
    for attempt in range(7):
        rows = []
        try:
            H = Herbie(run_s, model="ifs", product="enfo", fxx=fxx, verbose=False,
                       priority=["aws", "azure", "google"])
            if not H.grib:
                return run_s, fxx, "nofile", []
            dss = H.xarray(COMBINED, remove_grib=True)
            if not isinstance(dss, list):
                dss = [dss]
            ok_any = False
            for ds in dss:
                try:
                    sub = ds.sel(latitude=slice(LAT0, LAT1), longitude=slice(LON0, LON1))
                    numc = sub.coords.get("number")
                    if numc is None or numc.ndim == 0:  # cf (number=0 스칼라)
                        mem_list = [("cf", None)]
                    else:  # pf: number 차원 (7멤버)
                        mem_list = [(f"p{int(nv)}", int(nv)) for nv in numc.values]
                    for m, nv in mem_list:
                        s2 = sub.sel(number=nv) if nv is not None else sub
                        for dv in ("u10", "v10"):
                            if dv not in s2:
                                continue
                            arr = s2[dv]
                            las, los = np.meshgrid(arr.latitude.values, arr.longitude.values,
                                                   indexing="ij")
                            for la, lo, val in zip(las.ravel(), los.ravel(), arr.values.ravel()):
                                rows.append([run_s, fxx, m, dv, round(float(la), 2),
                                             round(float(lo), 2), round(float(val), 3)])
                        ok_any = True
                except Exception:
                    continue  # ds 단위 격리
            if ok_any:
                return run_s, fxx, "ok", rows
            return run_s, fxx, "err:parse-empty", []
        except Exception as e:
            msg = str(e)
            if "503" in msg or "Slow Down" in msg or "429" in msg:
                time.sleep(min(15 * (2 ** attempt), 180) + random.uniform(0, 15))
                continue
            if attempt < 2:
                time.sleep(5)
                continue
            return run_s, fxx, f"err:{type(e).__name__}:{msg[:60]}", []
    return run_s, fxx, "err:503-exhausted", []


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
    print(f"전체 {total} | 완료 {n0} | 남음 {len(tasks)} | 멤버 {MEMBERS}", flush=True)

    t0 = time.time()
    fails = 0
    ndone = 0
    with ProcessPoolExecutor(max_workers=3) as ex, \
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
            if ndone % 50 == 0:
                f.flush()
                rate = ndone / max(time.time() - t0, 1)
                eta = (len(tasks) - ndone) / max(rate, 1e-9) / 3600
                print(f"[{n0+ndone}/{total}] {100*(n0+ndone)/total:.1f}% | {rate:.2f}/s | "
                      f"ETA {eta:.1f}h | fails={fails}", flush=True)
    print(f"완료: fails={fails}", flush=True)


if __name__ == "__main__":
    main()
