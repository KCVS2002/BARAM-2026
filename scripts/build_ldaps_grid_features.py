"""#115: LDAPS 28×28 공간장 → 물리 유도 공간 피처 (시간별 테이블, parquet 캐시).

원료: external_data/kma_ldaps_grid/{D}.npz — u,v (24,28,28) @875hPa, 1.5km.
ef 16~39 ↔ forecast_kst_dtm = D일 01시 ~ D+1일 00시 (제공 데이터 규약 동일).
사이트 = 크롭 중심 (14,14). dx=dy=1.5km.

피처 (시간별 15개):
  gws_mean/std/max_ratio  : 도메인 풍속 평균·산포·스피드업(max/mean)
  gu_mean/gv_mean         : 도메인 평균 벡터풍 (주풍향)
  gdiv/gvort              : 중앙차분 발산·와도의 도메인 평균
  gdiv_site/gvort_site    : 사이트 국소(중심 8×8) 발산·와도
  gblock_w/gblock_s       : 상류(서/남 가장자리 8열) 풍속 − 사이트(중심 4×4) 풍속
  ggrad_x/ggrad_y         : 사이트 풍속의 동서/남북 구배 (±5셀)
  gsite_rel               : 사이트 풍속 / 도메인 평균 (지형 증폭비)
  gdir_std                : 도메인 풍향 산포 (흐름 정합성)
저장: experiments/cache/ldaps_grid_features.parquet
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
GRID = PROJECT / "external_data" / "kma_ldaps_grid"
OUT = PROJECT / "experiments" / "cache" / "ldaps_grid_features.parquet"
DX = 1500.0  # m
C = 14  # 사이트 중심 인덱스


def day_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(24,28,28)×2 → (24,15)"""
    ws = np.hypot(u, v)
    du_dx = np.gradient(u, DX, axis=2)
    dv_dy = np.gradient(v, DX, axis=1)
    dv_dx = np.gradient(v, DX, axis=2)
    du_dy = np.gradient(u, DX, axis=1)
    div = du_dx + dv_dy
    vort = dv_dx - du_dy
    site = ws[:, C - 2:C + 2, C - 2:C + 2].mean((1, 2))
    dirs = np.arctan2(v, u)
    dir_mean = np.arctan2(v.mean((1, 2)), u.mean((1, 2)))
    ddir = np.angle(np.exp(1j * (dirs - dir_mean[:, None, None])))
    feats = np.column_stack([
        ws.mean((1, 2)),
        ws.std((1, 2)),
        ws.max((1, 2)) / np.maximum(ws.mean((1, 2)), 0.1),
        u.mean((1, 2)),
        v.mean((1, 2)),
        div.mean((1, 2)) * 1e4,
        vort.mean((1, 2)) * 1e4,
        div[:, C - 4:C + 4, C - 4:C + 4].mean((1, 2)) * 1e4,
        vort[:, C - 4:C + 4, C - 4:C + 4].mean((1, 2)) * 1e4,
        ws[:, :, :8].mean((1, 2)) - site,
        ws[:, -8:, :].mean((1, 2)) - site,
        (ws[:, C, C + 5] - ws[:, C, C - 5]) / (10 * DX) * 1e4,
        (ws[:, C + 5, C] - ws[:, C - 5, C]) / (10 * DX) * 1e4,
        site / np.maximum(ws.mean((1, 2)), 0.1),
        ddir.std((1, 2)),
    ])
    return feats.astype(np.float32)


COLS = ["gws_mean", "gws_std", "gws_maxr", "gu_mean", "gv_mean", "gdiv", "gvort",
        "gdiv_site", "gvort_site", "gblock_w", "gblock_s", "ggrad_x", "ggrad_y",
        "gsite_rel", "gdir_std"]


def main() -> None:
    t0 = time.time()
    rows, dtms = [], []
    files = sorted(GRID.glob("*.npz"))
    for i, f in enumerate(files):
        z = np.load(f)
        rows.append(day_features(z["u"], z["v"]))
        d = pd.Timestamp(f.stem)
        dtms.append(pd.date_range(d + pd.Timedelta(hours=1), periods=24, freq="h"))
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(np.concatenate(rows), columns=COLS)
    df.insert(0, "forecast_kst_dtm", pd.DatetimeIndex(np.concatenate([d.values for d in dtms])))
    df.to_parquet(OUT)
    print(f"저장 {OUT.name}: {len(df)}행 ({df.forecast_kst_dtm.min()} ~ {df.forecast_kst_dtm.max()})"
          f" ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
