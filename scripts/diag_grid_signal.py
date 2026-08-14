"""#115 진단: 공간장 피처의 잔차 설명력 — 정보 우위 여부의 선행 판정.

질문: 공식 모델(124+IFS10)의 2024 OOF 잔차(r = actual − q50)를
공간장 피처가 기존 주력 피처 '대비 추가로' 설명하는가?
방법: 잔차 예측 소형 LGBM 3종 (기존만 / 공간만 / 병합) — 시간 블록 분할
(fold 01~07 학습, 09/11 검증). 한계 정보 = MAE(병합) − MAE(기존만).
부가: |r|(오차 크기) 설명력 — FICR 관련 불확실성 정보.
전부 캐시 기반 (oof_cache_q3 + ldaps_grid_features + base_frame), 학습 수 초.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame

CACHE = PROJECT / "experiments" / "oof_cache_q3"
GRIDF = PROJECT / "experiments" / "cache" / "ldaps_grid_features.parquet"
TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
EXIST = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
         "ldaps_ws50", "gfs_ws10", "ifs_ws700", "ifs_dt", "air_density", "ifs_q850"]
SMALL = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30,
             colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)


def main() -> None:
    gf = pd.read_parquet(GRIDF)
    gcols = [c for c in gf.columns if c != "forecast_kst_dtm"]
    df, _, cols, _ = load_base_frame()
    exist = [c for c in EXIST if c in df.columns]
    print(f"기존 통제 피처 {len(exist)}개: {exist}")

    for tgt in TARGETS:
        frames = []
        for m0 in range(1, 12, 2):
            fold = f"2024-{m0:02d}"
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            fr = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(z["dtm"]),
                               "r": z["actual"] - z["qp"][:, 9]})
            fr["fold"] = m0
            frames.append(fr)
        d = pd.concat(frames)
        d = d.merge(gf, on="forecast_kst_dtm", how="left")
        d = d.merge(df[["forecast_kst_dtm"] + exist].drop_duplicates("forecast_kst_dtm"),
                    on="forecast_kst_dtm", how="left")
        d = d.dropna(subset=gcols + exist)
        tr = d[d.fold <= 7]
        te = d[d.fold >= 9]
        print(f"\n=== {tgt}: 학습 {len(tr)} / 검증 {len(te)}행 ===")
        base_mae = np.abs(te["r"]).mean()
        base_std = np.abs(te["r"] - tr["r"].mean()).mean()
        print(f"  잔차 |r| 기준 MAE (0 예측): {base_mae:.0f} kWh / (학습 평균 예측): {base_std:.0f}")
        for label, use in [("기존만", exist), ("공간만", gcols), ("병합", exist + gcols)]:
            m = lgb.LGBMRegressor(objective="regression_l1", **SMALL).fit(tr[use], tr["r"])
            mae = np.abs(te["r"] - m.predict(te[use])).mean()
            print(f"  r  예측 [{label:4s}]: 검증 MAE {mae:.0f} kWh ({(base_mae-mae)/base_mae*+100:+.1f}%)")
        # |r| (오차 크기) — 불확실성 정보
        for label, use in [("기존만", exist), ("병합", exist + gcols)]:
            m = lgb.LGBMRegressor(objective="regression_l1", **SMALL).fit(tr[use], np.abs(tr["r"]))
            pred = m.predict(te[use])
            corr = np.corrcoef(pred, np.abs(te["r"]))[0, 1]
            print(f"  |r| 예측 [{label:4s}]: 검증 상관 {corr:.3f}")
        # 개별 공간 피처 상관 상위
        cors = {c: abs(np.corrcoef(d[c], d["r"])[0, 1]) for c in gcols}
        top = sorted(cors, key=cors.get, reverse=True)[:5]
        print("  공간 피처 |corr(r)| 상위:", " ".join(f"{c}:{cors[c]:.3f}" for c in top))


if __name__ == "__main__":
    main()
