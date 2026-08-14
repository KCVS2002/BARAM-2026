"""v12: 터빈 단위 bottom-up 학습.

- 타깃: SCADA 터빈별 시간 에너지 (17기, 이상치 정제 후 10분→시간 합)
- 모델: 터빈 스택 공유 퀀타일 LGBM (터빈 정격/로터/좌표 피처)
- 집계: 그룹 내 터빈들의 동일 분위수 합 (comonotonic 근사 — 인접 터빈 고상관)
        + 학습기간 (라벨 합 / SCADA 합) 비율로 스케일 보정
- 평가: 그룹 라벨 기준 기존 metric, 기준 0.6346 (GBM 단독) 및 0.6470(AnEn 블렌드) 비교
- 추가: 터빈 bottom-up 분위수를 제3의 원자 소스로 AnEn 블렌드에 결합한 변형도 평가
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, QUANTILES
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"

TURBINES = ([("vestas", i, "kpx_group_1") for i in range(1, 7)]
            + [("vestas", i, "kpx_group_2") for i in range(7, 13)]
            + [("unison", i, "kpx_group_3") for i in range(1, 6)])
RATED = {"vestas": 3600, "unison": 4200}
ROTOR = {"vestas": 126, "unison": 136}


def load_turbine_hourly():
    frames = []
    for maker in ("vestas", "unison"):
        sc = pd.read_csv(DATA / f"train/scada_{maker}_train.csv", encoding="utf-8-sig",
                         parse_dates=["kst_dtm"])
        pw_cols = [c for c in sc.columns if "power" in c]
        x = sc[pw_cols].where((sc[pw_cols] > -100) & (sc[pw_cols] < 1000))
        sc[pw_cols] = x
        sc["hour_end"] = sc["kst_dtm"].dt.ceil("h")
        agg = sc.groupby("hour_end")[pw_cols].sum(min_count=5)
        frames.append(agg)
    return pd.concat(frames, axis=1)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    tb = load_turbine_hourly()
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(tb, left_on="forecast_kst_dtm", right_index=True, how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    print(f"ready ({time.time()-t0:.0f}s)")

    fold_scores = {}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr = df[df.forecast_kst_dtm < va_start]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        # 터빈 스택 학습 데이터
        xs, ys = [], []
        for maker, num, grp in TURBINES:
            col = f"{maker}_wtg{num:02d}_power_kw10m"
            rated = RATED[maker]
            m = tr[col].notna()
            X = tr.loc[m, base_cols].copy()
            X["t_rated"] = rated
            X["t_rotor"] = ROTOR[maker]
            X["t_id"] = TURBINES.index((maker, num, grp))
            xs.append(X)
            ys.append((tr.loc[m, col] / (rated / 6 * 6)).clip(0, 1.2))  # 시간 kWh / 정격시간에너지
        X_all, y_all = pd.concat(xs), pd.concat(ys)
        shared = base_cols + ["t_rated", "t_rotor", "t_id"]
        models = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
            m.fit(X_all[shared], y_all)
            models.append(m)

        fs = []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            grp_turbines = [(mk, n) for mk, n, g in TURBINES if g == tgt]
            vam = va[tgt].notna()
            sub = va.loc[vam].sort_values("forecast_kst_dtm")
            qp_sum = np.zeros((len(sub), len(QUANTILES)))
            for mk, n in grp_turbines:
                rated = RATED[mk]
                Xv = sub[base_cols].copy()
                Xv["t_rated"], Xv["t_rotor"] = rated, ROTOR[mk]
                Xv["t_id"] = TURBINES.index((mk, n, tgt))
                qp_t = np.column_stack([np.clip(m.predict(Xv[shared]), 0, 1.2) * rated for m in models])
                qp_sum += qp_t
            # 스케일 보정: 학습기간 라벨합/SCADA합
            grp_cols = [f"{mk}_wtg{n:02d}_power_kw10m" for mk, n in grp_turbines]
            both = tr.dropna(subset=[tgt] + grp_cols)
            ratio = both[tgt].sum() / both[grp_cols].sum(axis=1).sum()
            qp_sum = np.clip(qp_sum * ratio, 0, cap)
            qp_sum.sort(axis=1)
            qp_sum = np.sort(smooth_quantiles_by_day(qp_sum, sub["forecast_kst_dtm"]), axis=1)
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(interp_atoms(qp_sum), cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            fs.append(s)
        fold = f"{va_start:%Y-%m}"
        fold_scores[fold] = np.nanmean(fs)
        print(f"fold {fold}: {fold_scores[fold]:.4f} ({time.time()-t0:.0f}s)")

    print(f"\n=== v12 터빈 bottom-up: {np.mean(list(fold_scores.values())):.4f} "
          f"(GBM 단독 기준 0.6346) ===")


if __name__ == "__main__":
    main()
