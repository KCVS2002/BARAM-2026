"""v8: KMA LDAPS 875hPa 통합 실험 (기여 분리 설계).

  v8a: 기존 + KMA 연직 프로파일 피처
  v8b: v8a + 파워커브 프라이어 (fold-train isotonic: ws875 → cf)
  v8c: v8b + 최근성 가중(v9b) + 꼬리 분위(q0.02/0.98)

기준: v2b + smooth/interp = 0.6346. 판정: fold 일관성 4/6 이상 + 평균 개선.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, QUANTILES, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.external_kma import build_kma_features
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
Q_TAIL = [0.02] + QUANTILES + [0.98]


def run_variant(df, feature_cols, use_recency, use_tails, use_pc, label):
    t0 = time.time()
    quantiles = Q_TAIL if use_tails else QUANTILES
    qlevels = np.array(quantiles)
    fold_scores = {}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx].copy(), df[va_idx].copy()

        cols = list(feature_cols)
        if use_pc:
            # fold-train만으로 그룹별 isotonic 파워커브 적합 (누수 차단)
            for tgt in TARGET_COLS:
                cap = CAPACITY_KWH[tgt]
                m = tr[tgt].notna() & tr.kma_ws875.notna()
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(tr.loc[m, "kma_ws875"], tr.loc[m, tgt] / cap)
                pc_col = f"pc_{tgt}"
                for frame in (tr, va):
                    vals = np.full(len(frame), np.nan)
                    ok = frame.kma_ws875.notna().to_numpy()
                    vals[ok] = iso.predict(frame.kma_ws875.to_numpy()[ok])
                    frame[pc_col] = vals
            # 공유 학습이므로 그룹별 pc를 하나의 컬럼으로: stack 시 그룹에 맞는 값 사용
            # → group_X에서 처리 불가하므로 여기선 그룹별 컬럼 3개 모두 피처로 추가
            cols = cols + [f"pc_{t}" for t in TARGET_COLS]

        w_all = df_weights[tr_idx.to_numpy()].copy()
        if use_recency:
            age = (va_start - tr.forecast_kst_dtm).dt.days.to_numpy()
            decay = 0.5 ** (age / 540.0)
            for tgt in TARGET_COLS:
                w_all[tgt] = w_all[tgt].to_numpy() * decay

        X, y, w = stack_groups(tr, cols, w_all)
        shared_cols = cols + ["g_rated", "g_rotor", "g_id"]
        models = []
        for q in quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
            m.fit(X[shared_cols], y, sample_weight=w)
            models.append(m)

        fs = []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            sub = va.loc[vam].sort_values("forecast_kst_dtm")
            Xv = group_X(sub, cols, tgt)
            qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in models])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            atoms = interp_atoms(qp, levels=qlevels)
            pred = optimize_submission(atoms, cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            fs.append(s)
        fold_scores[f"{va_start:%Y-%m}"] = np.nanmean(fs)
    mean = np.mean(list(fold_scores.values()))
    print(f"{label}: mean={mean:.4f} | " + " ".join(f"{k}:{v:.4f}" for k, v in fold_scores.items())
          + f" ({time.time()-t0:.0f}s)")
    return mean


def main() -> None:
    global df_weights
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    kma = build_kma_features(PROJECT)
    df_weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(kma, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    kma_cols = [c for c in kma.columns if c != "forecast_kst_dtm"]
    print(f"KMA coverage: {df.kma_ws875.notna().mean()*100:.1f}%")

    if "--skip-a" not in sys.argv:
        run_variant(df, base_cols + kma_cols, False, False, False, "v8a KMA피처")
    run_variant(df, base_cols + kma_cols, False, False, True, "v8b +파워커브")
    run_variant(df, base_cols + kma_cols, True, True, True, "v8c +최근성+꼬리")


if __name__ == "__main__":
    main()
