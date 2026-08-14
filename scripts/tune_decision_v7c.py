"""v7c: 다변수 |오차| 예측모델 기반 퀀타일 폭 조절 (v7b 보류 건의 재설계).

- 오차모델: fold 외 OOF 잔차로 학습(leave-one-fold-out), |actual - q50| 예측 (L1)
- 입력: 퀀타일 폭(q90-q10), 중앙값/용량, 기존 핵심 기상 피처, NOAA 불확실성 변수, 시간대
- 폭 조절: factor = pred_err / 기대오차(학습분 평균), clip [0.6, 1.8]
- 평가: fold별 일관성 중심 (v7b는 1/6 fold 의존으로 보류됐음)
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.external_noaa import build_noaa_features
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

QCOLS = [f"q{j}" for j in range(19)]


def main() -> None:
    oof = pd.read_parquet(PROJECT / "experiments" / "oof_quantiles.parquet")
    oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"])

    feat = build_features(
        pd.read_csv(PROJECT / "Data/train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(PROJECT / "Data/train/gfs_train.csv", encoding="utf-8-sig"),
    )
    noaa = build_noaa_features(PROJECT)
    aux_cols = ["ldaps_ws50max", "gfs_ws100", "gfs_surface_0_gust", "ws_ldaps_gfs_diff"]
    aux = feat[["forecast_kst_dtm"] + aux_cols].merge(noaa, on="forecast_kst_dtm", how="left")

    df = oof.merge(aux, on="forecast_kst_dtm", how="left")
    df["cap"] = df.target.map(CAPACITY_KWH)
    df["q50"] = df["q9"]
    df["qwidth"] = (df["q16"] - df["q2"]) / df["cap"]      # q85-q15
    df["qwidth_wide"] = (df["q18"] - df["q0"]) / df["cap"]  # q95-q05
    df["med_cf"] = df["q50"] / df["cap"]
    dt = df["forecast_kst_dtm"]
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["g_id"] = df.target.map({t: i for i, t in enumerate(TARGET_COLS)})
    df["abs_err_cf"] = (df.actual - df.q50).abs() / df["cap"]

    err_feats = (["qwidth", "qwidth_wide", "med_cf", "hour_sin", "hour_cos",
                  "month_sin", "month_cos", "g_id"] + aux_cols
                 + [c for c in noaa.columns if c != "forecast_kst_dtm"])

    results = {}
    for mode in ["base", "errmodel"]:
        fold_scores = {}
        for fold in sorted(df.fold.unique()):
            if mode == "errmodel":
                tr = df[df.fold != fold]
                m = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.05,
                                      num_leaves=31, min_child_samples=60,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                      random_state=42, verbose=-1)
                m.fit(tr[err_feats], tr["abs_err_cf"])
            fs = []
            for tgt in TARGET_COLS:
                sub = df[(df.fold == fold) & (df.target == tgt)].sort_values("forecast_kst_dtm")
                cap = CAPACITY_KWH[tgt]
                a_bar = sub.a_bar_train.iloc[0]
                qp = np.sort(smooth_quantiles_by_day(sub[QCOLS].to_numpy(), sub["forecast_kst_dtm"]), axis=1)
                atoms = interp_atoms(qp)
                if mode == "errmodel":
                    pred_err = m.predict(sub[err_feats])
                    base_err = m.predict(df[df.fold != fold][err_feats]).mean()
                    factor = np.clip(pred_err / max(base_err, 1e-6), 0.6, 1.8)
                    med = atoms[:, atoms.shape[1] // 2][:, None]
                    atoms = np.clip(np.sort(med + (atoms - med) * factor[:, None], axis=1), 0, cap)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(sub.actual.to_numpy(), pred, cap)
                fs.append(s)
            fold_scores[fold] = np.nanmean(fs)
        results[mode] = fold_scores
        print(f"{mode}: mean={np.mean(list(fold_scores.values())):.4f} | "
              + " ".join(f"{k}:{v:.4f}" for k, v in fold_scores.items()))

    diff = {k: results["errmodel"][k] - results["base"][k] for k in results["base"]}
    n_up = sum(v > 0 for v in diff.values())
    print(f"\nfold별 개선: {n_up}/6 | " + " ".join(f"{k}:{v:+.4f}" for k, v in diff.items()))


if __name__ == "__main__":
    main()
