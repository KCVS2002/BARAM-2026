"""튜닝 상위 설정 전체 6fold 재검증 (채택 판정용)."""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import scripts.tune_lgbm as T
from scripts.train_v2_clean_shared import label_weights
from src.features import build_features

CONFIGS = {
    "rand1": dict(num_leaves=31, min_child_samples=20, learning_rate=0.03,
                  n_estimators=500, colsample_bytree=1.0, subsample=1.0, reg_lambda=0.0),
    "rand9": dict(num_leaves=31, min_child_samples=80, learning_rate=0.03,
                  n_estimators=500, colsample_bytree=0.6, subsample=1.0, reg_lambda=5.0),
}


def main() -> None:
    t0 = time.time()
    T.FOLDS = [1, 3, 5, 7, 9, 11]  # 전체 6fold
    lab = pd.read_csv(T.DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(T.DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(T.DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    shared_cols = base_cols + ["g_rated", "g_rotor", "g_id"]

    for name, cfg in CONFIGS.items():
        s = T.evaluate(cfg, df, weights, base_cols, shared_cols)
        print(f"{name}: 6-fold CV = {s:.4f} (기준 v2b+smooth 0.6346) ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
