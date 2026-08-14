"""v8d: KMA 피처 + 최근성 가중 + 꼬리 분위 (파워커브 제외) — v8c에서 역효과 성분 제거."""

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import scripts.train_v8_kma as v8
from scripts.train_v2_clean_shared import label_weights
from src.external_kma import build_kma_features
from src.features import build_features

lab = pd.read_csv(PROJECT / "Data/train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
feat = build_features(
    pd.read_csv(PROJECT / "Data/train/ldaps_train.csv", encoding="utf-8-sig"),
    pd.read_csv(PROJECT / "Data/train/gfs_train.csv", encoding="utf-8-sig"),
)
kma = build_kma_features(PROJECT)
v8.df_weights = label_weights(lab)
df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
df = df.merge(kma, on="forecast_kst_dtm", how="left")
base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
kma_cols = [c for c in kma.columns if c != "forecast_kst_dtm"]
v8.run_variant(df, base_cols + kma_cols, True, True, False, "v8d KMA+최근성+꼬리(no pc)")
