"""패키지 재현 차이 원인 판별 — 패키지 GBM qp vs 대회 당시 test_atoms 캐시 직접 대조."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
PKG = PROJECT / "submission_package"
sys.path.insert(0, str(PKG))
import os
os.environ["DATA_DIR"] = str(PROJECT / "Data")

from lib import CAPACITY_KWH, GROUP_META, QUANTILES, TARGET_COLS, add_consensus, load_ifs2_features, load_ifs_features, load_om
from src.features import build_features
from src.decision import smooth_quantiles_by_day

DATA = PROJECT / "Data"
CACHE = PROJECT / "experiments" / "test_cache"

lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
sub = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"])
feat_tr = build_features(
    pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
    pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"))
feat_te = build_features(
    pd.read_csv(DATA / "test/ldaps_test.csv", encoding="utf-8-sig"),
    pd.read_csv(DATA / "test/gfs_test.csv", encoding="utf-8-sig"))
feature_cols = [c for c in feat_tr.columns if c != "forecast_kst_dtm"]
om = load_om()
df = (lab.rename(columns={"kst_dtm": "forecast_kst_dtm"})
      .merge(feat_tr, on="forecast_kst_dtm", how="left").merge(om, on="forecast_kst_dtm", how="left"))
te = (sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
      .merge(om, on="forecast_kst_dtm", how="left"))
ifs = load_ifs_features()
df = df.merge(ifs, on="forecast_kst_dtm", how="left")
te = te.merge(ifs, on="forecast_kst_dtm", how="left")
ifs2 = load_ifs2_features()
df = df.merge(ifs2, on="forecast_kst_dtm", how="left")
te = te.merge(ifs2, on="forecast_kst_dtm", how="left")
add_consensus(df, te)
ifs_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
            "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
cols = feature_cols + ifs_cols
shared = cols + ["g_rated", "g_rotor", "g_id"]
qmodels = [joblib.load(PKG / "models" / f"lgb_main_q{int(q*100):02d}.joblib") for q in QUANTILES]

for tgt_i, tgt in enumerate(TARGET_COLS):
    cap = CAPACITY_KWH[tgt]
    Xv = te[cols].copy()
    rated, rotor = GROUP_META[tgt]
    Xv["g_rated"], Xv["g_rotor"] = rated, rotor
    Xv["g_id"] = tgt_i
    qp = np.column_stack([np.clip(m.predict(Xv[shared]) * cap, 0, cap) for m in qmodels])
    qp.sort(axis=1)
    qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
    z = np.load(CACHE / f"test_atoms_{tgt}.npz")
    d = np.abs(qp - z["qp"])
    dd = np.abs(z["dtm"] - sub["forecast_kst_dtm"].values.astype("datetime64[ns]").astype(np.int64)).max()
    print(f"{tgt}: qp |Δ| mean {d.mean():.1f} / p99 {np.quantile(d,0.99):.1f} / max {d.max():.1f} kWh "
          f"| q50 상관 {np.corrcoef(qp[:,9], z['qp'][:,9])[0,1]:.5f} | dtm 정렬 {dd}")
