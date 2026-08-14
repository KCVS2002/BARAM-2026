"""패키지 재현 차이 원인 판별 2 — AnEn·OM sister·행순서 캐시 대조."""

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

from lib import (ANEN_FEATS, CAPACITY_KWH, FEAT_W, GROUP_META, QUANTILES, TARGET_COLS,
                 add_consensus, load_ifs2_features, load_ifs_features, load_om)
from src.features import build_features
from src.decision import smooth_quantiles_by_day
from sklearn.neighbors import NearestNeighbors

DATA = PROJECT / "Data"
CACHE = PROJECT / "experiments" / "test_cache"

# 행순서: 원본 sub_055 vs 재현본
ref = pd.read_csv(PROJECT / "submissions" / "sub_055_tabpfn.csv", encoding="utf-8-sig")
rep = pd.read_csv(PKG / "output" / "sub_055_reproduced.csv", encoding="utf-8-sig")
print("행순서 일치:", (ref["forecast_kst_dtm"].values == rep["forecast_kst_dtm"].values).all())
m = ref.merge(rep, on="forecast_kst_dtm", suffixes=("_o", "_r"))
for tgt in TARGET_COLS:
    d = np.abs(m[f"{tgt}_o"] - m[f"{tgt}_r"])
    print(f"  (dtm 조인) {tgt}: |Δ| mean {d.mean():.1f} / max {d.max():.1f}")

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
om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
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
sis_cols = feature_cols + om_cols
shared_sis = sis_cols + ["g_rated", "g_rotor", "g_id"]
sisters = [joblib.load(PKG / "models" / f"lgb_sister_q{int(q*100):02d}.joblib") for q in QUANTILES]

for tgt_i, tgt in enumerate(TARGET_COLS):
    cap = CAPACITY_KWH[tgt]
    z = np.load(CACHE / f"test_atoms_{tgt}.npz")
    # AnEn 대조
    trm = df[tgt].notna()
    tr_ok = df.loc[trm].dropna(subset=ANEN_FEATS)
    mu_a = tr_ok[ANEN_FEATS].mean()
    sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
    nn = NearestNeighbors(n_neighbors=200).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
    dist, idx = nn.kneighbors(((te[ANEN_FEATS].fillna(mu_a) - mu_a) / sd_a * FEAT_W).to_numpy())
    anen = np.sort((tr_ok[tgt] / cap).to_numpy()[idx] * cap, axis=1)
    da = np.abs(anen - z["anen"])
    dd = np.abs(dist - z["dist"])
    line = f"{tgt}: anen |Δ| mean {da.mean():.2f} max {da.max():.1f} | dist |Δ| max {dd.max():.4f}"
    if tgt == "kpx_group_3":
        Xvs = te[sis_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xvs["g_rated"], Xvs["g_rotor"] = rated, rotor
        Xvs["g_id"] = tgt_i
        sq = np.column_stack([np.clip(mm.predict(Xvs[shared_sis]) * cap, 0, cap) for mm in sisters])
        sq.sort(axis=1)
        sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
        ds = np.abs(sq - z["sisq"])
        line += f" | sisq |Δ| mean {ds.mean():.1f} max {ds.max():.1f}"
    print(line)
