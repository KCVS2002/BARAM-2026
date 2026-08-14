"""CatBoost 퀀타일 fold별 OOF 캐시 생성 (3원 결합 실험용)."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import label_weights
from scripts.train_v10_ensemble import fit_predict_cat
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]

    rows = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]
        va_subs = {t: va.loc[va[t].notna()].sort_values("forecast_kst_dtm") for t in TARGET_COLS}
        qps = fit_predict_cat(tr, va_subs, base_cols, w_tr)
        for tgt, qp in qps.items():
            rec = pd.DataFrame({
                "fold": f"{va_start:%Y-%m}", "target": tgt,
                "forecast_kst_dtm": va_subs[tgt].forecast_kst_dtm.values,
            })
            for j in range(qp.shape[1]):
                rec[f"q{j}"] = qp[:, j]
            rows.append(rec)
        print(f"fold {va_start:%Y-%m} done ({time.time()-t0:.0f}s)")
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(PROJECT / "experiments" / "oof_quantiles_cat.parquet", index=False)
    print(f"saved: {out.shape} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
