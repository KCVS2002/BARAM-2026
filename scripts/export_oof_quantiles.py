"""v2b 파이프라인의 fold별 검증(OOF) 퀀타일 예측을 저장.

의사결정 레이어(FICR 최적화)·캘리브레이션 실험을 모델 재학습 없이
초 단위로 반복하기 위한 캐시. experiments/oof_quantiles.parquet 생성.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import label_weights
from scripts.train_v3_sister import fit_quantiles, group_X, predict_quantiles, stack_groups
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
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    shared_cols = base_cols + ["g_rated", "g_rotor", "g_id"]

    out_rows = []
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights[tr_idx.to_numpy()]

        X, y, w = stack_groups(tr, base_cols, w_tr)
        models = fit_quantiles(X, y, w, shared_cols)

        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            sub = va.loc[vam]
            qp = predict_quantiles(models, group_X(sub, base_cols, tgt), shared_cols, cap)
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            rec = pd.DataFrame({
                "fold": f"{va_start:%Y-%m}",
                "target": tgt,
                "forecast_kst_dtm": sub.forecast_kst_dtm.values,
                "actual": sub[tgt].values,
                "a_bar_train": a_tr[a_tr >= cap * 0.10].mean(),
            })
            for j in range(qp.shape[1]):
                rec[f"q{j}"] = qp[:, j]
            out_rows.append(rec)
        print(f"fold {va_start:%Y-%m} done ({time.time()-t0:.0f}s)")

    out = pd.concat(out_rows, ignore_index=True)
    out.to_parquet(PROJECT / "experiments" / "oof_quantiles.parquet", index=False)
    print(f"saved oof_quantiles.parquet: {out.shape} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
