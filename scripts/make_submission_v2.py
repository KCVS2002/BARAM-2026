"""제출 v2: 라벨 정제 + 그룹 공유 퀀타일 모델 (CV 0.6283 파이프라인) 전체 학습 → 제출 파일."""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_META, QUANTILES, label_weights
from src.decision import optimize_submission
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    sub = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"])

    feat_tr = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    feat_te = build_features(
        pd.read_csv(DATA / "test/ldaps_test.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "test/gfs_test.csv", encoding="utf-8-sig"),
    )
    feature_cols = [c for c in feat_tr.columns if c != "forecast_kst_dtm"]
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_tr, on="forecast_kst_dtm", how="left")
    X_te_base = sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")[feature_cols]
    print(f"ready ({time.time()-t0:.0f}s)")

    # 그룹 공유 학습 데이터 구성
    stack_X, stack_y, stack_w = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna()
        Xg = df.loc[trm, feature_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xg["g_rated"], Xg["g_rotor"] = rated, rotor
        Xg["g_id"] = list(GROUP_META).index(tgt)
        stack_X.append(Xg)
        stack_y.append(df.loc[trm, tgt] / cap)
        stack_w.append(weights.loc[trm.to_numpy(), tgt])
    X_all, y_all, w_all = pd.concat(stack_X), pd.concat(stack_y), pd.concat(stack_w)
    shared_cols = feature_cols + ["g_rated", "g_rotor", "g_id"]

    qmodels = []
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
        m.fit(X_all[shared_cols], y_all, sample_weight=w_all)
        qmodels.append(m)
        print(f"q{q:.2f} done ({time.time()-t0:.0f}s)")

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        Xv = X_te_base.copy()
        rated, rotor = GROUP_META[tgt]
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        qp.sort(axis=1)
        a = df.loc[df[tgt].notna(), tgt]
        a_bar = a[a >= cap * 0.10].mean()
        out[tgt] = optimize_submission(qp, cap, a_bar)
        print(f"{tgt}: pred mean={out[tgt].mean():.0f}")

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_002_v2b_clean_shared.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
