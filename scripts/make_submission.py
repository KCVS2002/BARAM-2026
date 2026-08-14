"""전체 학습 데이터로 퀀타일 모델 학습 → FICR 최적화 → 제출 파일 생성."""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import optimize_submission
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
BASE_PARAMS = dict(
    n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=42, verbose=-1,
)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    sub = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"])

    ldaps_tr = pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_tr = pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_te = pd.read_csv(DATA / "test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_te = pd.read_csv(DATA / "test/gfs_test.csv", encoding="utf-8-sig")

    # train/test 피처를 한 번에 생성하면 사이클 내 rolling이 각자 기간 안에서만 계산되도록
    # 기간별로 따로 build (사이클 블록 단위라 결과 동일하지만 명시적으로 분리)
    feat_tr = build_features(ldaps_tr, gfs_tr)
    feat_te = build_features(ldaps_te, gfs_te)
    feature_cols = [c for c in feat_tr.columns if c != "forecast_kst_dtm"]
    print(f"features: train {feat_tr.shape}, test {feat_te.shape} ({time.time()-t0:.0f}s)")

    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_tr, on="forecast_kst_dtm", how="left")
    X_te = sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")[feature_cols]

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna()
        X_tr, y_tr = df.loc[trm, feature_cols], df.loc[trm, tgt] / cap
        a = df.loc[trm, tgt]
        a_bar = a[a >= cap * 0.10].mean()

        qp = np.empty((len(X_te), len(QUANTILES)))
        for j, q in enumerate(QUANTILES):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
            m.fit(X_tr, y_tr)
            qp[:, j] = np.clip(m.predict(X_te) * cap, 0, cap)
        qp.sort(axis=1)
        out[tgt] = optimize_submission(qp, cap, a_bar)
        print(f"{tgt}: done ({time.time()-t0:.0f}s), pred mean={out[tgt].mean():.0f}")

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_001_quantile_ficr.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    # 형식 검증
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == len(sub) and list(chk.columns) == list(sub.columns)
    assert chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name}: {chk.shape}, NaN 없음, forecast_id 일치 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
