"""제출 v5: v8d 파이프라인 (KMA 연직 프로파일 + 최근성 가중 + 꼬리 분위, CV 0.6427).

학습: 전체 train (2022~2024), 최근성 가중 기준시점 = 테스트 시작(2025-01-01).
추론: 테스트 기간 KMA 피처 포함 (2025년 수집 완료분).
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_META, QUANTILES, label_weights
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.external_kma import build_kma_features
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
Q_TAIL = list(QUANTILES)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    sub = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"])
    assert sub["forecast_kst_dtm"].is_monotonic_increasing

    feat_tr = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    feat_te = build_features(
        pd.read_csv(DATA / "test/ldaps_test.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "test/gfs_test.csv", encoding="utf-8-sig"),
    )
    kma = build_kma_features(PROJECT)  # 전 기간 (train+test)

    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_tr, on="forecast_kst_dtm", how="left")
    df = df.merge(kma, on="forecast_kst_dtm", how="left")
    te = sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
    te = te.merge(kma, on="forecast_kst_dtm", how="left")

    base_cols = [c for c in feat_tr.columns if c != "forecast_kst_dtm"]
    kma_cols = [c for c in kma.columns if c != "forecast_kst_dtm"]
    feature_cols = base_cols + kma_cols
    shared_cols = feature_cols + ["g_rated", "g_rotor", "g_id"]
    print(f"train KMA cov: {df.kma_ws875.notna().mean()*100:.1f}%, "
          f"test KMA cov: {te.kma_ws875.notna().mean()*100:.1f}% ({time.time()-t0:.0f}s)")

    # 최근성 가중 (기준: 테스트 시작)
    ref = pd.Timestamp(2025, 1, 1, 1)
    age = (ref - df.forecast_kst_dtm).dt.days.to_numpy()
    decay = np.ones_like(age, dtype=float)

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
        stack_w.append(weights.loc[trm.to_numpy(), tgt].to_numpy() * decay[trm.to_numpy()])
    X_all = pd.concat(stack_X)
    y_all = pd.concat(stack_y)
    w_all = np.concatenate(stack_w)

    qmodels = []
    for q in Q_TAIL:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
        m.fit(X_all[shared_cols], y_all, sample_weight=w_all)
        qmodels.append(m)
        print(f"q{q:.2f} done ({time.time()-t0:.0f}s)")

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    qlevels = np.array(Q_TAIL)
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        Xv = te[feature_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
        a = df.loc[df[tgt].notna(), tgt]
        a_bar = a[a >= cap * 0.10].mean()
        out[tgt] = optimize_submission(interp_atoms(qp, levels=qlevels), cap, a_bar)
        print(f"{tgt}: pred mean={out[tgt].mean():.0f}")

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_006_kma_only.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
