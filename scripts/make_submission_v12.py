"""제출 sub_012: sub_009 + g3 가용률 상태 혼합 원자 (실험 #46 보류 카드의 LB A/B).

sub_009(GBM 퀀타일 + AnEn 재정렬 블렌드, LB 0.64018)와 완전 동일하되,
kpx_group_3에만 GBM 원자의 0.8배(4/5 가동)·0.6배(3/5 가동) 사본을 질량 p=0.15로
추가(2:1 배분) — 부분정지 하방 모드 보강. CV: g3 +0.0022 (g1/g2 대조군은 악화 →
g3 특이성 확인), 이득이 2024 H2 fold 집중이라 LB만이 판정 가능.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_META, QUANTILES, label_weights
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W, K
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
P_MIX = 0.15  # g3 혼합 질량 (v20 CV 최적)


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
    te = sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")

    # GBM 퀀타일 (공유 학습) — sub_009와 동일
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
    print(f"GBM done ({time.time()-t0:.0f}s)", flush=True)

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        rated, rotor = GROUP_META[tgt]
        Xv = te[feature_cols].copy()
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
        gbm_atoms = interp_atoms(qp, n=K)

        trm = df[tgt].notna()
        tr_ok = df.loc[trm].dropna(subset=ANEN_FEATS)
        mu = tr_ok[ANEN_FEATS].mean()
        sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
        Xtr = ((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy()
        ytr = (tr_ok[tgt] / cap).to_numpy()
        nn = NearestNeighbors(n_neighbors=K).fit(Xtr)
        Xte = ((te[ANEN_FEATS].fillna(mu) - mu) / sd * FEAT_W).to_numpy()
        _, idx = nn.kneighbors(Xte)
        anen_atoms = np.sort(ytr[idx] * cap, axis=1)

        shift = np.median(gbm_atoms, axis=1) - np.median(anen_atoms, axis=1)
        anen_atoms = np.clip(anen_atoms + shift[:, None], 0, cap)
        parts = [anen_atoms, gbm_atoms]

        # g3 전용: 가용률 하방 모드 (v20 구성 재현)
        if tgt == "kpx_group_3":
            n = anen_atoms.shape[1] + gbm_atoms.shape[1]
            n8 = max(int(round(n * P_MIX * 2 / 3)), 1)
            n6 = max(int(round(n * P_MIX * 1 / 3)), 1)
            idx8 = np.linspace(0, gbm_atoms.shape[1] - 1, n8).astype(int)
            idx6 = np.linspace(0, gbm_atoms.shape[1] - 1, n6).astype(int)
            parts += [gbm_atoms[:, idx8] * 0.8, gbm_atoms[:, idx6] * 0.6]
            print(f"g3 혼합: +{n8}원자(x0.8) +{n6}원자(x0.6)", flush=True)

        atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
        a = df.loc[trm, tgt]
        a_bar = a[a >= cap * 0.10].mean()
        out[tgt] = optimize_submission(atoms, cap, a_bar)
        print(f"{tgt}: pred mean={out[tgt].mean():.0f} ({time.time()-t0:.0f}s)", flush=True)

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_012_g3mix.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
