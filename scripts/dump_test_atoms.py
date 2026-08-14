"""테스트(2025) 쪽 원자 재료 캐시 — 캐시 우선 규칙의 제출판 (#109 부속).

make_submission_v29_q3.py의 불변 부분(전체학습 LGBM 19분위 + OM sister +
AnEn 이웃)을 1회 계산해 저장. 이후 원자 구성·결정층만 바꾸는 제출은
make_submission_from_cache.py로 재학습 없이 분 단위 생성.

저장: experiments/test_cache/test_atoms_{tgt}.npz
  dtm(int64 ns) / qp(8760,19 정렬·평활 완료) / anen(8760,200) / dist(8760,200)
  / a_bar / cap / (g3만) sisq(8760,19 정렬·평활 완료)
검증: make_submission_from_cache.py가 sub_029를 재현해 원본 CSV와 대조.
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
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from scripts.train_v22_omsister import load_om
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
OUT = PROJECT / "experiments" / "test_cache"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
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
    weights_main = weights.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        weights_main[tgt] = weights_main[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    om = load_om()
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    df = (lab.rename(columns={"kst_dtm": "forecast_kst_dtm"})
          .merge(feat_tr, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))
    te = (sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))
    ifs = load_ifs_features()
    df = df.merge(ifs, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs, on="forecast_kst_dtm", how="left")
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    mu_t, sd_t = df[tri].mean(), df[tri].std()
    for d_ in (df, te):
        z = (d_[tri] - mu_t) / sd_t
        d_["cons3_mean"] = z.mean(axis=1)
        d_["cons3_std"] = z.std(axis=1)
    ifs2 = load_ifs2_features()
    df = df.merge(ifs2, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs2, on="forecast_kst_dtm", how="left")
    for d_ in (df, te):
        d_["ifs_shear"] = d_["ifs_ws700"] - d_["ifs_ws925"]
    ifs_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
                "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    print(f"features ready ({time.time()-t0:.0f}s)", flush=True)

    # ── base GBM (v29와 동일: 전 기간 공유 학습, q3 가중) ──
    stack_X, stack_y, stack_w = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna()
        Xg = df.loc[trm, feature_cols + ifs_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xg["g_rated"], Xg["g_rotor"] = rated, rotor
        Xg["g_id"] = list(GROUP_META).index(tgt)
        stack_X.append(Xg)
        stack_y.append(df.loc[trm, tgt] / cap)
        stack_w.append(weights_main.loc[trm.to_numpy(), tgt])
    X_all, y_all, w_all = pd.concat(stack_X), pd.concat(stack_y), pd.concat(stack_w)
    shared_cols = feature_cols + ifs_cols + ["g_rated", "g_rotor", "g_id"]
    qmodels = []
    for qi, q in enumerate(QUANTILES):
        qmodels.append(lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
                       .fit(X_all[shared_cols], y_all, sample_weight=w_all))
        print(f"base GBM {qi+1}/{len(QUANTILES)} ({time.time()-t0:.0f}s)", flush=True)

    # ── OM sister GBM (2024년 전체, clean 가중) ──
    sis_cols = feature_cols + om_cols
    shared_sis = sis_cols + ["g_rated", "g_rotor", "g_id"]
    in24 = df.forecast_kst_dtm.dt.year == 2024
    sX, sy, sw = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna() & in24
        Xg = df.loc[trm, sis_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xg["g_rated"], Xg["g_rotor"] = rated, rotor
        Xg["g_id"] = list(GROUP_META).index(tgt)
        sX.append(Xg)
        sy.append(df.loc[trm, tgt] / cap)
        sw.append(weights.loc[trm.to_numpy(), tgt])
    sX, sy, sw = pd.concat(sX), pd.concat(sy), pd.concat(sw)
    sisters = [lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
               .fit(sX[shared_sis], sy, sample_weight=sw) for q in QUANTILES]
    print(f"OM sister done ({time.time()-t0:.0f}s)", flush=True)

    dtm = sub["forecast_kst_dtm"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        rated, rotor = GROUP_META[tgt]
        Xv = te[feature_cols + ifs_cols].copy()
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)

        trm = df[tgt].notna()
        tr_ok = df.loc[trm].dropna(subset=ANEN_FEATS)
        mu = tr_ok[ANEN_FEATS].mean()
        sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
        nn = NearestNeighbors(n_neighbors=200).fit(((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
        dist, idx = nn.kneighbors(((te[ANEN_FEATS].fillna(mu) - mu) / sd * FEAT_W).to_numpy())
        anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[idx] * cap, axis=1)

        a = df.loc[trm, tgt]
        a_bar = float(a[a >= cap * 0.10].mean())
        extra = {}
        if tgt == "kpx_group_3":
            Xvs = te[sis_cols].copy()
            Xvs["g_rated"], Xvs["g_rotor"] = rated, rotor
            Xvs["g_id"] = list(GROUP_META).index(tgt)
            sq = np.column_stack([np.clip(m.predict(Xvs[shared_sis]) * cap, 0, cap) for m in sisters])
            sq.sort(axis=1)
            sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
            extra["sisq"] = sq
        np.savez_compressed(OUT / f"test_atoms_{tgt}.npz",
                            dtm=dtm, qp=qp, anen=anen200, dist=dist,
                            a_bar=a_bar, cap=float(cap), **extra)
        print(f"{tgt}: 캐시 저장 완료 ({time.time()-t0:.0f}s)", flush=True)
    print("=== 테스트 원자 캐시 덤프 완료 ===", flush=True)


if __name__ == "__main__":
    main()
