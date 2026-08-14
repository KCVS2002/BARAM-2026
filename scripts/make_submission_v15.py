"""제출 sub_015: sub_013 + g2 분위 블렌드 (공유 0.25 + g2 자기모델 0.75, 실험 #51b).

sub_012(공식 최고 0.64069)와 동일하되, kpx_group_3에만 OM sister 원자 150개를 추가:
- sister = LGBM 19q, 학습 2024년 전체(3그룹 공유 스택), 피처 = 기존 125 + OM 18
  (ECMWF IFS025/ICON/GFS previous_day2·3 — 항상 D-1 13:00 이전 발표, 누수 안전)
- raw 결합(재정렬 없음): 홀드아웃에서 raw(+0.018) > shift(+0.010)
- g1/g2는 홀드아웃에서 악화(-0.01~-0.03)라 미적용
홀드아웃(2024-10~12) ablation: 이득 +0.018 = 최근성 +0.009 + OM 정보 +0.009.
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
from scripts.train_v22_omsister import load_om
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
P_MIX = 0.15


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
    om = load_om()
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    df = (lab.rename(columns={"kst_dtm": "forecast_kst_dtm"})
          .merge(feat_tr, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))
    te = (sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))

    # ── base GBM (sub_009/012와 동일: 전 기간 공유 학습) ──
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
    qmodels = [lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
               .fit(X_all[shared_cols], y_all, sample_weight=w_all) for q in QUANTILES]
    print(f"base GBM done ({time.time()-t0:.0f}s)", flush=True)

    # ── g2 자기모델 (g2 데이터만, 그룹 피처 없음) — v23b w=0.75, CV +0.0062 5/6 fold ──
    g2 = "kpx_group_2"
    cap2 = CAPACITY_KWH[g2]
    trm2 = df[g2].notna()
    g2models = [lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
                .fit(df.loc[trm2, feature_cols], df.loc[trm2, g2] / cap2,
                     sample_weight=weights.loc[trm2.to_numpy(), g2]) for q in QUANTILES]
    print(f"g2 own GBM done ({time.time()-t0:.0f}s)", flush=True)

    # ── OM sister GBM (2024년 전체, 공유 학습, 기존+OM 피처) ──
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
    print(f"OM sister done: {len(sX)}행 학습 ({time.time()-t0:.0f}s)", flush=True)

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        rated, rotor = GROUP_META[tgt]
        Xv = te[feature_cols].copy()
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        if tgt == "kpx_group_2":
            qp_own = np.column_stack([np.clip(m.predict(te[feature_cols]) * cap, 0, cap)
                                      for m in g2models])
            qp = 0.25 * qp + 0.75 * qp_own
            print("g2: 분위 블렌드 w=0.75 적용", flush=True)
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
        gbm_atoms = interp_atoms(qp, n=K)

        trm = df[tgt].notna()
        tr_ok = df.loc[trm].dropna(subset=ANEN_FEATS)
        mu = tr_ok[ANEN_FEATS].mean()
        sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
        nn = NearestNeighbors(n_neighbors=K).fit(((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
        _, idx = nn.kneighbors(((te[ANEN_FEATS].fillna(mu) - mu) / sd * FEAT_W).to_numpy())
        anen_atoms = np.sort((tr_ok[tgt] / cap).to_numpy()[idx] * cap, axis=1)
        shift = np.median(gbm_atoms, axis=1) - np.median(anen_atoms, axis=1)
        anen_atoms = np.clip(anen_atoms + shift[:, None], 0, cap)
        parts = [anen_atoms, gbm_atoms]

        if tgt == "kpx_group_3":
            # (1) 가용률 혼합 (sub_012 채택분)
            n = anen_atoms.shape[1] + gbm_atoms.shape[1]
            n8 = max(int(round(n * P_MIX * 2 / 3)), 1)
            n6 = max(int(round(n * P_MIX * 1 / 3)), 1)
            parts += [gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
                      gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6]
            # (2) OM sister 원자 (raw, 이번 A/B 대상)
            Xvs = te[sis_cols].copy()
            Xvs["g_rated"], Xvs["g_rotor"] = rated, rotor
            Xvs["g_id"] = list(GROUP_META).index(tgt)
            sq = np.column_stack([np.clip(m.predict(Xvs[shared_sis]) * cap, 0, cap) for m in sisters])
            sq.sort(axis=1)
            sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
            parts.append(interp_atoms(sq, n=K))
            print(f"g3: 혼합 +{n8+n6}, OM sister +{K} 원자", flush=True)

        atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
        a = df.loc[trm, tgt]
        a_bar = a[a >= cap * 0.10].mean()
        out[tgt] = optimize_submission(atoms, cap, a_bar)
        print(f"{tgt}: pred mean={out[tgt].mean():.0f} ({time.time()-t0:.0f}s)", flush=True)

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_015_g2_blend.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
