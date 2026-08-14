"""제출 v11: sub_009 + ECC 터빈 원자 (v12b, CV 0.6484).

GBM 퀀타일(제공 피처) + 시간아날로그(재정렬) + ECC 터빈 합산 원자의 3소스 결합.
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
from scripts.train_v12_turbine import RATED, ROTOR, TURBINES, load_turbine_hourly
from src.decision import QLEVELS, interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
K, N_GBM = 150, 150


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
    tb = load_turbine_hourly()
    feature_cols = [c for c in feat_tr.columns if c != "forecast_kst_dtm"]
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_tr, on="forecast_kst_dtm", how="left")
    df = df.merge(tb, left_on="forecast_kst_dtm", right_index=True, how="left")
    te = sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")

    # --- GBM (그룹 공유) ---
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
    shared_g = feature_cols + ["g_rated", "g_rotor", "g_id"]
    gmodels = []
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
        m.fit(X_all[shared_g], y_all, sample_weight=w_all)
        gmodels.append(m)
    print(f"GBM done ({time.time()-t0:.0f}s)")

    # --- 터빈 스택 모델 ---
    xs, ys = [], []
    for maker, num, grp in TURBINES:
        col = f"{maker}_wtg{num:02d}_power_kw10m"
        m = df[col].notna()
        X = df.loc[m, feature_cols].copy()
        X["t_rated"], X["t_rotor"] = RATED[maker], ROTOR[maker]
        X["t_id"] = TURBINES.index((maker, num, grp))
        xs.append(X)
        ys.append((df.loc[m, col] / RATED[maker]).clip(0, 1.2))
    Xt_all, yt_all = pd.concat(xs), pd.concat(ys)
    shared_t = feature_cols + ["t_rated", "t_rotor", "t_id"]
    tmodels = []
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
        m.fit(Xt_all[shared_t], yt_all)
        tmodels.append(m)
    print(f"turbine models done ({time.time()-t0:.0f}s)")

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        rated, rotor = GROUP_META[tgt]
        grp_tb = [(mk, n) for mk, n, g in TURBINES if g == tgt]
        grp_cols = [f"{mk}_wtg{n:02d}_power_kw10m" for mk, n in grp_tb]

        # GBM 원자
        Xv = te[feature_cols].copy()
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_g]) * cap, 0, cap) for m in gmodels])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
        gbm = interp_atoms(qp, n=N_GBM)
        med = np.median(gbm, axis=1)

        # 아날로그 (라벨 + dependence template)
        trm = df[tgt].notna()
        tr_ok = df.loc[trm].dropna(subset=ANEN_FEATS + grp_cols)
        mu = tr_ok[ANEN_FEATS].mean()
        sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
        nn = NearestNeighbors(n_neighbors=K).fit(((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
        Xte_sim = ((te[ANEN_FEATS].fillna(mu) - mu) / sd * FEAT_W).to_numpy()
        _, idx = nn.kneighbors(Xte_sim)
        ytr_lab = (tr_ok[tgt] / cap).to_numpy()
        anen = np.sort(ytr_lab[idx] * cap, axis=1)
        anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)

        # ECC 터빈 원자
        marg = {}
        for mk, n in grp_tb:
            Xv2 = te[feature_cols].copy()
            Xv2["t_rated"], Xv2["t_rotor"] = RATED[mk], ROTOR[mk]
            Xv2["t_id"] = TURBINES.index((mk, n, tgt))
            qp_t = np.column_stack([np.clip(m.predict(Xv2[shared_t]), 0, 1.2) * RATED[mk] for m in tmodels])
            qp_t.sort(axis=1)
            marg[(mk, n)] = qp_t
        tb_actual = tr_ok[grp_cols].to_numpy()
        neigh_vals = tb_actual[idx]
        ranks = neigh_vals.argsort(axis=1).argsort(axis=1) / (K - 1)
        ecc = np.zeros((len(te), K))
        for ti, (mk, n) in enumerate(grp_tb):
            qp_t = marg[(mk, n)]
            for row in range(len(te)):
                ecc[row] += np.interp(ranks[row, :, ti], QLEVELS, qp_t[row])
        both = df.dropna(subset=[tgt] + grp_cols)
        ratio = both[tgt].sum() / both[grp_cols].sum(axis=1).sum()
        ecc = np.sort(np.clip(ecc * ratio, 0, cap), axis=1)
        ecc = np.clip(ecc + (med - np.median(ecc, axis=1))[:, None], 0, cap)

        atoms = np.sort(np.concatenate([gbm, anen, ecc], axis=1), axis=1)
        a = df.loc[trm, tgt]
        a_bar = a[a >= cap * 0.10].mean()
        out[tgt] = optimize_submission(atoms, cap, a_bar)
        print(f"{tgt}: pred mean={out[tgt].mean():.0f} ({time.time()-t0:.0f}s)")

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_011_ecc.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
