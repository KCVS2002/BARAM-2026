"""v12b: 터빈 bottom-up + ECC(앙상블 코퓰라 커플링) 상관 반영 합산.

v12 보류 사유 해소: comonotonic 합산(여름 분포 과대추정) 대신,
아날로그 시간들의 터빈 간 실제 순위 패턴을 dependence template로 사용.

절차 (그룹 g, 검증 시각 t):
 1. 터빈 스택 퀀타일 모델 → 터빈별 마지널 분포 (19q)
 2. t의 아날로그 K개 시간(기존 ANEN 유사도) → 각 아날로그 k에서 터빈 i의 실측 순위 r_ik
 3. 원자_k = Σ_i MarginalQuantile_i(r_ik) → K개의 상관-현실적 그룹 합 원자
 4. 라벨/SCADA 비율 보정 + GBM 중앙값 재정렬 → sub_009 원자에 결합 → FICR 최적화

기준: sub_009 구성 CV 0.6470.
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

from scripts.train_v2_clean_shared import BASE_PARAMS, QUANTILES
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from scripts.train_v12_turbine import RATED, ROTOR, TURBINES, load_turbine_hourly
from src.decision import QLEVELS, interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
K = 150
N_GBM = 150


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    tb = load_turbine_hourly()
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(tb, left_on="forecast_kst_dtm", right_index=True, how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    oof = pd.read_parquet(PROJECT / "experiments" / "oof_quantiles.parquet")
    oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"])
    QC = [f"q{j}" for j in range(19)]
    print(f"ready ({time.time()-t0:.0f}s)")

    res = {}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr = df[df.forecast_kst_dtm < va_start]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        # 터빈 스택 퀀타일 모델
        xs, ys = [], []
        for maker, num, grp in TURBINES:
            col = f"{maker}_wtg{num:02d}_power_kw10m"
            m = tr[col].notna()
            X = tr.loc[m, base_cols].copy()
            X["t_rated"], X["t_rotor"] = RATED[maker], ROTOR[maker]
            X["t_id"] = TURBINES.index((maker, num, grp))
            xs.append(X)
            ys.append((tr.loc[m, col] / RATED[maker]).clip(0, 1.2))
        X_all, y_all = pd.concat(xs), pd.concat(ys)
        shared = base_cols + ["t_rated", "t_rotor", "t_id"]
        tmodels = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
            m.fit(X_all[shared], y_all)
            tmodels.append(m)

        fs = []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            grp_tb = [(mk, n) for mk, n, g in TURBINES if g == tgt]
            grp_cols = [f"{mk}_wtg{n:02d}_power_kw10m" for mk, n in grp_tb]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)

            # 아날로그 이웃 (dependence template 소스)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS + grp_cols)
            mu = tr_ok[ANEN_FEATS].mean()
            sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
            nn = NearestNeighbors(n_neighbors=K).fit(((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
            _, idx = nn.kneighbors(((sub[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
            # 아날로그별 터빈 실측 순위 (0~1)
            tb_actual = tr_ok[grp_cols].to_numpy()  # (n_train, n_turb)

            # 터빈별 마지널 퀀타일 예측
            marg = {}
            for mk, n in grp_tb:
                Xv = sub[base_cols].copy()
                Xv["t_rated"], Xv["t_rotor"] = RATED[mk], ROTOR[mk]
                Xv["t_id"] = TURBINES.index((mk, n, tgt))
                qp_t = np.column_stack([np.clip(m.predict(Xv[shared]), 0, 1.2) * RATED[mk] for m in tmodels])
                qp_t.sort(axis=1)
                marg[(mk, n)] = qp_t  # (n_va, 19)

            # ECC 합산
            n_va = len(sub)
            ecc = np.zeros((n_va, K))
            neigh_vals = tb_actual[idx]  # (n_va, K, n_turb)
            ranks = neigh_vals.argsort(axis=1).argsort(axis=1) / (K - 1)  # (n_va, K, n_turb)
            for ti, (mk, n) in enumerate(grp_tb):
                qp_t = marg[(mk, n)]
                for row in range(n_va):
                    ecc[row] += np.interp(ranks[row, :, ti], QLEVELS, qp_t[row])
            # 스케일 보정 (라벨 vs SCADA 합)
            both = tr.dropna(subset=[tgt] + grp_cols)
            ratio = both[tgt].sum() / both[grp_cols].sum(axis=1).sum()
            ecc = np.sort(np.clip(ecc * ratio, 0, cap), axis=1)

            # GBM + 시간아날로그(라벨) + ECC 결합 (sub_009 + ECC)
            o = oof[(oof.fold == fold) & (oof.target == tgt)].set_index("forecast_kst_dtm")
            o = o.reindex(sub.forecast_kst_dtm)
            qp = np.sort(smooth_quantiles_by_day(o[QC].to_numpy(), sub.forecast_kst_dtm), axis=1)
            gbm = interp_atoms(qp, n=N_GBM)
            med = np.median(gbm, axis=1)

            ytr_lab = (tr_ok[tgt] / cap).to_numpy()
            anen = np.sort(ytr_lab[idx] * cap, axis=1)
            anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)
            ecc = np.clip(ecc + (med - np.median(ecc, axis=1))[:, None], 0, cap)

            atoms = np.sort(np.concatenate([gbm, anen, ecc], axis=1), axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            pred = optimize_submission(atoms, cap, a_bar)
            s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
            fs.append(s)
        res[fold] = np.nanmean(fs)
        print(f"fold {fold}: {res[fold]:.4f} ({time.time()-t0:.0f}s)")

    print(f"\n=== v12b GBM+AnEn+ECC: {np.mean(list(res.values())):.4f} (기준 sub_009 구성 0.6470) ===")


if __name__ == "__main__":
    main()
