"""v11: Analog Ensemble (AnEn) 프로토타입.

각 검증 시각의 NWP 피처와 가장 유사한 과거 K개 시각을 찾아,
그때의 실제 발전량(cf)들을 예측분포의 원자로 사용 → FICR 최적화.

변형:
  anen      : AnEn 단독 (K=150)
  blend     : AnEn 원자 + GBM 보간 원자 50:50 결합

기준: v2b + smooth/interp = 0.6346.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
K = 150
# 유사도 피처 (풍속·풍향·계절·시각 — 물리 핵심만)
ANEN_FEATS = ["ldaps_ws50max", "gfs_ws100", "gfs_ws80", "gfs_ws850",
              "ldaps_ws10", "gfs_surface_0_gust",
              "gfs_ws100_dsin", "gfs_ws100_dcos",
              "hour_sin", "hour_cos", "month_sin", "month_cos"]
# 풍속류에 높은 가중 (표준화 후 배율)
FEAT_W = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    oof = pd.read_parquet(PROJECT / "experiments" / "oof_quantiles.parquet")
    oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"])
    QC = [f"q{j}" for j in range(19)]

    res = {"anen": {}, "blend": {}}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr = df[df.forecast_kst_dtm < va_start]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        for variant in ("anen", "blend"):
            fs = []
            for tgt in TARGET_COLS:
                cap = CAPACITY_KWH[tgt]
                trm = tr[tgt].notna()
                tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
                mu = tr_ok[ANEN_FEATS].mean()
                sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
                Xtr = ((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy()
                ytr = (tr_ok[tgt] / cap).to_numpy()
                nn = NearestNeighbors(n_neighbors=K).fit(Xtr)

                sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm")
                sub_ok = sub.dropna(subset=ANEN_FEATS)
                Xva = ((sub_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy()
                _, idx = nn.kneighbors(Xva)
                atoms = np.sort(ytr[idx] * cap, axis=1)  # (n, K)

                if variant == "blend":
                    o = oof[(oof.fold == fold) & (oof.target == tgt)].set_index("forecast_kst_dtm")
                    o = o.reindex(sub_ok.forecast_kst_dtm)
                    qp = np.sort(smooth_quantiles_by_day(o[QC].to_numpy(), sub_ok.forecast_kst_dtm), axis=1)
                    gbm_atoms = interp_atoms(qp, n=K)
                    atoms = np.sort(np.concatenate([atoms, gbm_atoms], axis=1), axis=1)

                a_tr = tr.loc[trm, tgt]
                a_bar = a_tr[a_tr >= cap * 0.10].mean()
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(sub_ok[tgt].to_numpy(), pred, cap)
                fs.append(s)
            res[variant][fold] = np.nanmean(fs)
        print(f"fold {fold}: anen={res['anen'][fold]:.4f} blend={res['blend'][fold]:.4f} "
              f"({time.time()-t0:.0f}s)")

    print("\n=== 요약 (기준 0.6346) ===")
    for k, d in res.items():
        print(f"{k}: {np.mean(list(d.values())):.4f}")


if __name__ == "__main__":
    main()
