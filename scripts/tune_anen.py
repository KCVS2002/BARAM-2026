"""AnEn 재정렬 블렌드 정교화 스윕 (OOF 기반, 재학습 없음).

새 기준: 재정렬 블렌드 K=150, 50:50 = CV 0.6470 / LB 0.64018.
변형: K 민감도, GBM:AnEn 비율, 계절 윈도우(연중일 ±60일 이내 아날로그만).
판정: 5/6 fold 이상 + 명확한 마진일 때만 후보로.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QC = [f"q{j}" for j in range(19)]


def main() -> None:
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    oof = pd.read_parquet(PROJECT / "experiments" / "oof_quantiles.parquet")
    oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"])

    configs = [
        ("anen3to1", dict(K=450, n_gbm=150, season=None)),
        ("anen4to1", dict(K=600, n_gbm=150, season=None)),
    ]

    for name, cfg in configs:
        fold_scores = {}
        for m0 in range(1, 12, 2):
            va_start = pd.Timestamp(2024, m0, 1, 1)
            va_end = va_start + pd.DateOffset(months=2)
            fold = f"{va_start:%Y-%m}"
            tr = df[df.forecast_kst_dtm < va_start]
            va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
            fs = []
            for tgt in TARGET_COLS:
                cap = CAPACITY_KWH[tgt]
                trm = tr[tgt].notna()
                tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
                if cfg["season"]:
                    mid_doy = (va_start + pd.Timedelta(days=30)).dayofyear
                    doy = tr_ok.forecast_kst_dtm.dt.dayofyear
                    dist = np.minimum((doy - mid_doy) % 366, (mid_doy - doy) % 366)
                    tr_ok = tr_ok[dist <= cfg["season"]]
                mu = tr_ok[ANEN_FEATS].mean()
                sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
                nn = NearestNeighbors(n_neighbors=min(cfg["K"], len(tr_ok))).fit(
                    ((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
                ytr = (tr_ok[tgt] / cap).to_numpy()
                sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
                _, idx = nn.kneighbors(((sub[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
                anen = np.sort(ytr[idx] * cap, axis=1)
                o = oof[(oof.fold == fold) & (oof.target == tgt)].set_index("forecast_kst_dtm")
                o = o.reindex(sub.forecast_kst_dtm)
                qp = np.sort(smooth_quantiles_by_day(o[QC].to_numpy(), sub.forecast_kst_dtm), axis=1)
                gbm = interp_atoms(qp, n=cfg["n_gbm"])
                shift = np.median(gbm, axis=1) - np.median(anen, axis=1)
                anen = np.clip(anen + shift[:, None], 0, cap)
                atoms = np.sort(np.concatenate([anen, gbm], axis=1), axis=1)
                a_tr = tr.loc[trm, tgt]
                a_bar = a_tr[a_tr >= cap * 0.10].mean()
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(sub[tgt].to_numpy(), pred, cap)
                fs.append(s)
            fold_scores[fold] = np.nanmean(fs)
        vals = list(fold_scores.values())
        print(f"{name}: mean={np.mean(vals):.4f} | " + " ".join(f"{k}:{v:.4f}" for k, v in fold_scores.items()))


if __name__ == "__main__":
    main()
