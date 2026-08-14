"""v26: 분위 노트 조밀화 (19개 → 37개, 0.025 간격) — FICR 공략 2탄.

진단 (#54 ①): 원자 분포의 밴드 확률이 실현 대비 ~4%p 과소 — 19개 노트의 선형보간이
조건부 밀도의 국소 디테일(밴드 스케일 ±6~8%C)을 뭉갠 것이 원인 후보.
처방: 실제 분위 모델을 37개(0.05~0.95, 0.025 간격)로 늘려 밀도 해상도를 2배로.
(#47의 '보간 원자 수 증가'와 다름 — 그것은 같은 19노트의 재보간이라 무효였음.
 이것은 새 분위 학습 = 실제 분포 정보 증가.)

변형: base(19노트) vs dense(37노트) — 둘 다 동일 파이프라인(평활→보간150→AnEn→FICR).
기준: sub_009 구성 CV 0.6470. FICR 분해도 출력.
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

from scripts.train_v2_clean_shared import BASE_PARAMS, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
Q19 = np.round(np.arange(0.05, 0.951, 0.05), 3).tolist()
Q37 = np.round(np.arange(0.05, 0.951, 0.025), 3).tolist()


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in ("base", "dense")}
    fic = {v: {} for v in ("base", "dense")}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]
        X, y, w = stack_groups(tr, base_cols, w_tr)
        shared = base_cols + ["g_rated", "g_rotor", "g_id"]
        models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in Q37}  # 37개 학습 (19는 부분집합)
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in res}
        fi_ = {v: [] for v in res}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            preds = {q: np.clip(models[q].predict(Xv[shared]) * cap, 0, cap) for q in Q37}
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            for v, levels in (("base", Q19), ("dense", Q37)):
                qp = np.column_stack([preds[q] for q in levels])
                qp.sort(axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm = interp_atoms(qp, n=150, levels=np.array(levels))
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, fi_v, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
                fi_[v].append(fi_v)
        for v in res:
            res[v][fold] = np.nanmean(fs[v])
            fic[v][fold] = np.nanmean(fi_[v])
        print(f"fold {fold}: base={res['base'][fold]:.4f} dense={res['dense'][fold]:.4f} "
              f"(FICR {fic['base'][fold]:.3f}→{fic['dense'][fold]:.3f}) ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v26 분위 조밀화 ===")
    for v in res:
        print(f"{v}: 총 {np.mean(list(res[v].values())):.4f} | FICR {np.mean(list(fic[v].values())):.4f}")


if __name__ == "__main__":
    main()
