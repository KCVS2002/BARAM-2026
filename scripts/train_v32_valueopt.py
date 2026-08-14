"""v32: 가치지향 포인트 모델 (decision-focused learning) — FICR 산식을 직접 학습.

- 손실: 대회 시간별 기여의 부드러운 근사
    L(g;a) = 0.5·|g−a| − 0.5·(a/(4ā))·[3·σ((0.08−|e|)/τ) + σ((0.06−|e|)/τ)],  e=g−a (cf 단위)
  τ=0.015. 비유효시간(a<0.1)은 가중 0.05 (점수 기여 0이므로 안정화용).
- 비볼록 → 헤시안 1 고정(경사 스텝), lr 0.03 × 1200트리.
- init_score = 공유 q50 모델 예측 (합리적 출발점에서 가치 방향으로 정제).
- 평가: base(현행 파이프라인) / value 단독 / blend(최적화기 출력과 50:50).
  총점·NMAE·FICR 분해와 fold 상세를 출력 — 판정은 사용자.
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

from scripts.train_v2_clean_shared import BASE_PARAMS, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TAU = 0.04  # 0.015는 FICR 기울기가 NMAE의 ~14배로 폭주(fold1 붕괴) → 완화
VALUE_PARAMS = dict(
    n_estimators=1200, learning_rate=0.03, num_leaves=63, min_child_samples=40,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=42, verbose=-1,
)


def make_objective(a: np.ndarray, w_eval: np.ndarray, a_bar: float):
    """부드러운 대회 산식 손실의 (grad, hess). g,a는 cf 단위."""
    coef = a / (4.0 * a_bar)

    def obj(y_true, y_pred):
        e = y_pred - a
        ae = np.abs(e)
        # NMAE 항: 0.5·|e| (tanh 근사 기울기)
        g_abs = 0.5 * np.tanh(e / 0.01)
        # FICR 항: −0.5·coef·R(|e|), R = 3σ(z8)+σ(z6)
        s8 = 1.0 / (1.0 + np.exp(-(0.08 - ae) / TAU))
        s6 = 1.0 / (1.0 + np.exp(-(0.06 - ae) / TAU))
        dR_dae = -(3.0 * s8 * (1 - s8) + s6 * (1 - s6)) / TAU
        g_ficr = -0.5 * coef * dR_dae * np.sign(e)
        grad = np.clip(g_abs + g_ficr, -1.0, 1.0) * w_eval  # pinball과 동급 스케일로 제한
        hess = np.ones_like(grad)
        return grad, hess

    return obj


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

    variants = ("base", "value", "blend")
    res = {v: {} for v in variants}
    dec = {v: {} for v in variants}  # (nmae, ficr)
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
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}

        # 가치지향 모델: init = q50 in-sample 예측, ā = 스택 전체 유효시간 평균 cf
        a_arr = y.to_numpy()
        a_bar_cf = a_arr[a_arr >= 0.10].mean()
        w_eval = np.where(a_arr >= 0.10, 1.0, 0.05) * w.to_numpy()
        init_tr = models[0.50].predict(X[shared])
        vmodel = lgb.LGBMRegressor(objective=make_objective(a_arr, w_eval, a_bar_cf),
                                   **VALUE_PARAMS)
        vmodel.fit(X[shared], y, init_score=init_tr)
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        dc = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm = interp_atoms(qp, n=150)
            med = np.median(gbm, axis=1)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)
            atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            g_opt = optimize_submission(atoms, cap, a_bar)
            init_va = models[0.50].predict(Xv[shared])
            g_val = np.clip((init_va + vmodel.predict(Xv[shared])) * cap, 0, cap)
            for v, g in (("base", g_opt), ("value", g_val), ("blend", 0.5 * g_opt + 0.5 * g_val)):
                s, nm, fi, _ = metric_single(actual, g, cap)
                fs[v].append(s)
                dc[v].append((nm, fi))
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([d[0] for d in dc[v]]), np.nanmean([d[1] for d in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + " | FICR " + " ".join(f"{dec[v][fold][1]:.3f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v32 가치지향 포인트 모델 ===")
    for v in variants:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {1-nm:.4f} | FICR {fi:.4f}")


if __name__ == "__main__":
    main()
