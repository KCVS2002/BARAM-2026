"""v27: 풍속 공간 분위 회귀 + 파워커브 사상 원자 (FICR 구조 공략 — 새 아키텍처).

가설: 발전량 직접 회귀는 파워커브의 구간 구조(정격/컷인=평평, 큐빅=증폭)를 뭉갠다.
풍속 공간에서 분위를 예측하고 실측 커브로 사상하면, 커브가 평평한 구간(고출력·A-가중
지배 = FICR 지배 시간대)에서 조건부 분포가 물리적으로 날카로워진다.

- 풍속 타깃: SCADA 팜 평균 풍속 (그룹별, 학습에만 사용 — 추론은 NWP 피처만)
- 커브: 학습기간 (팜풍속, 발전량) 0.25m/s bin 중앙값 → 단조화 보간
- 사상 원자 150개를 GBM 중앙값으로 재정렬(shift) 후 결합 (AnEn과 동일 방식) + raw 변형
- 기준: sub_009 구성(gbm+anen) CV 0.6470. FICR 분해 출력.
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

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_TURBINES, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def farm_wind() -> pd.DataFrame:
    """그룹별 SCADA 팜 평균 풍속 (시간 종료 기준)."""
    out = {}
    cache = {}
    for tgt, (fname, turbines) in GROUP_TURBINES.items():
        if fname not in cache:
            sc = pd.read_csv(DATA / "train" / fname, encoding="utf-8-sig", parse_dates=["kst_dtm"])
            sc["hour_end"] = sc["kst_dtm"].dt.ceil("h")
            cache[fname] = sc
        sc = cache[fname]
        ws_cols = [f"{t}_ws" for t in turbines]
        ws = sc[ws_cols].where((sc[ws_cols] >= 0) & (sc[ws_cols] < 60))
        out[tgt] = ws.mean(axis=1).groupby(sc["hour_end"]).mean()
    return pd.DataFrame(out)


def fit_curve(ws: np.ndarray, power: np.ndarray, cap: float):
    """0.25m/s bin 중앙값 커브 → 단조화 → 보간 함수 반환."""
    m = np.isfinite(ws) & np.isfinite(power)
    b = np.round(ws[m] / 0.25).astype(int)
    dfc = pd.DataFrame({"b": b, "p": power[m]}).groupby("b")["p"].median()
    dfc = dfc[dfc.index >= 0]
    xs = dfc.index.to_numpy() * 0.25
    ys = np.maximum.accumulate(dfc.to_numpy())  # 단조 비감소
    ys = np.clip(ys, 0, cap)
    return lambda w: np.interp(w, xs, ys)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    fw = farm_wind()
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    for tgt in TARGET_COLS:
        df[f"ws_{tgt}"] = fw[tgt].reindex(df.forecast_kst_dtm.values).to_numpy()
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    cov = df[[f"ws_{t}" for t in TARGET_COLS]].notna().mean().mean()
    print(f"ready: 팜풍속 커버리지 {cov*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    variants = ("base", "wind_raw", "wind_shift")
    res = {v: {} for v in variants}
    fic = {v: {} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        # 발전량 GBM (공유, 기존)
        X, y, w = stack_groups(tr, base_cols, w_tr)
        shared = base_cols + ["g_rated", "g_rotor", "g_id"]
        pmodels = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        # 풍속 GBM (공유: 3그룹 팜풍속 스택)
        sw_X, sw_y = [], []
        for tgt in TARGET_COLS:
            wm = tr[f"ws_{tgt}"].notna()
            Xg = tr.loc[wm, base_cols].copy()
            rated_rotor = {"kpx_group_1": (3600, 126), "kpx_group_2": (3600, 126),
                           "kpx_group_3": (4200, 136)}[tgt]
            Xg["g_rated"], Xg["g_rotor"] = rated_rotor
            Xg["g_id"] = TARGET_COLS.index(tgt) if isinstance(TARGET_COLS, list) else list(TARGET_COLS).index(tgt)
            sw_X.append(Xg)
            sw_y.append(tr.loc[wm, f"ws_{tgt}"])
        sw_X, sw_y = pd.concat(sw_X), pd.concat(sw_y)
        wmodels = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            sw_X[shared], sw_y) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        fi_ = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            qp = np.column_stack([np.clip(pmodels[q].predict(Xv[shared]) * cap, 0, cap)
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

            # 풍속 분위 → 커브 사상 원자
            wq = np.column_stack([wmodels[q].predict(Xv[shared]) for q in QUANTILES_FULL])
            wq.sort(axis=1)
            wq = np.sort(smooth_quantiles_by_day(wq, sub["forecast_kst_dtm"]), axis=1)
            w_atoms = interp_atoms(wq, n=150)
            wsm = tr[f"ws_{tgt}"].notna() & trm
            curve = fit_curve(tr.loc[wsm, f"ws_{tgt}"].to_numpy(), tr.loc[wsm, tgt].to_numpy(), cap)
            wind_raw = np.clip(curve(w_atoms), 0, cap)
            wind_raw.sort(axis=1)
            wind_shift = np.clip(wind_raw + (med - np.median(wind_raw, axis=1))[:, None], 0, cap)

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            for v, extra in (("base", None), ("wind_raw", wind_raw), ("wind_shift", wind_shift)):
                parts = [gbm, anen] + ([extra] if extra is not None else [])
                atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, fi_v, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
                fi_[v].append(fi_v)
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            fic[v][fold] = np.nanmean(fi_[v])
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" | FICR " + " ".join(f"{fic[v][fold]:.3f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v27 풍속 공간 + 파워커브 사상 ===")
    for v in variants:
        print(f"{v}: 총 {np.mean(list(res[v].values())):.4f} | FICR {np.mean(list(fic[v].values())):.4f}")


if __name__ == "__main__":
    main()
