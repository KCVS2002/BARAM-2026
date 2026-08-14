"""v19: 실측 관측(AWS/ASOS) 기반 피처 — 새 정보원 축.

가설: 피처 추가 5연속 실패는 모두 '같은 정보원(NWP)의 재가공'이었다. 실측 관측은
NWP에 없는 독립 정보(실제 대기 상태, NWP 계통 편차)를 담으므로 별개 가설이다.
특히 rolling NWP 오차(err_7d)는 2025 NWP 버전 변화·계절 편차를 실시간 교정할 수
있는 유일한 통로 (CV로는 이 이득이 안 보일 수 있음 — v6의 역방향 사례 가능성).

누수 규칙: 대상일 D의 모든 시간에 대해 예측기준시점 cutoff = (D-1) 13:00 KST.
모든 관측 피처는 cutoff 이전 관측만 사용 (per-day 상수 → hour 피처와 조합은 GBM 몫).

피처 (지점 100·320·314, 사전 검증 상관 0.59/0.60/0.35):
- obs{stn}_r12  : cutoff 직전 12h 평균 풍속 (D-1 오전 실황)
- obs{stn}_r24  : cutoff 직전 24h 평균 풍속
- obs{stn}_err7 : cutoff 이전 7일간 (obs_ws - ldaps_ws50max) 평균 (rolling NWP 편차)

판정 주의: 관측 lag 피처는 '최근성' 계열 → rolling-origin CV가 이득을 부풀릴 수
있음(v8d 교훈). fold 일관성 + 보수적 판정, LB 확인 전 잠정 채택 금지.

기준: sub_009 구성(gbm+anen) CV 0.6470.
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
OBS_CSV = PROJECT / "external_data" / "kma_aws" / "aws_hourly_merged.csv"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
OBS_STATIONS = [100, 320, 314]


def build_obs_features(target_dtms: pd.Series, nwp_ws: pd.Series) -> pd.DataFrame:
    """target_dtms의 각 시각(집계 종료시각)에 대해 cutoff=(대상일-1) 13:00 이전
    관측만으로 만든 피처를 반환. nwp_ws는 forecast_kst_dtm 인덱스의 ldaps_ws50max."""
    obs = pd.read_csv(OBS_CSV, encoding="utf-8-sig", dtype={"tm": str})
    obs["ws"] = pd.to_numeric(obs.ws, errors="coerce")
    obs.loc[obs.ws <= -90, "ws"] = np.nan
    obs["dtm"] = pd.to_datetime(obs.tm, format="%Y%m%d%H%M")
    piv = obs.pivot_table(index="dtm", columns="stn", values="ws")
    full_idx = pd.date_range(piv.index.min(), piv.index.max(), freq="h")
    piv = piv.reindex(full_idx)
    nwp = nwp_ws.reindex(full_idx)

    # 대상일: dtm은 종료시각이므로 (dtm-1h)의 날짜. cutoff = 그 전날 13:00.
    tgt = pd.DataFrame({"forecast_kst_dtm": target_dtms.unique()})
    day = (tgt.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.normalize()
    tgt["cutoff"] = day - pd.Timedelta(days=1) + pd.Timedelta(hours=13)

    out = tgt[["forecast_kst_dtm"]].copy()
    for stn in OBS_STATIONS:
        s = piv[stn]
        r12 = s.rolling(12, min_periods=6).mean()
        r24 = s.rolling(24, min_periods=12).mean()
        err7 = (s - nwp).rolling(168, min_periods=48).mean()
        for name, series in ((f"obs{stn}_r12", r12), (f"obs{stn}_r24", r24),
                             (f"obs{stn}_err7", err7)):
            out[name] = series.reindex(tgt.cutoff).to_numpy()
    return out


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    nwp_ws = feat.set_index("forecast_kst_dtm")["ldaps_ws50max"]
    obs_feat = build_obs_features(df.forecast_kst_dtm, nwp_ws)
    df = df.merge(obs_feat, on="forecast_kst_dtm", how="left")
    obs_cols = [c for c in obs_feat.columns if c != "forecast_kst_dtm"]
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    aug_cols = base_cols + obs_cols
    cov = df[df.forecast_kst_dtm.dt.year == 2024][obs_cols].notna().mean().mean()
    print(f"ready: 관측피처 {len(obs_cols)}개, 2024 커버리지 {cov*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    # ablation: 관측 피처를 수준(r12/r24)과 오차(err7)로 분리
    lvl_cols = [c for c in obs_cols if c.endswith(("_r12", "_r24"))]
    err_cols = [c for c in obs_cols if c.endswith("_err7")]
    variant_cols = {
        "base": base_cols,
        "level": base_cols + lvl_cols,      # 관측 풍속 수준만 (지속성/기후 신호)
        "err": base_cols + err_cols,        # NWP 오차 rolling만 (v8d 위험 계열)
        "all": aug_cols,
    }
    res = {v: {} for v in variant_cols}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        fits = {}
        for v, cols in variant_cols.items():
            X, y, w = stack_groups(tr, cols, w_tr)
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            fits[v] = (cols, shared, {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
                                      .fit(X[shared], y, sample_weight=w) for q in QUANTILES_FULL})
        fs = {v: [] for v in res}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_base = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)

            for v in res:
                cols, shared, models = fits[v]
                Xv = group_X(sub, cols, tgt)
                qp = np.column_stack([np.clip(models[q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm = interp_atoms(qp, n=150)
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_base + (med - np.median(anen_base, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
        for v in res:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in res)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v19 실측 관측 피처 ablation ===")
    for v in res:
        print(f"{v}: {np.mean(list(res[v].values())):.4f}")


if __name__ == "__main__":
    main()
