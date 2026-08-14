"""v44: ECMWF ENS 8멤버 앙상블 피처 — 유일하게 남은 신규 정보축 (흐름 의존 불확실성).

- 재료: ens_point_2022_2025.csv (cf+pf7, 10m u/v, 3h, 전 기간 100%).
- 신규성 논거: 현 피처엔 '예보 불확실성의 크기'가 없음 (cons3_std는 3개 결정론
  모델의 불일치 — 단일 모델 초기조건 앙상블의 흐름 의존 산포와 다른 정보).
  #50(GEFS 시나리오 주입 실패)과 구별: 원자 주입이 아니라 **조건부 피처** —
  GBM이 "산포가 클 때의 조건부 분포"를 스스로 배우게 한다.
- 피처: ens_ws_mean(8멤버 평균), ens_ws_std(멤버 간 산포), ens_ws_rstd(상대 산포).
- 변형: base(공식 feat10) / mean / spread(std+rstd) / all(3개).
기준: CV 0.6591. 하네스 v31b 동일. 판정 기준: 명확한 CV 이득 (6-fold 방향 일관).
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
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
ENS_CSV = PROJECT / "external_data" / "ecmwf_ifs" / "ens_point_2022_2025.csv"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_ens_features() -> pd.DataFrame:
    d = pd.read_csv(ENS_CSV, encoding="utf-8-sig")
    d = d.drop_duplicates(subset=["run", "fxx", "member", "var", "lat", "lon"])
    piv = d.pivot_table(index=["run", "fxx", "member", "lat", "lon"],
                        columns="var", values="value")
    ws = np.sqrt(piv["u10"] ** 2 + piv["v10"] ** 2)  # 격자·멤버별 스칼라 풍속
    mem = ws.groupby(level=["run", "fxx", "member"]).mean()  # 격자 평균 → 멤버 풍속
    g = mem.groupby(level=["run", "fxx"])
    out = pd.DataFrame({"ens_ws_mean": g.mean(), "ens_ws_std": g.std()})
    out["ens_ws_rstd"] = out.ens_ws_std / out.ens_ws_mean.clip(lower=0.5)
    out = out.reset_index()
    out["forecast_kst_dtm"] = (pd.to_datetime(out.run) + pd.Timedelta(hours=9)
                               + pd.to_timedelta(out.fxx, unit="h"))
    # 연속 런의 fxx39/15가 같은 KST 시각으로 겹침 → 평균 (기존 로더들과 동일 처리)
    out = out.groupby("forecast_kst_dtm")[["ens_ws_mean", "ens_ws_std", "ens_ws_rstd"]].mean()
    out = out.sort_index()
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full).interpolate(limit=2)
    out.index.name = "forecast_kst_dtm"
    return out.reset_index()


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ens_features(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    adds = {
        "base": [],
        "mean": ["ens_ws_mean"],
        "spread": ["ens_ws_std", "ens_ws_rstd"],
        "all": ["ens_ws_mean", "ens_ws_std", "ens_ws_rstd"],
    }
    cov24 = df.loc[df.forecast_kst_dtm.dt.year == 2024, adds["all"]].notna().mean()
    print("2024 커버리지:\n" + cov24.to_string(), flush=True)
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    variants = list(adds)
    res = {v: {} for v in variants}
    dec = {v: {} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        models = {}
        cols_v = {}
        for v in variants:
            cols = base_cols + ifs10 + adds[v]
            X, y, w = stack_groups(tr, cols, w_tr)
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[v] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
            cols_v[v] = (cols, shared)
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        dc = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            for v in variants:
                cols, shared = cols_v[v]
                Xv = group_X(sub, cols, tgt)
                qp = np.column_stack([np.clip(models[v][q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm = interp_atoms(qp, n=150)
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v44 ENS 앙상블 피처 ===")
    for v in variants:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v:6s}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f}")


if __name__ == "__main__":
    main()
