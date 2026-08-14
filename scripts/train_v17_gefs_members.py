"""v17: GEFS 실물 16멤버 시나리오 원자 (plan_v17_gefs_members.md 실행).

v16(스프레드 요약 z-시나리오, 기각)과의 차이: 균일 배율이 아니라 멤버별 실물 풍속비
f_m(t) = ws_member(t) / ws_ensmean(t) — 시간 상관을 가진 진짜 대안 흐름.
- 예측 시점 피처 치환(학습 불변): 풍속류 × f_m, CUBE_COLS × f_m³
- 멤버 16 × Q_SCEN 5분위 = 80원자 → GBM 중앙값 재정렬 → gbm+anen에 결합
- 기준: sub_009 구성(gbm+anen) CV 0.6470 (v15/v16과 동일 골격·동일 기준)
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
GEFS_CSV = PROJECT / "external_data" / "gefs" / "gefs_members_2024_2025.csv"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
Q_SCEN = [0.1, 0.3, 0.5, 0.7, 0.9]
CUBE_COLS = ("wpd_100m", "wpd_ldaps50")
GUST_COLS = ("gfs_surface_0_gust",)


def load_member_factors() -> pd.DataFrame:
    """멤버별 풍속비 f_m: (forecast_kst_dtm × 16멤버) 1h 보간 완료 상태로 반환."""
    g = pd.read_csv(GEFS_CSV, encoding="utf-8-sig")
    g["run_dt"] = pd.to_datetime(g["run"].astype(str), format="%Y%m%d%H")
    g["forecast_kst_dtm"] = g.run_dt + pd.Timedelta(hours=9) + pd.to_timedelta(g.fxx, unit="h")
    # 격자별 스칼라 풍속 → 9격자 평균 (벡터평균 금지 원칙)
    piv = g.pivot_table(index=["forecast_kst_dtm", "member", "lat", "lon"],
                        columns="var", values="value", aggfunc="mean")
    ws = np.sqrt(piv["u10"] ** 2 + piv["v10"] ** 2)
    ws = ws.groupby(level=["forecast_kst_dtm", "member"]).mean().unstack("member")
    ens = ws.mean(axis=1).clip(lower=0.3)
    f = ws.div(ens, axis=0).clip(0.4, 1.8)
    # 3h → 1h 보간
    full = pd.date_range(f.index.min(), f.index.max(), freq="h")
    f = f.reindex(full).interpolate(limit=3)
    f.index.name = "forecast_kst_dtm"
    return f


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    fmat = load_member_factors()
    members = list(fmat.columns)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ws_cols = [c for c in base_cols if c.startswith(("ldaps_ws", "gfs_ws"))
               and not c.endswith(("_dsin", "_dcos"))] + list(GUST_COLS)
    cov24 = fmat.reindex(df.loc[df.forecast_kst_dtm.dt.year == 2024, "forecast_kst_dtm"]).notna().all(axis=1).mean()
    print(f"ready: 멤버 {len(members)}개, 2024 f_m 커버리지 {cov24*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in ("base", "mem")}
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
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in res}
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

            fsub = fmat.reindex(sub["forecast_kst_dtm"])
            scen_atoms = []
            for m in members:
                f_use = fsub[m].fillna(1.0).to_numpy()
                Xs = Xv.copy()
                for c in ws_cols:
                    Xs[c] = Xs[c] * f_use
                for c in CUBE_COLS:
                    if c in Xs:
                        Xs[c] = Xs[c] * f_use ** 3
                for q in Q_SCEN:
                    scen_atoms.append(np.clip(models[q].predict(Xs[shared]) * cap, 0, cap))
            scen = np.sort(np.column_stack(scen_atoms), axis=1)  # (n, 80)
            scen = np.clip(scen + (med - np.median(scen, axis=1))[:, None], 0, cap)
            scen = np.repeat(scen, 2, axis=1)  # 160원자로 gbm/anen과 균형

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            for v, extra in (("base", None), ("mem", scen)):
                parts = [gbm, anen] + ([extra] if extra is not None else [])
                atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
        for v in res:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: base={res['base'][fold]:.4f} mem={res['mem'][fold]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v17 GEFS 실물 멤버 시나리오 ===")
    for v in res:
        print(f"{v}: {np.mean(list(res[v].values())):.4f}")


if __name__ == "__main__":
    main()
