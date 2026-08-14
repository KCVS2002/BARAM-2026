"""v25: ECMWF IFS 제3 NWP — 활용법 3종 동시 검증 (외부데이터 감사 결론 실행).

IFS는 유일하게 LDAPS 대비 '모델 스킬 우위'를 주장할 수 있는 소스 (OM으로 신호 실재
확인 +0.0002, 단 0.25°·2024-only 한계). 이번엔 전 기간(2022~2025)·기압면(925/850) 원본.

변형 (base 대비):
- feat  : 절제된 5피처 추가 — ifs_ws10/925/850 + 3원 컨센서스 평균·불일치(std)
          (피처 추가 7연속 실패 이력 — 명확한 CV 이득 없으면 즉시 기각)
- resid : 잔차 오프셋 부스팅 — base 분위 예측을 init_score로, IFS 피처 5개+핵심 풍속
          2개만으로 소형 보정 모델(분위별 200트리) 학습. 희석 없는 주입 (미시도 방법론)
- anen  : ANEN_FEATS에 ifs_ws925 추가(가중 1.5) — 아날로그 검색 공간 업그레이드

판정 기준(강화): 물리 메커니즘 + CV 명확 이득 동시 충족 시에만 LB 프로브.
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
IFS_CSV = PROJECT / "external_data" / "ecmwf_ifs" / "ifs_point_2022_2025.csv"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RESID_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                    min_child_samples=60, subsample=0.8, subsample_freq=1,
                    colsample_bytree=1.0, random_state=42, verbose=-1)


def load_ifs_features() -> pd.DataFrame:
    d = pd.read_csv(IFS_CSV, encoding="utf-8-sig")
    d = d.drop_duplicates(subset=["run", "fxx", "var", "lat", "lon"])
    d["forecast_kst_dtm"] = (pd.to_datetime(d.run) + pd.Timedelta(hours=9)
                             + pd.to_timedelta(d.fxx, unit="h"))
    piv = d.pivot_table(index=["forecast_kst_dtm", "lat", "lon"], columns="var", values="value")
    out = pd.DataFrame(index=piv.index.get_level_values(0).unique().sort_values())
    for lev in ("10", "925", "850"):
        u, v = piv.get(f"u{lev}"), piv.get(f"v{lev}")
        ws = np.sqrt(u ** 2 + v ** 2)  # 격자별 스칼라 → 평균 (벡터평균 금지)
        out[f"ifs_ws{lev}"] = ws.groupby(level="forecast_kst_dtm").mean()
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full).interpolate(limit=2)  # 3h → 1h
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
    ifs = load_ifs_features()
    df = df.merge(ifs, on="forecast_kst_dtm", how="left")
    # 3원 컨센서스: 표준화 후 평균·불일치
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std"]
    resid_cols = ifs_cols + ["ldaps_ws50max", "gfs_ws100"]
    anen_feats2 = ANEN_FEATS + ["ifs_ws925"]
    feat_w2 = np.append(FEAT_W, 1.5)
    cov24 = df.loc[df.forecast_kst_dtm.dt.year == 2024, ifs_cols].notna().mean().mean()
    print(f"ready: IFS 피처 5, 2024 커버리지 {cov24*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)

    variants = ("base", "feat", "resid", "anen")
    res = {v: {} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]
        w_tr = clean_w[tr_idx.to_numpy()]

        # base 모델 (공유)
        X, y, w = stack_groups(tr, base_cols, w_tr)
        shared = base_cols + ["g_rated", "g_rotor", "g_id"]
        bmodels = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        # feat 모델 (base+IFS 피처)
        Xf, yf, wf = stack_groups(tr, base_cols + ifs_cols, w_tr)
        shared_f = base_cols + ifs_cols + ["g_rated", "g_rotor", "g_id"]
        fmodels = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            Xf[shared_f], yf, sample_weight=wf) for q in QUANTILES_FULL}
        # resid 보정 모델: init_score = base 예측(cf 스케일)
        Xr, yr, wr = stack_groups(tr, resid_cols, w_tr)
        shared_r = resid_cols + ["g_rated", "g_rotor", "g_id"]
        rmodels = {}
        for q in QUANTILES_FULL:
            init = bmodels[q].predict(X[shared])
            rmodels[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, **RESID_PARAMS).fit(
                Xr[shared_r], yr, sample_weight=wr, init_score=init)
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, base_cols, tgt)
            Xvf = group_X(sub, base_cols + ifs_cols, tgt)
            Xvr = group_X(sub, resid_cols, tgt)
            qp_b = np.column_stack([np.clip(bmodels[q].predict(Xv[shared]) * cap, 0, cap)
                                    for q in QUANTILES_FULL])
            qp_f = np.column_stack([np.clip(fmodels[q].predict(Xvf[shared_f]) * cap, 0, cap)
                                    for q in QUANTILES_FULL])
            qp_r = np.column_stack([np.clip(
                (bmodels[q].predict(Xv[shared]) + rmodels[q].predict(Xvr[shared_r])) * cap, 0, cap)
                for q in QUANTILES_FULL])

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_b = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            # anen 변형: IFS 포함 검색 공간
            tr_ok2 = tr.loc[trm].dropna(subset=anen_feats2)
            sub2_ok = sub.dropna(subset=anen_feats2)
            if len(tr_ok2) > 1000 and len(sub2_ok) == len(sub):
                mu2 = tr_ok2[anen_feats2].mean()
                sd2 = tr_ok2[anen_feats2].std().replace(0, 1)
                knn2 = NearestNeighbors(n_neighbors=150).fit(
                    ((tr_ok2[anen_feats2] - mu2) / sd2 * feat_w2).to_numpy())
                _, aidx2 = knn2.kneighbors(((sub[anen_feats2].fillna(mu2) - mu2) / sd2 * feat_w2).to_numpy())
                anen_v = np.sort((tr_ok2[tgt] / cap).to_numpy()[aidx2] * cap, axis=1)
            else:
                anen_v = anen_b

            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            for v, (qp_use, anen_use) in {
                "base": (qp_b, anen_b), "feat": (qp_f, anen_b),
                "resid": (qp_r, anen_b), "anen": (qp_b, anen_v),
            }.items():
                qp = np.sort(qp_use, axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm = interp_atoms(qp, n=150)
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_use + (med - np.median(anen_use, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                pred = optimize_submission(atoms, cap, a_bar)
                s, _, _, _ = metric_single(actual, pred, cap)
                fs[v].append(s)
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v25 IFS 활용법 3종 ===")
    for v in variants:
        print(f"{v}: {np.mean(list(res[v].values())):.4f}")


if __name__ == "__main__":
    main()
