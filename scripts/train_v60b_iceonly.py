"""#93 v60: 착빙 다일 누적 피처 (티어3b, domain_theory.md §1 착빙 지속·이월 효과).

v42(당일 예보 착빙 조건부 원자, 무효)와의 차이: 당일 조건이 아니라 **과거 수일의
관측된 착빙 조건 누적** — 정착한 얼음이 사건 후 수일 지속되는 이월 효과. 현행
피처는 전부 당일 예보라 이 상태를 원리적으로 못 봄 (진짜 신규 정보).

피처 (대상일 D, 관측창 종료 = D-1 12:00 KST — 예측기준시점 D-1 13:00 이전):
- ice24 / ice72: 직전 24h/72h 중 착빙 조건(기온 -12~0°C & 습도 ≥85%) 시간수 (태백 216)
- tmin48: 직전 48h 최저기온
누수: 관측은 실시간 공개 — 각 예측값의 예측기준시점 이전 자료만 사용. CV(2024)는
학습기간 관측만 관여. 테스트 적용(2025 관측)은 별도 규칙 판단 후.
변형: base / ice. 하네스 v58 동일 (6-fold q3, 풀링 병기). 겨울 fold(01·11) 주목.
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
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_ice_features() -> pd.DataFrame:
    """대상일별 착빙 누적 피처 (관측창 종료 D-1 12:00)."""
    obs = pd.read_csv(PROJECT / "external_data/kma_aws/aws_hourly_merged.csv",
                      encoding="utf-8-sig", parse_dates=["tm"])
    obs = obs[obs.stn == 216].sort_values("tm").set_index("tm")  # 태백 ASOS (712m, 최근접)
    obs["ice"] = ((obs.ta <= 0) & (obs.ta >= -12) & (obs.hm >= 85)).astype(float)
    rows = []
    days = pd.date_range(obs.index.min().normalize() + pd.Timedelta(days=4),
                         obs.index.max().normalize() + pd.Timedelta(days=1))
    for d in days:  # d = 대상일 D
        end = d - pd.Timedelta(hours=12)  # D-1 12:00
        w24 = obs.loc[end - pd.Timedelta(hours=24): end]
        w72 = obs.loc[end - pd.Timedelta(hours=72): end]
        w48 = obs.loc[end - pd.Timedelta(hours=48): end]
        rows.append({"day": d, "ice24": w24["ice"].sum(), "ice72": w72["ice"].sum(),
                     "tmin48": w48["ta"].min()})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    w_q3 = clean_w.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        w_q3[tgt] = w_q3[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)

    ice = load_ice_features()
    # 대상일 = (forecast_kst_dtm − 1h)의 날짜 (kst_dtm은 구간 종료 시각)
    df["day"] = (df.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.normalize()
    df = df.merge(ice, on="day", how="left").drop(columns=["day"])
    print(f"ice 커버리지: {df['ice24'].notna().mean()*100:.2f}% | "
          f"착빙일(ice24>0) 비율: {(df['ice24'] > 0).mean()*100:.1f}%", flush=True)

    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols0 = base_cols + ifs10
    VARIANTS = {"base": cols0, "ice2": cols0 + ["ice24", "ice72"]}
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    variants = list(VARIANTS)
    res = {v: {} for v in variants}
    dec = {v: {} for v in variants}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in variants}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        models = {}
        for v in variants:
            cols = VARIANTS[v]
            X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[v] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
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
                cols = VARIANTS[v]
                Xv = group_X(sub, cols, tgt)
                shared = cols + ["g_rated", "g_rotor", "g_id"]
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
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v60b 착빙 누적 (tmin48 제외) — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in variants:
        fm = np.mean(list(res[v].values()))
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        ps = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_)
        print(f"{v:5s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
