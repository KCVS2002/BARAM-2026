"""#107 v70 (롤링 Phase 1): 관측 기반 NWP 편향의 롤링 보정 — 구조 전환 1탄.

Phase 0(#106) 설계 반영:
- 참조 = 합성 지수: 지점 {100 대관령, 320 백운산, 216 태백} 각각
  g_s(t) = log((obs+0.5)/(ldws+0.5)) (ldws>2 시간만) → 3지점 평균 g(t)
- 각 행의 컷오프 C = 대상일 D-1 13:00. **C 이전 관측만** 사용 (시간 단위 인과):
  bias_W(D) = mean g(t), t ∈ [C−W일, C), W ∈ {7, 28}
  anom_W(D) = bias_W − bias_base (base = C 이전 확장 평균, 최소 60일) — 자기 기준선
  대비 이상치 (2024 상수 아님, 자기적응)
- 전 연도 대칭 적용 (2022~2025 동일 프로토콜) — 2025 특별취급 없음

변형:
  base   : 현행 공식
  roll_f : (1a) 피처 추가 [anom7, anom28, bias7]
  roll_c : (1b) 입력 보정 — 주 풍속 피처(ldaps_ws50max·gfs_ws100·ifs_ws925)를
           ×exp(clip(anom7, ±0.25))로 교정한 사본으로 **교체**
  roll_fc: 1a+1b 동시
하네스 v58 동일 (6-fold q3 + 풀링 병기). 판정 #91 + 창 둔감성 기준.
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
NA_OFF = {2: 175, 1: 150, 0: 125}
VARIANTS = ["base", "roll_f", "roll_c", "roll_fc"]
STATIONS = [100, 320, 216]
ROLL_COLS = ["anom7", "anom28", "bias7"]
WIND_MAIN = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]


def build_rolling_bias() -> pd.DataFrame:
    """대상일별 롤링 편향 지수 (컷오프 D-1 13:00 이전 관측만 — 시간 단위 인과)."""
    obs = pd.read_csv(PROJECT / "external_data/kma_aws/aws_hourly_merged.csv",
                      encoding="utf-8-sig", parse_dates=["tm"])
    obs.loc[(obs["ws"] < 0) | (obs["ws"] > 60), "ws"] = np.nan
    parts = []
    for f in ["train/ldaps_train.csv", "test/ldaps_test.csv"]:
        d = pd.read_csv(DATA / f, encoding="utf-8-sig",
                        usecols=["forecast_kst_dtm", "heightAboveGround_50_50MUmax",
                                 "heightAboveGround_50_50MVmax"],
                        parse_dates=["forecast_kst_dtm"])
        ws = np.hypot(d["heightAboveGround_50_50MUmax"], d["heightAboveGround_50_50MVmax"])
        parts.append(ws.groupby(d["forecast_kst_dtm"]).mean())
    ldws = pd.concat(parts).sort_index()

    gs = []
    for stn in STATIONS:
        s = obs[obs.stn == stn].set_index("tm")["ws"]
        j = pd.concat([s, ldws], axis=1, join="inner").dropna()
        j.columns = ["obs", "ld"]
        j = j[j["ld"] > 2]
        gs.append(np.log((j["obs"] + 0.5) / (j["ld"] + 0.5)))
    g = pd.concat(gs, axis=1).mean(axis=1).dropna().sort_index()  # 합성 지수 g(t)

    # 대상일별 컷오프 기반 롤링 통계 (누적합으로 O(n))
    gv = g.to_numpy()
    gt = g.index.to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(gv)])
    days = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    rows = []
    for D in days:
        C = (D - pd.Timedelta(days=1)) + pd.Timedelta(hours=13)
        i_c = np.searchsorted(gt, np.datetime64(C))
        row = {"day": D}
        for W, name in ((7, "bias7"), (28, "bias28")):
            i_s = np.searchsorted(gt, np.datetime64(C - pd.Timedelta(days=W)))
            n = i_c - i_s
            row[name] = (csum[i_c] - csum[i_s]) / n if n >= 24 else np.nan
        n_base = i_c
        row["bias_base"] = csum[i_c] / n_base if n_base >= 60 * 12 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["anom7"] = out["bias7"] - out["bias_base"]
    out["anom28"] = out["bias28"] - out["bias_base"]
    return out[["day", "bias7", "anom7", "anom28"]]


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

    rb = build_rolling_bias()
    df["day"] = (df.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.normalize()
    df = df.merge(rb, on="day", how="left").drop(columns=["day"])
    print(f"롤링 편향 커버리지: {df['anom7'].notna().mean()*100:.1f}% | "
          f"anom7 std {df['anom7'].std():.3f} (ready {time.time()-t0:.0f}s)", flush=True)
    # (1b) 보정 사본: 주 풍속 ×exp(clip(anom7, ±0.25))
    fac = np.exp(np.clip(df["anom7"].fillna(0), -0.25, 0.25))
    for c in WIND_MAIN:
        df[f"{c}_corr"] = df[c] * fac

    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols0 = base_cols + ifs10

    def swap_corr(cols):
        return [f"{c}_corr" if c in WIND_MAIN else c for c in cols]

    COLSET = {
        "base": cols0,
        "roll_f": cols0 + ROLL_COLS,
        "roll_c": swap_corr(cols0),
        "roll_fc": swap_corr(cols0) + ROLL_COLS,
    }

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        models = {}
        for v in VARIANTS:
            cols = COLSET[v]
            X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[v] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(
                ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            d_bar = dist[:, :150].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)

            for v in VARIANTS:
                cols = COLSET[v]
                Xv = group_X(sub, cols, tgt)
                shared = cols + ["g_rated", "g_rotor", "g_id"]
                qp = np.column_stack([np.clip(models[v][q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm300 = interp_atoms(qp, n=300)
                med = np.median(gbm300, axis=1)
                atoms = np.empty((len(sub), 300))
                for i in range(len(sub)):
                    na = NA_OFF[terc[i]]
                    ng = 300 - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 299, ng), np.arange(300), gbm300[i])
                    atoms[i] = np.sort(np.concatenate([an, gb]))
                pred = optimize_submission(atoms, cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v70 롤링 관측 편향 보정 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in VARIANTS:
        fm = np.mean(list(res[v].values()))
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        ps = []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, _, _, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_)
        print(f"{v:7s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
