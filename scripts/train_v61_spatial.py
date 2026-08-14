"""#94 v61: GFS 광역(0.25°, 11×11, ~275km) 공간 통계 피처 (티어3a, Andrade&Bessa).

제공 GFS는 9격자(~50km) 지점값 — 종관 패턴의 '형태'(구배·이상·주성분)는 미표현.
피처:
- sp: g_mean/g_std (광역 평균·표준편차), gradEW/gradNS (동서·남북 구배 = 종관 기압계
      위상), site_anom (사이트 인접 격자 − 광역 평균 = 국지 이상)
- pc : 121차원 풍속장의 PC1~3 (fold별 학습 구간에서만 적합 — 보수적)
변형: base / sp / pc / sp_pc. 하네스 v58 동일 (6-fold q3, 풀링 병기).
판정 기준(#93): 봄 fold 페널티(-0.005)를 상쇄할 IFS10급 신호여야 생존.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


def load_grid():
    d = pd.read_csv(PROJECT / "external_data/noaa_gfs/gfs_grid100_20220101_20251231.csv",
                    encoding="utf-8-sig")
    run_dt = pd.to_datetime(d["run"].astype(str), format="%Y%m%d%H")
    d["forecast_kst_dtm"] = run_dt + pd.to_timedelta(d["fxx"] + 9, unit="h")
    ws_cols = [f"ws_{i}_{j}" for i in range(11) for j in range(11)]
    W = d[ws_cols].to_numpy()  # (n, 121), 남→북(i)·서→동(j)
    out = pd.DataFrame({"forecast_kst_dtm": d["forecast_kst_dtm"]})
    out["g_mean"] = W.mean(axis=1)
    out["g_std"] = W.std(axis=1)
    G = W.reshape(-1, 11, 11)
    out["gradEW"] = G[:, :, 6:].mean(axis=(1, 2)) - G[:, :, :5].mean(axis=(1, 2))
    out["gradNS"] = G[:, 6:, :].mean(axis=(1, 2)) - G[:, :5, :].mean(axis=(1, 2))
    # 사이트(37.284N,128.958E) 최근접 셀 = i=5(37.25), j=6(129.0)
    out["site_anom"] = G[:, 5, 6] - out["g_mean"]
    return out.sort_values("forecast_kst_dtm").reset_index(drop=True), \
        pd.DataFrame(W, index=out.index), out["forecast_kst_dtm"]


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

    sp, W_raw, w_dtm = load_grid()
    df = df.merge(sp, on="forecast_kst_dtm", how="left")
    # 원장(121차원)을 df 행에 정렬 (PC용)
    wmap = pd.DataFrame(W_raw.to_numpy(), columns=[f"_w{k}" for k in range(121)])
    wmap["forecast_kst_dtm"] = w_dtm.to_numpy()
    df = df.merge(wmap, on="forecast_kst_dtm", how="left")
    wcols = [f"_w{k}" for k in range(121)]
    print(f"광역 커버리지: {df['g_mean'].notna().mean()*100:.2f}%", flush=True)

    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols0 = base_cols + ifs10
    sp_cols = ["g_mean", "g_std", "gradEW", "gradNS", "site_anom"]
    pc_cols = ["pc1", "pc2", "pc3"]
    VARIANTS = {"base": cols0, "sp": cols0 + sp_cols, "pc": cols0 + pc_cols,
                "sp_pc": cols0 + sp_cols + pc_cols}
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
        # PC: fold 학습 구간에서만 적합 → 전체 변환
        wtr = df.loc[tr_idx, wcols].dropna()
        pca = PCA(n_components=3).fit(wtr.to_numpy())
        wall = df[wcols]
        pcs = np.full((len(df), 3), np.nan)
        ok = wall.notna().all(axis=1).to_numpy()
        pcs[ok] = pca.transform(wall[ok].to_numpy())
        df["pc1"], df["pc2"], df["pc3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]

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

    print("\n=== v61 광역 공간 통계 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:6s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
