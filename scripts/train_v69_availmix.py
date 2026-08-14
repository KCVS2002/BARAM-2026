"""#105 v69 (SCADA 마지막 카드): 정지 시나리오 원자 명시화 — g1/g2 가용률 혼합 확장.

#101: g1 정지율 최다 (유효시간 22.8%)인데 가용률 혼합 원자는 g3만 적용 중.
#102: 다운웨이트는 FICR·NMAE 1:1 교환 — 올바른 처방은 정지를 지우는 게 아니라
**원자 분포에 정지 시나리오를 명시**하는 것 (g3 혼합의 일반화).
과거 실패(sub_012 홀드아웃)는 5기용 인자(0.8/0.6)를 6기 그룹에 그대로 쓴 것 —
이번엔 6기 물리 인자 (1기 정지 5/6≈0.833, 2기 4/6≈0.667).

변형: base(현행 — g3만 혼합) / mixA(g1/g2 p=0.15, g3와 동일 상수)
      / mixB(SCADA 실측 비례: g1 p=0.20 / g2 p=0.11) / mixL(저용량 p=0.08).
혼합 문법은 공식 g3와 동일: n8=p·2/3 (1기 정지), n6=p·1/3 (2기 정지) 원자를
gbm150 서브샘플 ×인자로 추가. 하네스 v58 동일. 판정 #91 기준.
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

from scripts.diag_scada_outage import RATED, hourly_turbines
from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_TURBINES, label_weights
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
VARIANTS = ["base", "mixA", "mixB", "mixL"]
P_MIX_BY = {"mixA": {"kpx_group_1": 0.15, "kpx_group_2": 0.15},
            "mixB": {"kpx_group_1": 0.20, "kpx_group_2": 0.11},
            "mixL": {"kpx_group_1": 0.08, "kpx_group_2": 0.08}}


def outage_flags(tgt):
    """kst_dtm 인덱스의 (정지 ≥1기, 부분결손 ≥1기) 불리언 시리즈."""
    scada_file, turbines = GROUP_TURBINES[tgt]
    rated = RATED[tgt]
    energy, _, _ = hourly_turbines(scada_file, turbines, rated)
    E = energy.dropna()
    Ev = E.to_numpy()
    peer = np.empty(Ev.shape)
    for j in range(Ev.shape[1]):
        peer[:, j] = np.median(np.delete(Ev, j, axis=1), axis=1)
    out = ((Ev / rated < 0.01) & (peer / rated > 0.15)).any(axis=1)
    part = ((Ev < 0.4 * peer) & (peer / rated > 0.20)).any(axis=1) & ~out
    return pd.Series(out, index=E.index), pd.Series(part, index=E.index)


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
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols = base_cols + ifs10
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

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

        X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
        models_one = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        models = {v: models_one for v in VARIANTS}
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

            Xv = group_X(sub, cols, tgt)
            qp = np.column_stack([np.clip(models_one[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm300 = interp_atoms(qp, n=300)
            gbm150 = gbm300[:, np.linspace(0, 299, 150).astype(int)]
            med = np.median(gbm300, axis=1)
            base_atoms = np.empty((len(sub), 300))
            for i in range(len(sub)):
                na = NA_OFF[terc[i]]
                ng = 300 - na
                an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                gb = np.interp(np.linspace(0, 299, ng), np.arange(300), gbm300[i])
                base_atoms[i] = np.sort(np.concatenate([an, gb]))
            for v in VARIANTS:
                p_mix = P_MIX_BY.get(v, {}).get(tgt, 0.0)
                if p_mix <= 0:
                    atoms = base_atoms
                else:
                    n = 300
                    n8 = max(int(round(n * p_mix * 2 / 3)), 1)
                    n6 = max(int(round(n * p_mix * 1 / 3)), 1)
                    extra = [gbm150[:, np.linspace(0, 149, n8).astype(int)] * (5 / 6),
                             gbm150[:, np.linspace(0, 149, n6).astype(int)] * (4 / 6)]
                    atoms = np.sort(np.concatenate([base_atoms] + extra, axis=1), axis=1)
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

    print("\n=== v69 g1/g2 가용률 혼합 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:5s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
