"""#102 v67 (SCADA ③): 터빈 정지 오염 시간 다운웨이트 — 라벨 정제 실험.

진단(#101): 유효시간의 12~23%에 터빈 ≥1기 완전 정지 (1기 = 그룹 16.7~20% 하락),
현행 label_weights와 중첩 ~1% (신규 신호). 정지 시간의 라벨은 NWP로 설명 불가한
수준 이동 → 모델이 조건부 분산으로 오학습. 다운웨이트로 조건부 매핑을 정화.

원칙: **학습 가중만 변경, 검증 라벨 원본 유지** (정직한 CV — 2025에도 정지는
있으므로 정제가 과보정이면 CV가 벌한다). 탐지는 견고한 완전 정지(동료 중앙값
cf>15% & 자기 <1%)를 주 신호로, 부분결손(동료의 40% 미만, 웨이크 오검 가능)은
보조로만.

변형: base(현행 q3+label_weights) / dw03(정지시간 ×0.3) / dw01(×0.1)
      / dwp(정지 ×0.3 + 부분 ×0.7). 하네스 v58 동일. 판정 #91 기준.
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
VARIANTS = ["base", "dw03", "dw01", "dwp"]


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
    # SCADA 정지/부분 플래그 → 변형별 가중
    weights_by = {"base": w_q3}
    for v in ("dw03", "dw01", "dwp"):
        weights_by[v] = w_q3.copy()
    for tgt in TARGET_COLS:
        out_s, part_s = outage_flags(tgt)
        out_al = out_s.reindex(lab.kst_dtm).fillna(False).to_numpy()
        part_al = part_s.reindex(lab.kst_dtm).fillna(False).to_numpy()
        print(f"{tgt}: 정지 {out_al.sum()}시간 / 부분 {part_al.sum()}시간 플래그", flush=True)
        weights_by["dw03"][tgt] = weights_by["dw03"][tgt].to_numpy() * np.where(out_al, 0.3, 1.0)
        weights_by["dw01"][tgt] = weights_by["dw01"][tgt].to_numpy() * np.where(out_al, 0.1, 1.0)
        weights_by["dwp"][tgt] = (weights_by["dwp"][tgt].to_numpy()
                                  * np.where(out_al, 0.3, 1.0) * np.where(part_al, 0.7, 1.0))

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

        models = {}
        for v in VARIANTS:
            X, y, w = stack_groups(tr, cols, weights_by[v][tr_idx.to_numpy()])
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
                Xv = group_X(sub, cols, tgt)
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

    print("\n=== v67 SCADA 정지 다운웨이트 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
