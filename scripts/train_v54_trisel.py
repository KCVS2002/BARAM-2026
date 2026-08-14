"""v54: 3단 모델 선택 — 저→plain / 중→q3 / 고→super(1+6cf³) (#85 후속).

v53 발견: 선택 구조가 저출력 NMAE를 보호하므로, 고출력 전용 모델은 단독 사용
불가능한 극단 가중도 투입 가능 — LB의 FICR 증폭(~2배) 하에서 파레토 개선 겨냥.

가설: q3 가중의 NMAE 비용은 저·중출력 시간의 정확도 저하에서 오고(용량 재배분),
FICR 이득은 고출력 시간에서 온다. 시간별로 모델을 **선택**하면 파레토 개선 가능.
- 함정 회피: 분위 평균 금지(선명화 v37/38), 원자 동시 풀링 금지(희석 v38b) — 선택만.
- 선택 신호: 무가중 모델의 med/cap 3분위 (가중 모델은 위치가 상향 편이라 부적합).

변형: plain(무가중 전체) / q3(가중 전체, 현행 공식) / sel_a(저·중→plain, 고→q3)
      / sel_b(저→plain, 중·고→q3)
+ 내장 진단: cf 대역별 |오차| 프로파일 (plain vs q3) — FICR 이득의 소재 확인.
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


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    w_q3 = clean_w.copy()
    w_sup = clean_w.copy()
    for _t in TARGET_COLS:
        _cf = np.clip((lab[_t] / CAPACITY_KWH[_t]).fillna(0).to_numpy(), 0, 1)
        w_q3[_t] = w_q3[_t].to_numpy() * (1 + 3 * _cf ** 2)
        w_sup[_t] = w_sup[_t].to_numpy() * (1 + 6 * _cf ** 3)
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
    print(f"ready ({time.time()-t0:.0f}s)", flush=True)

    variants = ["q3", "super", "sel3", "sel_hs"]
    res = {v: {} for v in variants}
    dec = {v: {} for v in variants}
    band_prof = {m: {b: [] for b in range(4)} for m in ("q3", "super")}  # 진단
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        tr = df[tr_idx]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        models = {}
        for mname, wsrc in (("plain", clean_w), ("q3", w_q3), ("super", w_sup)):
            X, y, w = stack_groups(tr, cols, wsrc[tr_idx.to_numpy()])
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[mname] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in variants}
        dc = {v: [] for v in variants}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            Xv = group_X(sub, cols, tgt)
            qps = {}
            for mname in ("plain", "q3", "super"):
                qp = np.column_stack([np.clip(models[mname][q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qps[mname] = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            med_sig = np.median(qps["plain"], axis=1) / cap
            terc = np.searchsorted(np.quantile(med_sig, [1 / 3, 2 / 3]), med_sig)

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen_raw = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()

            def qp_of(v):
                if v in ("plain", "q3", "super"):
                    return qps[v]
                if v == "sel3":  # 저→plain / 중→q3 / 고→super
                    qp = qps["plain"].copy()
                    qp[terc == 1] = qps["q3"][terc == 1]
                    qp[terc == 2] = qps["super"][terc == 2]
                    return qp
                qp = qps["q3"].copy()  # sel_hs: 저·중→q3 / 고→super
                qp[terc == 2] = qps["super"][terc == 2]
                return qp

            gs = {}
            for v in variants:
                qp = qp_of(v)
                gbm = interp_atoms(qp, n=150)
                med = np.median(gbm, axis=1)
                anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
                atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
                g = optimize_submission(atoms, cap, a_bar)
                gs[v] = g
                s_, nm, fi, _ = metric_single(actual, g, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
            # 진단: cf 대역별 |e| (valid 시간)
            vmask = actual >= 0.1 * cap
            bands = np.digitize(actual[vmask] / cap, [0.3, 0.5, 0.7])
            for mname in ("q3", "super"):
                e = np.abs(gs[mname] - actual)[vmask] / cap
                for b in range(4):
                    mm = bands == b
                    if mm.any():
                        band_prof[mname][b].append(e[mm].mean())
        for v in variants:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in variants)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v54 3단 모델 선택 ===")
    for v in variants:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v:6s}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f}")
    print("\n[진단] cf 대역별 평균 |오차| (q3 vs super):")
    labels = ["0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]
    for b in range(4):
        p = np.mean(band_prof["q3"][b])
        q = np.mean(band_prof["super"][b])
        print(f"  {labels[b]}: q3 {p:.4f} / super {q:.4f} ({(q-p)/p*+100:+.1f}%)")


if __name__ == "__main__":
    main()
