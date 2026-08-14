"""v41: 연도 경계 검증(2023→2024) 구축 + 과거 CV↔LB 역전 사례 사후 재현 테스트.

가설: rolling CV의 실패 원인이 "fold가 연도 경계 안쪽"이라면, 학습 <2024 → 검증
2024 전체(연도 경계를 넘는 단일 홀드아웃)는 LB(2024→2025 전이)의 방향을 재현해야 한다.

재현 대상 (LB 실측 부호):
- feat10 vs feat5 : LB +0.0016 (#66 채택)      | rolling CV +0.0057
- reg vs base     : LB -0.0020 (#52 기각)      | rolling CV +0.0052 (역전!)
- recency vs base : LB -0.013  (#20 기각)      | rolling CV +0.008  (역전!)
- vincent vs base : LB -0.0017 (#61 기각)      | rolling CV +0.0039 (역전!)
- alpha vs base   : LB -0.0012 (#54 기각)      | rolling CV +0.0009 (역전)
- bag vs base     : LB +0.0003 (#64 채택)      | rolling CV +0.0024

판정: 대형 역전 3건(reg·recency·vincent)의 부호를 맞추면 도구 유효 —
이후 제출 전 필터로 사용 + 봉인 축 재심사 근거. 소형 2건(alpha·bag)은 참고.
주의: g3는 2023 라벨뿐이라 학습 1년 — 그룹별 분해도 출력.
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
REG_PARAMS = {**BASE_PARAMS, "num_leaves": 31, "min_child_samples": 120,
              "subsample": 0.7, "colsample_bytree": 0.7}
VA_START = pd.Timestamp(2024, 1, 1, 1)
VA_END = pd.Timestamp(2025, 1, 1, 1)


def optimize_bagged(atoms, cap, a_bar, B=15, frac=0.6, seed=42):
    rng = np.random.default_rng(seed)
    n = atoms.shape[1]
    k = int(n * frac)
    gs = np.empty((atoms.shape[0], B))
    for b in range(B):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        gs[:, b] = optimize_submission(atoms[:, idx], cap, a_bar)
    return np.clip(gs.mean(axis=1), 0, cap)


def train_qmodels(tr, cols, w_tr, params):
    X, y, w = stack_groups(tr, cols, w_tr)
    shared = cols + ["g_rated", "g_rotor", "g_id"]
    return shared, {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(
        X[shared], y, sample_weight=w) for q in QUANTILES_FULL}


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
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs5 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std"]
    phys5 = ["ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols5 = base_cols + ifs5
    cols10 = base_cols + ifs5 + phys5

    tr_idx = df.forecast_kst_dtm < VA_START
    tr = df[tr_idx]
    va = df[(df.forecast_kst_dtm >= VA_START) & (df.forecast_kst_dtm < VA_END)]
    w_clean = clean_w[tr_idx.to_numpy()]
    # 최근성 가중 (#20: 반감기 540일)
    age_days = (VA_START - tr.forecast_kst_dtm).dt.days.to_numpy()
    decay = 0.5 ** (age_days / 540.0)
    w_rec = w_clean.mul(decay, axis=0)
    print(f"학습 {len(tr)}행 (<2024) / 검증 {len(va)}행 (2024 전체) ({time.time()-t0:.0f}s)", flush=True)

    trainings = {
        "feat5": (cols5, w_clean, BASE_PARAMS),
        "base": (cols10, w_clean, BASE_PARAMS),
        "reg": (cols10, w_clean, REG_PARAMS),
        "recency": (cols10, w_rec, BASE_PARAMS),
    }
    models = {}
    for name, (cols, w_, p) in trainings.items():
        models[name] = (cols, *train_qmodels(tr, cols, w_, p))
        print(f"{name} 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

    variants = ["feat5", "base", "reg", "recency", "vincent", "alpha", "bag"]
    res = {v: {} for v in variants}
    blocks = {v: {t: {} for t in TARGET_COLS} for v in variants}
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
        month_block = ((sub.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.month - 1) // 2

        def build_atoms(model_key):
            cols, shared, ms = models[model_key]
            Xv = group_X(sub, cols, tgt)
            qp = np.column_stack([np.clip(ms[q].predict(Xv[shared]) * cap, 0, cap)
                                  for q in QUANTILES_FULL])
            qp.sort(axis=1)
            qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
            gbm = interp_atoms(qp, n=150)
            med = np.median(gbm, axis=1)
            anen = np.clip(anen_raw + (med - np.median(anen_raw, axis=1))[:, None], 0, cap)
            return gbm, anen

        atoms_cache = {}
        for mk in ("feat5", "base", "reg", "recency"):
            gbm, anen = build_atoms(mk)
            atoms_cache[mk] = (gbm, anen, np.sort(np.concatenate([gbm, anen], axis=1), axis=1))

        preds = {}
        for mk in ("feat5", "base", "reg", "recency"):
            preds[mk] = optimize_submission(atoms_cache[mk][2], cap, a_bar)
        gbm_b, anen_b, atoms_b = atoms_cache["base"]
        vin = np.sort((gbm_b + anen_b) / 2, axis=1)
        mix = np.sort(np.concatenate([gbm_b[:, ::2], anen_b[:, ::2]], axis=1), axis=1)
        preds["vincent"] = optimize_submission(np.sort(np.concatenate([vin, mix], axis=1), axis=1), cap, a_bar)
        preds["alpha"] = optimize_submission(atoms_b, cap, a_bar * 0.7)
        preds["bag"] = optimize_bagged(atoms_b, cap, a_bar)

        for v in variants:
            s, nm, fi, _ = metric_single(actual, preds[v], cap)
            res[v][tgt] = (s, nm, fi)
            for b in range(6):
                m = month_block == b
                sb, _, _, _ = metric_single(actual[m.to_numpy()], preds[v][m.to_numpy()], cap)
                blocks[v][tgt][b] = sb
        print(f"{tgt} 완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v41 연도 경계 검증 (학습<2024 → 검증 2024 전체) ===")
    print("변형 | 총점(3그룹) | g1 / g2 / g3 | 1-NMAE | FICR | (base 대비)")
    base_tot = np.nanmean([res["base"][t][0] for t in TARGET_COLS])
    for v in variants:
        tot = np.nanmean([res[v][t][0] for t in TARGET_COLS])
        gs = " / ".join(f"{res[v][t][0]:.4f}" for t in TARGET_COLS)
        nm = np.nanmean([res[v][t][1] for t in TARGET_COLS])
        fi = np.nanmean([res[v][t][2] for t in TARGET_COLS])
        print(f"{v:8s} {tot:.4f} | {gs} | {nm:.4f} | {fi:.4f} | {tot-base_tot:+.4f}")
    print("\n2개월 블록별 (base 대비, 3그룹 평균):")
    for v in variants:
        if v == "base":
            continue
        d = [np.nanmean([blocks[v][t][b] - blocks["base"][t][b] for t in TARGET_COLS]) for b in range(6)]
        wins = sum(1 for x in d if x > 0)
        print(f"{v:8s} " + " ".join(f"{x:+.4f}" for x in d) + f" | {wins}/6")
    print("""
LB 실측 부호 (재현 목표): feat5 -0.0016(feat10이 +) / reg -0.0020 / recency -0.013
                          vincent -0.0017 / alpha -0.0012 / bag +0.0003""")


if __name__ == "__main__":
    main()
