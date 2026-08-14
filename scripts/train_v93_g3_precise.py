"""#142 v93: g3 정밀 재검 — 공식 g3 구성(가용률 혼합+OM sister 조건부) 재현 위 TabPFN.

#141 진단: g3가 지배 결핍 (FICR 0.388 vs g1/g2 0.46~49, 절반 회복 = 총점 +0.013).
v86의 'TabPFN g3 중립'은 단순 base 구성 위 측정 — 공식 구성과의 상호작용 미검.
OM sister는 2024-only 학습이라 fold 09/11에서만 재현 가능 (게이트 프로토콜).

변형 (g3만):
  plain     : 단순 base (v86 기준 재현)
  plain_tpf : + tpf75 (v86 중립 재확인용)
  off       : 공식 구성 (base + 가용률 0.8/0.6 혼합 + OM sister 조건부 125/150/175)
  off_tpf   : 공식 구성 + tpf75
  off_tpf50 : 공식 구성 + tpf50 (약결합)
판정: off_tpf vs off 델타가 본 질문. 판정은 사용자.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_META, label_weights
from scripts.train_v22_omsister import load_om
from scripts.exp_cache import load_base_frame
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
DATA = PROJECT / "Data"
K = 150
P_MIX = 0.15
NA = {0: 125, 1: 150, 2: 175}
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TGT = "kpx_group_3"
VARIANTS = ["plain", "plain_tpf", "off", "off_tpf", "off_tpf50"]


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    weights = label_weights(lab)
    df, _, cols, _ = load_base_frame()
    om = load_om()
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    df = df.merge(om, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in cols if not c.startswith("ifs") and c not in ("cons3_mean", "cons3_std")]
    sis_cols = base_cols + om_cols
    shared_sis = sis_cols + ["g_rated", "g_rotor", "g_id"]
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    print(f"frame ready — sister 피처 {len(sis_cols)}개 ({time.time()-t0:.0f}s)", flush=True)

    cap = CAPACITY_KWH[TGT]
    res = {v: {} for v in VARIANTS}
    for m0 in (9, 11):
        fold = f"2024-{m0:02d}"
        va_start = pd.Timestamp(2024, m0, 1, 1)
        z = np.load(CACHE / f"{fold}_{TGT}.npz")
        qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
        actual, a_bar = z["actual"], float(z["a_bar"])
        dtm = z["dtm"]
        n_row = len(qp_gbm)
        gbm300 = interp_atoms(qp_gbm, n=2 * K)
        gbm_atoms = gbm300[:, np.linspace(0, 2 * K - 1, K).astype(int)]
        med = np.median(gbm_atoms, axis=1)
        d_bar = dist[:, :K].mean(axis=1)
        terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
        qtp = np.load(CACHE / f"{fold}_{TGT}_tpfqp.npz")["qp"]
        tp300 = interp_atoms(qtp, n=2 * K)
        tp75 = tp300[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
        tp50 = tp300[:, np.linspace(0, 2 * K - 1, 50).astype(int)]

        # ── OM sister 재현 (공식: 2024-only, 3그룹 스택, clean 가중) ──
        sisq_f = CACHE / f"{fold}_{TGT}_sisq.npz"
        if sisq_f.exists():
            sisq = np.load(sisq_f)["qp"]
        else:
            in_win = (df.forecast_kst_dtm.dt.year == 2024) & (df.forecast_kst_dtm < va_start)
            sX, sy, sw = [], [], []
            for tgt_i, tgt in enumerate(TARGET_COLS):
                c_ = CAPACITY_KWH[tgt]
                trm = df[tgt].notna() & in_win
                Xg = df.loc[trm, sis_cols].copy()
                rated, rotor = GROUP_META[tgt]
                Xg["g_rated"], Xg["g_rotor"] = rated, rotor
                Xg["g_id"] = tgt_i
                sX.append(Xg)
                sy.append(df.loc[trm, tgt] / c_)
                sw.append(weights.loc[trm.to_numpy(), tgt])
            sX, sy, sw = pd.concat(sX), pd.concat(sy), pd.concat(sw)
            print(f"[{fold}] sister 학습 {len(sX)}행 시작 ({time.time()-t0:.0f}s)", flush=True)
            va_rows = df.set_index("key").loc[dtm]
            Xv = va_rows[sis_cols].copy()
            rated, rotor = GROUP_META[TGT]
            Xv["g_rated"], Xv["g_rotor"] = rated, rotor
            Xv["g_id"] = TARGET_COLS.index(TGT)
            preds = []
            for q in QUANTILES:
                m_ = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                    sX[shared_sis], sy, sample_weight=sw)
                preds.append(np.clip(m_.predict(Xv[shared_sis]) * cap, 0, cap))
            sq = np.column_stack(preds)
            sq.sort(axis=1)
            sisq = np.sort(smooth_quantiles_by_day(sq, pd.Series(pd.to_datetime(dtm))), axis=1)
            np.savez_compressed(sisq_f, qp=sisq)
            print(f"[{fold}] sister 캐시 저장 ({time.time()-t0:.0f}s)", flush=True)
        sis300 = interp_atoms(sisq, n=2 * K)

        base_atoms = np.empty((n_row, 2 * K))
        for i in range(n_row):
            na = NA[terc[i]]
            ng = 2 * K - na
            an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
            an = np.clip(an + (med[i] - np.median(an)), 0, cap)
            gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
            base_atoms[i] = np.sort(np.concatenate([an, gb]))
        n8 = max(int(round(2 * K * P_MIX * 2 / 3)), 1)
        n6 = max(int(round(2 * K * P_MIX * 1 / 3)), 1)
        avail = np.concatenate([gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
                                gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6], axis=1)

        for v in VARIANTS:
            pred = np.empty(n_row)
            for t_ in (0, 1, 2):
                mrows = np.where(terc == t_)[0]
                atoms_l = []
                for i in mrows:
                    parts = [base_atoms[i]]
                    if v.startswith("off"):
                        parts.append(avail[i])
                        ns = NA[t_]
                        parts.append(np.interp(np.linspace(0, 2 * K - 1, ns), np.arange(2 * K), sis300[i]))
                    if v in ("plain_tpf", "off_tpf"):
                        parts.append(tp75[i])
                    elif v == "off_tpf50":
                        parts.append(tp50[i])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                pred[mrows] = optimize_submission(np.array(atoms_l), cap, a_bar)
            s_, nm, fi, _ = metric_single(actual, pred, cap)
            res[v][fold] = (s_, nm, fi)
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold][0]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v93 g3 정밀 재검 (fold 09/11) — 총점 / 1-NMAE / FICR ===")
    for v in VARIANTS:
        sc = np.mean([res[v][f][0] for f in res[v]])
        nm = np.mean([res[v][f][1] for f in res[v]])
        fi = np.mean([res[v][f][2] for f in res[v]])
        print(f"{v:9s}: {sc:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | "
              + " ".join(f"{f}:{res[v][f][0]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
