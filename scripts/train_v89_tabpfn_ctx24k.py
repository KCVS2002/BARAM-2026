"""#137 v89: TabPFN 컨텍스트 24k 게이트 — 스케일링 끝점 (사실상 전체 풀) 검사.

v88: 6k→12k에서 품질↑ fold와 점수↑가 정합 (cmix +0.0014, 바 경계선).
24k는 CV per-그룹 풀(1.7만~2.1만행)을 초과 → 실질 '전체 풀 컨텍스트'.
주의: TabPFN v2 사전학습은 ≤10k행 — 한계 초과 외삽의 열화 여부도 이 게이트가 판별.

대상: fold 01(12k 최대 개선)·09·11(2025 인접) × g1/g2.
변형: base(tpf75 6k) / c24(24k 75 교체) / cmix24(6k 75 + 24k 50).
게이트: pinball 비율·GBM 상관·풀링 델타 (12k 결과와 3중 비교). 판정은 사용자.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
QS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CTX = 24000
FOLDS = ["2024-01", "2024-09", "2024-11"]
G12 = ["kpx_group_1", "kpx_group_2"]
VARIANTS = ["base", "c24", "cmix24"]


def pinball(y, qp, qs):
    tot = 0.0
    for j, q in enumerate(qs):
        d = y - qp[:, j]
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / len(qs)


def gen_tpf_qp(fold, tgt, df, w_q3, cols, t0):
    from tabpfn import TabPFNRegressor
    cap = CAPACITY_KWH[tgt]
    z = np.load(CACHE / f"{fold}_{tgt}.npz")
    dtm = z["dtm"]
    va_start = pd.Timestamp(int(fold[:4]), int(fold[5:7]), 1, 1)
    trm = (df.forecast_kst_dtm < va_start) & df[tgt].notna()
    tr = df[trm]
    w = w_q3[tgt].to_numpy()[trm.to_numpy()]
    n_ctx = min(CTX, len(tr))
    if n_ctx < len(tr):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(tr), size=n_ctx, replace=False, p=w / w.sum())
    else:
        idx = np.arange(len(tr))
    Xtr = tr[cols].to_numpy(dtype=np.float32)[idx]
    ytr = (tr[tgt] / cap).to_numpy(dtype=np.float32)[idx]
    Xva = df.set_index("key").loc[dtm][cols].to_numpy(dtype=np.float32)
    reg = TabPFNRegressor(n_estimators=2, ignore_pretraining_limits=True,
                          random_state=42, device="auto", memory_saving_mode=True)
    reg.fit(Xtr, ytr)
    qp_t = np.empty((len(Xva), len(QS)), dtype=np.float64)
    for s in range(0, len(Xva), 128):
        r = reg.predict(Xva[s:s + 128], output_type="quantiles", quantiles=QS)
        qp_t[s:s + 128] = np.column_stack(r) if isinstance(r, list) else r
    qp_t = np.clip(np.sort(qp_t, axis=1), 0, 1) * cap
    qp_t = np.sort(smooth_quantiles_by_day(qp_t, pd.Series(pd.to_datetime(dtm))), axis=1)
    np.savez_compressed(CACHE / f"{fold}_{tgt}_tpfqp_c24k.npz", qp=qp_t)
    print(f"[{fold} {tgt}] tpf ctx{len(Xtr)} 캐시 ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    print(f"frame ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    for fold in FOLDS:
        fs = {v: [] for v in VARIANTS}
        for tgt in G12:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
            cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            if not (CACHE / f"{fold}_{tgt}_tpfqp_c24k.npz").exists():
                gen_tpf_qp(fold, tgt, df, w_q3, cols, t0)
            q6 = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
            q12 = np.load(CACHE / f"{fold}_{tgt}_tpfqp_c12k.npz")["qp"]
            q24 = np.load(CACHE / f"{fold}_{tgt}_tpfqp_c24k.npz")["qp"]
            pbg = pinball(actual, qp_gbm, QS)
            print(f"  {fold} {tgt[-1]}: pinball비 6k {pinball(actual,q6,QS)/pbg:.3f} / "
                  f"12k {pinball(actual,q12,QS)/pbg:.3f} / 24k {pinball(actual,q24,QS)/pbg:.3f} | "
                  f"GBM상관 24k {np.corrcoef(q24[:,9]-actual, qp_gbm[:,9]-actual)[0,1]:.3f}", flush=True)
            t6 = interp_atoms(q6, n=2 * K)
            t24 = interp_atoms(q24, n=2 * K)
            tp6_75 = t6[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            tp24_75 = t24[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            tp24_50 = t24[:, np.linspace(0, 2 * K - 1, 50).astype(int)]
            for v in VARIANTS:
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb, cnn75[i]]
                    if v == "base":
                        parts.append(tp6_75[i])
                    elif v == "c24":
                        parts.append(tp24_75[i])
                    else:
                        parts += [tp6_75[i], tp24_50[i]]
                    atoms_l.append(np.sort(np.concatenate(parts)))
                pred = optimize_submission(np.array(atoms_l), cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
        print(f"fold {fold} (g1/g2 평균): " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v89 24k 게이트 (g1/g2, fold 01/09/11) ===")
    for v in VARIANTS:
        print(f"{v:6s}: 평균 {np.mean(list(res[v].values())):.4f} | "
              + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
