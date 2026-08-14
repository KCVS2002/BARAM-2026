"""#133 v86: TabPFN sister 6-fold 전면 확증 — 파일럿(#132) 관문 통과 후속.

파일럿: pinball 비율 0.928~0.994 (사상 최초 품질 동급~우위 sister), 풀링 +0.0011~+0.0164.
변형 (전 그룹 — g3 포함: 사전학습 in-context는 데이터 기근에 강할 가설):
  base   : 공식 cur75 (g1/g2 CNN75, g3 없음)
  tpf75  : base + TabPFN 75원자 (전 그룹)
  tpf150 : base + TabPFN 150원자 (강도)
  tpfrep : CNN75 → TabPFN75 교체 (g1/g2), g3는 tpf75와 동일 (보완 vs 대체 판별)
TabPFN qp 캐시: {fold}_{tgt}_tpfqp.npz (파일럿 4쌍 재사용, 나머지 14쌍 생성 ~25분).
설정: ctx 6000 (q3 가중 서브샘플, seed 42), n_estimators 2, 배치 128 (GPU 부하 절충).
판정 #91, 판정은 사용자.
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
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
QS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CTX = 6000
VARIANTS = ["base", "tpf75", "tpf150", "tpfrep"]


def gen_tpf_qp(fold, tgt, df, w_q3, cols, t0):
    from tabpfn import TabPFNRegressor
    cap = CAPACITY_KWH[tgt]
    z = np.load(CACHE / f"{fold}_{tgt}.npz")
    dtm = z["dtm"]
    va_start = pd.Timestamp(int(fold[:4]), int(fold[5:7]), 1, 1)
    trm = (df.forecast_kst_dtm < va_start) & df[tgt].notna()
    tr = df[trm]
    w = w_q3[tgt].to_numpy()[trm.to_numpy()]
    rng = np.random.default_rng(42)
    p = w / w.sum()
    idx = rng.choice(len(tr), size=min(CTX, len(tr)), replace=False, p=p)
    Xtr = tr[cols].to_numpy(dtype=np.float32)[idx]
    ytr = (tr[tgt] / cap).to_numpy(dtype=np.float32)[idx]
    Xva = df.set_index("key").loc[dtm][cols].to_numpy(dtype=np.float32)
    reg = TabPFNRegressor(n_estimators=2, ignore_pretraining_limits=True,
                          random_state=42, device="auto", memory_saving_mode=True)
    reg.fit(Xtr, ytr)
    qp_t = np.empty((len(Xva), len(QS)), dtype=np.float64)
    for s in range(0, len(Xva), 128):
        res = reg.predict(Xva[s:s + 128], output_type="quantiles", quantiles=QS)
        qp_t[s:s + 128] = np.column_stack(res) if isinstance(res, list) else res
    qp_t = np.clip(np.sort(qp_t, axis=1), 0, 1) * cap
    qp_t = np.sort(smooth_quantiles_by_day(qp_t, pd.Series(pd.to_datetime(dtm))), axis=1)
    np.savez_compressed(CACHE / f"{fold}_{tgt}_tpfqp.npz", qp=qp_t)
    print(f"[{fold} {tgt}] TabPFN qp 캐시 (ctx {len(Xtr)}) ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    print(f"frame ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    gsc = {v: {t: [] for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        fold = f"2024-{m0:02d}"
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            if not (CACHE / f"{fold}_{tgt}_tpfqp.npz").exists():
                gen_tpf_qp(fold, tgt, df, w_q3, cols, t0)
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            has_cnn = tgt != "kpx_group_3"
            if has_cnn:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            qtp = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
            tp300 = interp_atoms(qtp, n=2 * K)
            tp75 = tp300[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            tp150 = tp300[:, np.linspace(0, 2 * K - 1, 150).astype(int)]
            for v in VARIANTS:
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb]
                    if has_cnn and v != "tpfrep":
                        parts.append(cnn75[i])
                    if v in ("tpf75", "tpfrep"):
                        parts.append(tp75[i])
                    elif v == "tpf150":
                        parts.append(tp150[i])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                pred = optimize_submission(np.array(atoms_l), cap, a_bar)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                gsc[v][tgt].append(s_)
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v86 TabPFN 6-fold — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:6s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")
    print("-- 그룹별 fold평균 --")
    for v in VARIANTS:
        print(f"{v:6s}: " + " ".join(f"{t[-1]}={np.nanmean(gsc[v][t]):.4f}" for t in TARGET_COLS))
    for v in VARIANTS:
        print(f"{v:6s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
