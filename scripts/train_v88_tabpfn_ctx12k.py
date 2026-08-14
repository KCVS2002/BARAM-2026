"""#136 v88: TabPFN 컨텍스트 스케일링 6k→12k — '학습 데이터 증량' 클래스 (v75c 전례).

v87 반증(같은 분포 서브샘플 = 같은 모델, 상관 0.97) 후 잔여 경로 = 컨텍스트 확대.
in-context 학습기의 ctx는 파라미터가 아니라 학습 데이터 — 스케일링 법칙 클래스.
진단: 12k의 pinball 비율(품질)과 GBM 상관(수렴 리스크, #124) 동시 실측.

변형 (g1/g2, g3 불변):
  base : 공식 sub_055 구성 (cur75 + tpf75 ctx6k)
  c75  : tpf75를 ctx12k로 교체
  cmix : tpf 6k 75원자 + 12k 50원자 (강도 125, 스케일 간 분할 — v87 div150 단서)
캐시: {fold}_{tgt}_tpfqp_c12k.npz. 판정 #91, 판정은 사용자.
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
CTX2 = 12000
VARIANTS = ["base", "c75", "cmix"]
G12 = ["kpx_group_1", "kpx_group_2"]


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
    rng = np.random.default_rng(42)
    p = w / w.sum()
    idx = rng.choice(len(tr), size=min(CTX2, len(tr)), replace=False, p=p)
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
    np.savez_compressed(CACHE / f"{fold}_{tgt}_tpfqp_c12k.npz", qp=qp_t)
    print(f"[{fold} {tgt}] tpf ctx{len(Xtr)} 캐시 ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    print(f"frame ready ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        fold = f"2024-{m0:02d}"
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            is_g12 = tgt in G12
            if is_g12:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                if not (CACHE / f"{fold}_{tgt}_tpfqp_c12k.npz").exists():
                    gen_tpf_qp(fold, tgt, df, w_q3, cols, t0)
                q6 = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
                q12 = np.load(CACHE / f"{fold}_{tgt}_tpfqp_c12k.npz")["qp"]
                pb6 = pinball(actual, q6, QS)
                pb12 = pinball(actual, q12, QS)
                pbg = pinball(actual, qp_gbm, QS)
                c6 = np.corrcoef(q6[:, 9] - actual, qp_gbm[:, 9] - actual)[0, 1]
                c12 = np.corrcoef(q12[:, 9] - actual, qp_gbm[:, 9] - actual)[0, 1]
                print(f"  {fold} {tgt[-1]}: pinball비 6k {pb6/pbg:.3f} → 12k {pb12/pbg:.3f} | "
                      f"GBM상관 {c6:.3f} → {c12:.3f}", flush=True)
                t6 = interp_atoms(q6, n=2 * K)
                t12 = interp_atoms(q12, n=2 * K)
                tp6_75 = t6[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tp12_75 = t12[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tp12_50 = t12[:, np.linspace(0, 2 * K - 1, 50).astype(int)]
            for v in VARIANTS:
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb]
                    if is_g12:
                        parts.append(cnn75[i])
                        if v == "base":
                            parts.append(tp6_75[i])
                        elif v == "c75":
                            parts.append(tp12_75[i])
                        else:
                            parts += [tp6_75[i], tp12_50[i]]
                    atoms_l.append(np.sort(np.concatenate(parts)))
                pred = optimize_submission(np.array(atoms_l), cap, a_bar)
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

    print("\n=== v88 ctx 스케일링 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
    for v in VARIANTS:
        print(f"{v:5s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
