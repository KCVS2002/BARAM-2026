"""#144 v95: 관점 분할 TabPFN sister — LDAPS 뷰 / GFS+IFS 뷰 (다른 NWP를 보는 눈).

동기: #141 '불일치 최상위 5분위 ΔFICR +0.0302' — 관점이 다를수록 풀링이 번다.
리스크: 반쪽 피처의 품질 저하 (#22 GBM 소스별 sister 희석 전례). TabPFN의 품질
여유(전체피처 비율 ~0.96)가 버텨줄지가 관건. 교차 파생(wpd·shear·air_density)은
양쪽에서 제외 (뷰 순도 유지). 시간 피처는 공통 포함.

변형 (g1/g2, g3 불변):
  base  : 공식 (cur75 + tpf75 전체피처)
  vadd  : base + vL 38원자 + vG 37원자 (뷰 추가, 총 tpf 150)
  vrepl : cur75 + vL 38 + vG 37 (전체피처 tpf 제거, 뷰만)
진단: 뷰별 pinball 비율 / 전체 TabPFN·GBM과의 오차 상관 (다양성 크기).
캐시: {fold}_{tgt}_tpfqp_v{L,G}.npz. 판정 #91, 판정은 사용자.
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
VARIANTS = ["base", "vadd", "vrepl"]
G12 = ["kpx_group_1", "kpx_group_2"]
TIME_FEATS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "lead_h"]


def view_cols(cols, view):
    if view == "L":
        src = [c for c in cols if c.startswith("ldaps")]
    else:
        src = [c for c in cols if c.startswith(("gfs", "ifs", "cons3"))]
    return src + [c for c in TIME_FEATS if c in cols]


def pinball(y, qp, qs):
    tot = 0.0
    for j, q in enumerate(qs):
        d = y - qp[:, j]
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / len(qs)


def gen_tpf_qp(fold, tgt, view, df, w_q3, cols, t0):
    from tabpfn import TabPFNRegressor
    cap = CAPACITY_KWH[tgt]
    z = np.load(CACHE / f"{fold}_{tgt}.npz")
    dtm = z["dtm"]
    vcols = view_cols(cols, view)
    va_start = pd.Timestamp(int(fold[:4]), int(fold[5:7]), 1, 1)
    trm = (df.forecast_kst_dtm < va_start) & df[tgt].notna()
    tr = df[trm]
    w = w_q3[tgt].to_numpy()[trm.to_numpy()]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(tr), size=min(CTX, len(tr)), replace=False, p=w / w.sum())
    Xtr = tr[vcols].to_numpy(dtype=np.float32)[idx]
    ytr = (tr[tgt] / cap).to_numpy(dtype=np.float32)[idx]
    Xva = df.set_index("key").loc[dtm][vcols].to_numpy(dtype=np.float32)
    reg = TabPFNRegressor(n_estimators=2, ignore_pretraining_limits=True,
                          random_state=42, device="auto", memory_saving_mode=True)
    reg.fit(Xtr, ytr)
    qp_t = np.empty((len(Xva), len(QS)), dtype=np.float64)
    for s in range(0, len(Xva), 128):
        r = reg.predict(Xva[s:s + 128], output_type="quantiles", quantiles=QS)
        qp_t[s:s + 128] = np.column_stack(r) if isinstance(r, list) else r
    qp_t = np.clip(np.sort(qp_t, axis=1), 0, 1) * cap
    qp_t = np.sort(smooth_quantiles_by_day(qp_t, pd.Series(pd.to_datetime(dtm))), axis=1)
    np.savez_compressed(CACHE / f"{fold}_{tgt}_tpfqp_v{view}.npz", qp=qp_t)
    print(f"[{fold} {tgt}] tpf view{view} ({len(vcols)}피처) 캐시 ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    print(f"frame ready — viewL {len(view_cols(cols,'L'))} / viewG {len(view_cols(cols,'G'))}피처 "
          f"({time.time()-t0:.0f}s)", flush=True)

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
                for view in ("L", "G"):
                    if not (CACHE / f"{fold}_{tgt}_tpfqp_v{view}.npz").exists():
                        gen_tpf_qp(fold, tgt, view, df, w_q3, cols, t0)
                qf = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
                qL = np.load(CACHE / f"{fold}_{tgt}_tpfqp_vL.npz")["qp"]
                qG = np.load(CACHE / f"{fold}_{tgt}_tpfqp_vG.npz")["qp"]
                pbg = pinball(actual, qp_gbm, QS)
                cLG = np.corrcoef(qL[:, 9] - actual, qG[:, 9] - actual)[0, 1]
                cLf = np.corrcoef(qL[:, 9] - actual, qf[:, 9] - actual)[0, 1]
                print(f"  {fold} {tgt[-1]}: pinball비 full {pinball(actual,qf,QS)/pbg:.3f} / "
                      f"L {pinball(actual,qL,QS)/pbg:.3f} / G {pinball(actual,qG,QS)/pbg:.3f} | "
                      f"상관 L↔G {cLG:.3f} L↔full {cLf:.3f}", flush=True)
                tpf75 = interp_atoms(qf, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                vL38 = interp_atoms(qL, n=2 * K)[:, np.linspace(0, 2 * K - 1, 38).astype(int)]
                vG37 = interp_atoms(qG, n=2 * K)[:, np.linspace(0, 2 * K - 1, 37).astype(int)]
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
                            parts.append(tpf75[i])
                        elif v == "vadd":
                            parts += [tpf75[i], vL38[i], vG37[i]]
                        else:
                            parts += [vL38[i], vG37[i]]
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

    print("\n=== v95 관점 분할 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
