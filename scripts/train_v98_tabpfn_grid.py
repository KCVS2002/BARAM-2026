"""#147 v98: TabPFN 피처에 GFS 광역 격자 121컬럼 직접 추가 — v97(CNN 경로)의 표 경로 대조.

같은 데이터(11×11 풍속 지도 + 도메인 u/v)의 두 소비 문법 대결:
  v97: 공간 구조를 아는 CNN 브랜치 / v98: attention 표 학습기에 날 컬럼 255개.
GBM '피처 추가 = 희석' 법칙이 TabPFN(attention)에도 적용되는지 자체가 미검증.

변형 (g1/g2, g3 불변, 공식 sub_055 구성 위):
  base  : tpf75 (표준 134피처)
  grep  : tpf75를 격자 확장(255피처) 버전으로 교체
  gboth : 표준 75 + 격자 확장 50 (다양성 문법)
프로토콜: ctx 6000·n_est 2·seed 42 동일. 캐시: {fold}_{tgt}_tpfqp_grid.npz.
진단: 확장판 pinball·표준판과의 오차 상관. 판정 #91, 판정은 사용자.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from scripts.train_v97_dualscale_cnn import load_gfs_maps
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
QS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CTX = 6000
VARIANTS = ["base", "grep", "gboth"]
G12 = ["kpx_group_1", "kpx_group_2"]


def pinball(y, qp, qs):
    tot = 0.0
    for j, q in enumerate(qs):
        d = y - qp[:, j]
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / len(qs)


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    f_dtm, f_maps, f_uv = load_gfs_maps()
    gcols = [f"gws_{i}" for i in range(121)]
    gdf = pd.DataFrame(f_maps.reshape(len(f_dtm), 121), columns=gcols)
    gdf["gu_mean"] = f_uv[:, 0]
    gdf["gv_mean"] = f_uv[:, 1]
    gdf["key"] = f_dtm
    df = df.merge(gdf, on="key", how="left")
    xcols = cols + gcols + ["gu_mean", "gv_mean"]
    print(f"frame ready — 확장 피처 {len(xcols)}개 ({time.time()-t0:.0f}s)", flush=True)

    from tabpfn import TabPFNRegressor
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
            dtm = z["dtm"]
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            is_g12 = tgt in G12
            if is_g12:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                out_f = CACHE / f"{fold}_{tgt}_tpfqp_grid.npz"
                if not out_f.exists():
                    va_start = pd.Timestamp(2024, m0, 1, 1)
                    trm = (df.forecast_kst_dtm < va_start) & df[tgt].notna()
                    tr = df[trm]
                    w = w_q3[tgt].to_numpy()[trm.to_numpy()]
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(tr), size=min(CTX, len(tr)), replace=False, p=w / w.sum())
                    Xtr = tr[xcols].to_numpy(dtype=np.float32)[idx]
                    ytr = (tr[tgt] / cap).to_numpy(dtype=np.float32)[idx]
                    Xva = df.set_index("key").loc[dtm][xcols].to_numpy(dtype=np.float32)
                    reg = TabPFNRegressor(n_estimators=2, ignore_pretraining_limits=True,
                                          random_state=42, device="auto", memory_saving_mode=True)
                    reg.fit(Xtr, ytr)
                    qp_t = np.empty((n_row, len(QS)), dtype=np.float64)
                    for s in range(0, n_row, 128):
                        r = reg.predict(Xva[s:s + 128], output_type="quantiles", quantiles=QS)
                        qp_t[s:s + 128] = np.column_stack(r) if isinstance(r, list) else r
                    qp_t = np.clip(np.sort(qp_t, axis=1), 0, 1) * cap
                    qp_t = np.sort(smooth_quantiles_by_day(qp_t, pd.Series(pd.to_datetime(dtm))), axis=1)
                    np.savez_compressed(out_f, qp=qp_t)
                    print(f"[{fold} {tgt}] tpf grid 캐시 ({time.time()-t0:.0f}s)", flush=True)
                q_std = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
                q_grd = np.load(out_f)["qp"]
                pbg = pinball(actual, qp_gbm, QS)
                cc = np.corrcoef(q_grd[:, 9] - actual, q_std[:, 9] - actual)[0, 1]
                print(f"  {fold} {tgt[-1]}: pinball비 표준 {pinball(actual,q_std,QS)/pbg:.3f} / "
                      f"격자 {pinball(actual,q_grd,QS)/pbg:.3f} | 상관 {cc:.3f}", flush=True)
                tp75 = interp_atoms(q_std, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tg75 = interp_atoms(q_grd, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tg50 = interp_atoms(q_grd, n=2 * K)[:, np.linspace(0, 2 * K - 1, 50).astype(int)]
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
                            parts.append(tp75[i])
                        elif v == "grep":
                            parts.append(tg75[i])
                        else:
                            parts += [tp75[i], tg50[i]]
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

    print("\n=== v98 TabPFN 격자 피처 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
