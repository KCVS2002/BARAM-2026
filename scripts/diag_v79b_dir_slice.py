"""#126b: CNN FICR 이득의 풍향 슬라이스 — #51a 'g1 서풍 편향' 연결 확인.

diag_v79와 동일 재구성이지만 g1/g2만, 풍향 8섹터(ldaps_ws50max_dsin/dcos)로
ΔFICR·Δ편향(잔차 평균)을 분해. 순수 캐시.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from src.decision import interp_atoms, optimize_submission
from src.metric import TARGET_COLS

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}


def main() -> None:
    t0 = time.time()
    df, _, _, _ = load_base_frame()
    df = df[["forecast_kst_dtm", "ldaps_ws50max_dsin", "ldaps_ws50max_dcos"]].copy()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    theta = np.degrees(np.arctan2(df.ldaps_ws50max_dsin, df.ldaps_ws50max_dcos)) % 360
    df["sector"] = (theta // 45).astype(int)  # 0=N 2=E 4=S 6=W

    rows = []
    for m0 in range(1, 12, 2):
        fold = f"2024-{m0:02d}"
        for tgt in ["kpx_group_1", "kpx_group_2"]:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            dtm = z["dtm"]
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
            cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            preds = {}
            for variant in ["base", "cnn"]:
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + ([cnn75[i]] if variant == "cnn" else [])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                preds[variant] = optimize_submission(np.array(atoms_l), cap, a_bar)
            rows.append(pd.DataFrame({
                "key": dtm.astype("int64"), "tgt": tgt, "cap": cap, "actual": actual,
                "pred_base": preds["base"], "pred_cnn": preds["cnn"]}))
        print(f"fold {fold} 완료 ({time.time()-t0:.0f}s)", flush=True)

    d = pd.concat(rows, ignore_index=True).merge(
        df[["key", "sector"]], on="key", how="left")
    v = d[d.actual >= d.cap * 0.10].copy()
    v["err_b"] = (v.pred_base - v.actual).abs() / v.cap
    v["err_c"] = (v.pred_cnn - v.actual).abs() / v.cap
    v["bias_b"] = (v.pred_base - v.actual) / v.cap
    v["bias_c"] = (v.pred_cnn - v.actual) / v.cap

    def ficr(sub, err_col):
        pr = np.select([sub[err_col] <= 0.06, sub[err_col] <= 0.08], [4.0, 3.0], 0.0)
        return (sub.actual * pr).sum() / (sub.actual * 4.0).sum()

    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    for tgt in ["kpx_group_1", "kpx_group_2"]:
        print(f"\n=== {tgt} 풍향 섹터별 (n / 에너지비중 / 편향b→c / ΔFICR) ===")
        s = v[v.tgt == tgt]
        for sec in range(8):
            g = s[s.sector == sec]
            if len(g) < 30:
                continue
            w = g.actual.sum() / s.actual.sum()
            print(f"  {names[sec]:>2}: n={len(g):4d} w={w:.3f} "
                  f"편향 {g.bias_b.mean():+.4f}→{g.bias_c.mean():+.4f} "
                  f"|오차| {g.err_b.mean():.4f}→{g.err_c.mean():.4f} "
                  f"ΔFICR={ficr(g,'err_c')-ficr(g,'err_b'):+.4f}")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
