"""#141 진단: TabPFN 이득 미시 해부 + sub_055 지형의 잔여 결핍 재지도화 (순수 캐시).

#126(CNN 해부)의 TabPFN판. base = sub_053 구성(cur75), new = sub_055 구성(+tpf75).
질문: ①TabPFN FICR +0.0059(CV)/+0.0037(LB)는 어디서 왔나 (계절·출력·레짐·그룹)
②CNN 때 제로였던 g2가 이번엔 얻었나 ③남은 실패 에너지의 새 지도 (다음 표적).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
G12 = ["kpx_group_1", "kpx_group_2"]


def band_of(err):
    return np.select([err <= 0.06, err <= 0.08], [0, 1], default=2)


def main() -> None:
    t0 = time.time()
    rows = []
    for m0 in range(1, 12, 2):
        fold = f"2024-{m0:02d}"
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
                qtp = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tp75 = interp_atoms(qtp, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tpf_med = qtp[:, 9]
            preds = {}
            for variant in (["base", "tpf"] if is_g12 else ["base"]):
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
                        if variant == "tpf":
                            parts.append(tp75[i])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                preds[variant] = optimize_submission(np.array(atoms_l), cap, a_bar)
            if not is_g12:
                preds["tpf"] = preds["base"]
                tpf_med = med
            month = ((pd.to_datetime(dtm) - pd.Timedelta(hours=1)).month).to_numpy()
            rows.append(pd.DataFrame({
                "fold": fold, "tgt": tgt, "cap": cap, "actual": actual,
                "month": month, "terc": terc,
                "pred_base": preds["base"], "pred_tpf": preds["tpf"],
                "gbm_med": med, "tpf_med": tpf_med,
            }))
        print(f"fold {fold} 완료 ({time.time()-t0:.0f}s)", flush=True)

    d = pd.concat(rows, ignore_index=True)
    v = d[d.actual >= d.cap * 0.10].copy()
    v["err_b"] = (v.pred_base - v.actual).abs() / v.cap
    v["err_t"] = (v.pred_tpf - v.actual).abs() / v.cap
    v["band_b"] = band_of(v.err_b)
    v["band_t"] = band_of(v.err_t)
    v["p"] = v.actual / v.cap
    v["dis"] = (v.tpf_med - v.gbm_med).abs() / v.cap

    price = {0: 4.0, 1: 3.0, 2: 0.0}

    def ficr(sub, band_col):
        w = sub.actual
        return (w * sub[band_col].map(price)).sum() / (w * 4.0).sum()

    def eshare(sub, mask):
        return sub.actual[mask].sum() / sub.actual.sum()

    print("\n=== [1] 그룹별 base(sub_053 구성) vs tpf(sub_055 구성) — FICR / 밴드 점유 ===")
    for tgt in TARGET_COLS:
        s = v[v.tgt == tgt]
        line = f"{tgt}: FICR {ficr(s,'band_b'):.4f}→{ficr(s,'band_t'):.4f}"
        for b, nm in [(0, "≤6%"), (1, "6-8%"), (2, ">8%")]:
            line += f" | {nm} {eshare(s, s.band_b==b):.3f}→{eshare(s, s.band_t==b):.3f}"
        print(line)

    g12 = v[v.tgt != "kpx_group_3"].copy()

    def slice_delta(sub, key):
        out = []
        for kval, s in sub.groupby(key, observed=True):
            d6 = eshare(s, s.band_t == 0) - eshare(s, s.band_b == 0)
            dfi = ficr(s, "band_t") - ficr(s, "band_b")
            w = s.actual.sum() / sub.actual.sum()
            out.append((kval, len(s), w, d6, dfi))
        return out

    print("\n=== [2] Δ적중 슬라이스 (g1/g2) — n / 에너지비중 / Δ(≤6%) / ΔFICR ===")
    print("-- 월 --")
    for kv, n, w, d6, dfi in slice_delta(g12, "month"):
        print(f"  {kv:>2}월: n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")
    print("-- 출력 구간 --")
    g12["pbin"] = pd.cut(g12.p, [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
    for kv, n, w, d6, dfi in slice_delta(g12, "pbin"):
        print(f"  {str(kv):>12}: n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")
    print("-- AnEn 거리 3분위 --")
    for kv, n, w, d6, dfi in slice_delta(g12, "terc"):
        print(f"  terc{kv}: n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")
    print("-- GBM↔TabPFN q50 불일치 5분위 --")
    g12["disq"] = pd.qcut(g12.dis, 5, labels=False, duplicates="drop")
    for kv, n, w, d6, dfi in slice_delta(g12, "disq"):
        md = g12.dis[g12.disq == kv].median()
        print(f"  q{kv} (중앙 {md:.3f}C): n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")

    print("\n=== [3] 잔여 실패 지도 (tpf 구성) — 오차 구간별 에너지 점유 ===")
    for tgt in TARGET_COLS:
        s = v[v.tgt == tgt]
        parts = []
        for lo, hi in [(0, .06), (.06, .08), (.08, .10), (.10, .14), (.14, 1.0)]:
            m = (s.err_t > lo) & (s.err_t <= hi)
            parts.append(f"{lo*100:.0f}-{hi*100:.0f}%:{eshare(s,m):.3f}")
        print(f"{tgt}: " + " ".join(parts))
    print("-- g1/g2 잔여 >8% 실패 에너지의 월·출력 분포 --")
    fail = g12[g12.err_t > 0.08]
    print("  월별: " + " ".join(f"{m}월:{fail.actual[fail.month==m].sum()/fail.actual.sum():.2f}"
                              for m in sorted(fail.month.unique())))
    print("  출력별: " + " ".join(f"{str(b)}:{fail.actual[fail.pbin==b].sum()/fail.actual.sum():.2f}"
                               for b in fail.pbin.cat.categories))
    print("  레짐별: " + " ".join(f"terc{t}:{fail.actual[fail.terc==t].sum()/fail.actual.sum():.2f}"
                               for t in (0, 1, 2)))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
