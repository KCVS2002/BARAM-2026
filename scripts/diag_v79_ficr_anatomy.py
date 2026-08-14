"""#126 진단: CNN sister FICR 이득의 미시 해부 + 잔여 여지 지도 (순수 캐시, 학습 0회).

질문 1: sub_053의 FICR +0.0032는 '어떤 시간들'이 밴드에 새로 들어와서 생겼나
  — 밴드 유입/유출 흐름 (에너지 가중), 월·출력·레짐(d̄)·GBM↔CNN 불일치 슬라이스.
질문 2: 이득이 '점 이동'인가 '분포 재배치'인가 — |Δpred| 분해.
질문 3: 남은 FICR 격차의 성분 — 원자구름 오라클로 '선택 개선 가능' vs '정보 부족' 분리.
질문 4: 근접 실패(6~10%) 에너지 분포 — 소폭 오차 축소의 지렛대가 어디에 있나.

구성은 v78의 base_off / cur75와 동일 (NA 3분위, K=150, g3 불변).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
HOUR_NS = 3_600_000_000_000


def band_of(err):
    return np.select([err <= 0.06, err <= 0.08], [0, 1], default=2)  # 0=4원 1=3원 2=0원


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
            has_cnn = tgt != "kpx_group_3"
            if has_cnn:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cur300 = interp_atoms(qcur, n=2 * K)
                cnn75 = cur300[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                cnn_med = qcur[:, 9]
            else:
                cnn_med = med
            preds = {}
            stats = {}
            for variant in (["base", "cnn"] if has_cnn else ["base"]):
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb]
                    if variant == "cnn":
                        parts.append(cnn75[i])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                atoms = np.array(atoms_l, dtype=object)
                pred = optimize_submission(
                    np.array([a.astype(float) for a in atoms_l]), cap, a_bar)
                preds[variant] = pred
                # 원자구름 통계 (오라클용)
                near = np.array([np.min(np.abs(a - actual[i])) / cap
                                 for i, a in enumerate(atoms_l)])
                frac6 = np.array([np.mean(np.abs(a - actual[i]) / cap <= 0.06)
                                  for i, a in enumerate(atoms_l)])
                stats[variant] = (near, frac6)
            if not has_cnn:
                preds["cnn"] = preds["base"]
                stats["cnn"] = stats["base"]
            month = ((pd.to_datetime(dtm) - pd.Timedelta(hours=1)).month).to_numpy()
            df = pd.DataFrame({
                "fold": fold, "tgt": tgt, "cap": cap,
                "actual": actual, "month": month, "terc": terc,
                "pred_base": preds["base"], "pred_cnn": preds["cnn"],
                "gbm_med": med, "cnn_med": cnn_med,
                "near_cnn": stats["cnn"][0], "frac6_cnn": stats["cnn"][1],
            })
            rows.append(df)
        print(f"fold {fold} 완료 ({time.time()-t0:.0f}s)", flush=True)

    d = pd.concat(rows, ignore_index=True)
    d["valid"] = d.actual >= d.cap * 0.10
    v = d[d.valid].copy()
    v["err_b"] = (v.pred_base - v.actual).abs() / v.cap
    v["err_c"] = (v.pred_cnn - v.actual).abs() / v.cap
    v["band_b"] = band_of(v.err_b)
    v["band_c"] = band_of(v.err_c)
    v["p"] = v.actual / v.cap
    v["dis"] = (v.cnn_med - v.gbm_med).abs() / v.cap
    v["dpred"] = (v.pred_cnn - v.pred_base).abs() / v.cap

    price = {0: 4.0, 1: 3.0, 2: 0.0}

    def ficr(sub, band_col):
        w = sub.actual
        earned = (w * sub[band_col].map(price)).sum()
        return earned / (w * 4.0).sum()

    def eshare(sub, mask):
        return sub.actual[mask].sum() / sub.actual.sum()

    print("\n=== [1] 그룹별 base vs cnn — FICR / 밴드 에너지 점유율 (유효시간) ===")
    for tgt in TARGET_COLS:
        s = v[v.tgt == tgt]
        line = f"{tgt}: FICR {ficr(s,'band_b'):.4f}→{ficr(s,'band_c'):.4f}"
        for b, nm in [(0, "≤6%"), (1, "6-8%"), (2, ">8%")]:
            line += f" | {nm} {eshare(s, s.band_b==b):.3f}→{eshare(s, s.band_c==b):.3f}"
        print(line)

    g12 = v[v.tgt != "kpx_group_3"]
    print("\n=== [2] 밴드 흐름 행렬 (g1/g2, 에너지 비중, base행→cnn열) ===")
    tot = g12.actual.sum()
    for b0 in range(3):
        cells = []
        for b1 in range(3):
            m = (g12.band_b == b0) & (g12.band_c == b1)
            cells.append(f"{g12.actual[m].sum()/tot:.4f}")
        print(f"  base {['≤6','6-8','>8'][b0]:>3}: " + "  ".join(cells))

    def slice_delta(sub, key, labels=None):
        out = []
        for kval, s in sub.groupby(key):
            d6 = eshare(s, s.band_c == 0) - eshare(s, s.band_b == 0)
            dfi = ficr(s, "band_c") - ficr(s, "band_b")
            w = s.actual.sum() / sub.actual.sum()
            out.append((kval, len(s), w, d6, dfi))
        return out

    print("\n=== [3] Δ적중 슬라이스 (g1/g2) — n / 에너지비중 / Δ(≤6%점유) / ΔFICR ===")
    print("-- 월 --")
    for kv, n, w, d6, dfi in slice_delta(g12, "month"):
        print(f"  {kv:>2}월: n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")
    print("-- 출력 구간 (actual/cap) --")
    g12["pbin"] = pd.cut(g12.p, [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
    for kv, n, w, d6, dfi in slice_delta(g12, "pbin"):
        print(f"  {str(kv):>12}: n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")
    print("-- AnEn 거리 3분위 (0=흔한 레짐) --")
    for kv, n, w, d6, dfi in slice_delta(g12, "terc"):
        print(f"  terc{kv}: n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")
    print("-- GBM↔CNN q50 불일치 5분위 --")
    g12["disq"] = pd.qcut(g12.dis, 5, labels=False, duplicates="drop")
    for kv, n, w, d6, dfi in slice_delta(g12, "disq"):
        med_dis = g12.dis[g12.disq == kv].median()
        print(f"  q{kv} (중앙 {med_dis:.3f}C): n={n:4d} w={w:.3f} Δ6={d6:+.4f} ΔFICR={dfi:+.4f}")

    print("\n=== [4] 점 이동 크기 |pred_cnn - pred_base|/C (g1/g2) ===")
    qs = g12.dpred.quantile([0.5, 0.75, 0.9, 0.99]).round(4).to_dict()
    print(f"  분위: {qs} | 평균 {g12.dpred.mean():.4f}")
    entered = g12[(g12.band_b == 2) & (g12.band_c < 2)]
    exited = g12[(g12.band_b < 2) & (g12.band_c == 2)]
    print(f"  신규 진입 {len(entered)}h (이동 중앙 {entered.dpred.median():.4f}C, 사전오차 중앙 {entered.err_b.median():.4f})")
    print(f"  이탈    {len(exited)}h (이동 중앙 {exited.dpred.median():.4f}C)")

    print("\n=== [5] 오라클 분해 (cnn 구성) — 실패 에너지의 성분 ===")
    for tgt in TARGET_COLS:
        s = v[v.tgt == tgt]
        miss = s[s.band_c == 2]
        me = miss.actual.sum() / s.actual.sum()
        sel = miss[miss.near_cnn <= 0.06]
        info8 = miss[miss.near_cnn > 0.08]
        oracle_band = band_of(s.near_cnn)
        w = s.actual
        of = (w * pd.Series(oracle_band, index=s.index).map(price)).sum() / (w * 4).sum()
        print(f"{tgt}: 실패에너지 {me:.3f} | 그중 구름내 6%점 존재(선택) {sel.actual.sum()/max(miss.actual.sum(),1):.3f}"
              f" / 구름에 8%점도 없음(정보) {info8.actual.sum()/max(miss.actual.sum(),1):.3f}"
              f" | 현 FICR {ficr(s,'band_c'):.4f} vs 최근접원자 오라클 {of:.4f}")

    print("\n=== [6] 근접 실패 지렛대 — 오차 구간별 에너지 점유 (cnn 구성) ===")
    for tgt in TARGET_COLS:
        s = v[v.tgt == tgt]
        bins = [(0, .06), (.06, .08), (.08, .10), (.10, .14), (.14, 1.0)]
        parts = []
        for lo, hi in bins:
            m = (s.err_c > lo) & (s.err_c <= hi)
            parts.append(f"{lo*100:.0f}-{hi*100:.0f}%:{eshare(s,m):.3f}")
        print(f"{tgt}: " + " ".join(parts))
    print("-- g1/g2 합산, 8-10% 구간의 월·출력 분포 (어디를 조금만 줄이면 3원이 되나) --")
    nm = g12[(g12.err_c > 0.08) & (g12.err_c <= 0.10)]
    print("  월별 에너지: " + " ".join(f"{m}월:{nm.actual[nm.month==m].sum()/max(nm.actual.sum(),1):.2f}"
                                    for m in sorted(nm.month.unique())))
    print("  출력별: " + " ".join(f"{str(b)}:{nm.actual[nm.pbin==b].sum()/max(nm.actual.sum(),1):.2f}"
                               for b in nm.pbin.cat.categories))

    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
