"""#127 v80: AnEn 공간장 유사도 재검색 — 수집된 28×28 u/v 장으로 이웃 재순위.

가설: 점 피처 12개 kNN이 놓치는 '국지 흐름 패턴'의 유사성을 장 거리로 보강하면
아날로그 이웃의 실측 분포가 더 정확해진다 (#126: 희귀 레짐 FICR 결핍이 표적,
같은 데이터의 정보 가치는 CNN sister로 실증됨). 상수 이식 없는 구조 변경
(rank-sum 결합은 스케일 자유).

변형 (전 그룹, 학습 0회 — kNN 재실행 + 장 거리 + 결정층만):
  base : 공식 cur75 (캐시 anen200)
  rs4  : 피처 kNN 풀 400 → rank(피처)+rank(장) 합산 → 상위 200
  ff4  : 피처 kNN 풀 400 → 장 거리 단독 정렬 → 상위 200
  rs8  : 풀 800 rank-sum → 상위 200
d̄/NA 3분위는 캐시 dist(피처 거리) 고정 — 이웃 집합 변화만 분리 측정.
진단: 기존 top-200과의 평균 교집합 비율. 판정 #91 (바 0.0016), 판정은 사용자.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from scripts.train_v75c_cnn_scaled import load_grid_hours
from src.decision import interp_atoms, optimize_submission
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
VARIANTS = ["base", "rs4", "ff4", "rs8"]
POOL = {"rs4": 400, "ff4": 400, "rs8": 800}


def rerank(variant, aidx_pool, fdist_pool):
    """풀 내 재순위 → 상위 200 인덱스 (원 라이브러리 인덱스)."""
    P = aidx_pool.shape[1]
    if variant == "ff4":
        order = np.argsort(fdist_pool, axis=1, kind="stable")
    else:  # rank-sum
        frank = np.argsort(np.argsort(fdist_pool, axis=1, kind="stable"),
                           axis=1, kind="stable")
        score = np.arange(P)[None, :] + frank
        order = np.argsort(score, axis=1, kind="stable")
    top = order[:, :200]
    return np.take_along_axis(aidx_pool, top, axis=1)


def main() -> None:
    t0 = time.time()
    df, _, _, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")

    gdtm, gfields = load_grid_hours()
    mu = gfields.mean(axis=(0, 2, 3), keepdims=True)
    sd = gfields.std(axis=(0, 2, 3), keepdims=True)
    F = ((gfields - mu) / sd).reshape(len(gfields), -1).astype(np.float32)
    fpos = {int(t): i for i, t in enumerate(gdtm)}
    print(f"grid {len(gdtm)}시간, 피처 {F.shape[1]}차원 ({time.time()-t0:.0f}s)", flush=True)

    def field_dists(val_keys, aidx_pool, lib_keys):
        """(nval, P) 장 거리². 장 없는 이웃은 풀 중앙값, 장 없는 val행은 NaN(재순위 생략)."""
        nval, P = aidx_pool.shape
        out = np.empty((nval, P), dtype=np.float32)
        lib_pos = np.array([fpos.get(int(k), -1) for k in lib_keys])
        val_pos = np.array([fpos.get(int(k), -1) for k in val_keys])
        for s in range(0, nval, 64):
            e = min(s + 64, nval)
            pool_pos = lib_pos[aidx_pool[s:e]]          # (c, P)
            vp = val_pos[s:e]
            ok_v = vp >= 0
            pool_ok = pool_pos >= 0
            vec = F[np.where(ok_v, vp, 0)]              # (c, D)
            pv = F[np.clip(pool_pos, 0, None)]          # (c, P, D)
            d2 = ((pv - vec[:, None, :]) ** 2).sum(axis=2)
            med = np.median(np.where(pool_ok, d2, np.nan), axis=1)
            d2 = np.where(pool_ok, d2, med[:, None])
            d2[~ok_v] = np.nan
            out[s:e] = d2
        return out

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    gsc = {v: {t: [] for t in TARGET_COLS} for v in VARIANTS}
    overlaps = []

    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        fold = f"{va_start:%Y-%m}"
        tr = df[df.forecast_kst_dtm < va_start]
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen_c, dist_c = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            dtm = z["dtm"]
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist_c[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            has_cnn = tgt != "kpx_group_3"
            if has_cnn:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]

            # AnEn 재검색 (dump_oof_atoms_v2와 동일 라이브러리 구성)
            tr_ok = tr.loc[tr[tgt].notna()].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            Xlib = ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy()
            va_rows = df.set_index("key").loc[dtm]
            Xva = ((va_rows[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy()
            Pmax = min(800, len(Xlib))
            knn = NearestNeighbors(n_neighbors=Pmax).fit(Xlib)
            dist_p, aidx_p = knn.kneighbors(Xva)
            lib_vals = tr_ok[tgt].to_numpy()
            lib_keys = tr_ok.forecast_kst_dtm.astype("datetime64[ns]").astype("int64").to_numpy()

            # 복제 검증: 피처 top-200 = 캐시 anen과 일치해야 함
            rep = np.sort(lib_vals[aidx_p[:, :200]], axis=1)
            rep_ok = np.allclose(rep, anen_c, atol=1e-6)
            if not rep_ok:
                print(f"  !! 복제 불일치 {fold} {tgt}: maxdiff {np.abs(rep-anen_c).max():.3f}", flush=True)

            fd = field_dists(dtm, aidx_p, lib_keys)
            anen_v = {"base": anen_c}
            for v in VARIANTS[1:]:
                P = min(POOL[v], Pmax)
                idx200 = rerank(v, aidx_p[:, :P], fd[:, :P])
                # 장 없는 val행은 원 top-200 유지
                bad = np.isnan(fd[:, 0])
                idx200[bad] = aidx_p[bad, :200]
                anen_v[v] = np.sort(lib_vals[idx200], axis=1)
                if v == "rs4":
                    ov = np.mean([len(np.intersect1d(idx200[i], aidx_p[i, :200])) / 200
                                  for i in range(0, n_row, 7)])
                    overlaps.append(ov)

            for v in VARIANTS:
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen_v[v][i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + ([cnn75[i]] if has_cnn else [])
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
              + f" | rs4 교집합 {np.mean(overlaps):.2f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v80 AnEn 공간장 재검색 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
    print("-- 그룹별 fold평균 --")
    for v in VARIANTS:
        print(f"{v:5s}: " + " ".join(f"{t[-1]}={np.nanmean(gsc[v][t]):.4f}" for t in TARGET_COLS))
    for v in VARIANTS:
        print(f"{v:5s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
