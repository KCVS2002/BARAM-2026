"""#130 v83: 결합 프로브 CV 확증 — 여름 CNN 게이트 + g2 임베딩 검색 (순수 캐시).

#126(여름 5~8월 CNN 손해)·#128(g2만 임베딩 검색 이득) 두 소액 신호의 결합.
변형:
  base   : 공식 cur75
  gate   : g1/g2 CNN75를 5~8월만 제외
  g2rs   : gate + g2 AnEn을 rank-sum 임베딩 재검색(rse4)으로 교체
  g2ek   : gate + g2 AnEn을 전역 임베딩 kNN(embk)으로 교체
g2 fold별 일관성(rse4 vs embk 선택 재료)까지 출력. 판정은 사용자.
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
VARIANTS = ["base", "gate", "g2rs", "g2ek"]
SUMMER = {5, 6, 7, 8}
HOUR = pd.Timedelta(hours=1)


def build_pred(qp_gbm, anen, dist, cap, a_bar, cnn75, cnn_mask):
    n_row = len(qp_gbm)
    gbm300 = interp_atoms(qp_gbm, n=2 * K)
    med = np.median(gbm300, axis=1)
    d_bar = dist[:, :K].mean(axis=1)
    terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
    atoms_l = []
    for i in range(n_row):
        na = NA[terc[i]]
        ng = 2 * K - na
        an = np.interp(np.linspace(0, 199, na), np.arange(200), anen[i])
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
        parts = [an, gb]
        if cnn75 is not None and cnn_mask[i]:
            parts.append(cnn75[i])
        atoms_l.append(np.sort(np.concatenate(parts)))
    return optimize_submission(np.array(atoms_l), cap, a_bar)


def main() -> None:
    t0 = time.time()
    df, _, _, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    g_dtm, _ = load_grid_hours()
    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    g2fold = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}

    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        fold = f"{va_start:%Y-%m}"
        Z = np.load(CACHE / f"{fold}_emb.npz")["z"]
        gidx = {int(t): i for i, t in enumerate(g_dtm)}
        tr = df[df.forecast_kst_dtm < va_start]
        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen_c, dist_c = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            dtm = z["dtm"]
            month = (pd.to_datetime(dtm) - HOUR).month.to_numpy()
            summer = np.isin(month, list(SUMMER))
            has_cnn = tgt != "kpx_group_3"
            cnn75 = None
            if has_cnn:
                qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]

            anen_emb = {}
            if tgt == "kpx_group_2":
                tr_ok = tr.loc[tr[tgt].notna()].dropna(subset=ANEN_FEATS)
                mu_a = tr_ok[ANEN_FEATS].mean()
                sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
                Xlib = ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy()
                Xva = ((df.set_index("key").loc[dtm][ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy()
                lib_vals = tr_ok[tgt].to_numpy()
                lib_keys = tr_ok.forecast_kst_dtm.astype("datetime64[ns]").astype("int64").to_numpy()
                lib_pos = np.array([gidx.get(int(k), -1) for k in lib_keys])
                val_pos = np.array([gidx.get(int(k), -1) for k in dtm])
                lib_has = lib_pos >= 0
                Zmu = Z[lib_pos[lib_has]].mean(0)
                Zsd = Z[lib_pos[lib_has]].std(0) + 1e-8
                Zn = (Z - Zmu) / Zsd
                P = min(400, len(Xlib))
                knn = NearestNeighbors(n_neighbors=P).fit(Xlib)
                _, aidx_p = knn.kneighbors(Xva)
                ed = np.full(aidx_p.shape, np.nan, dtype=np.float32)
                for s in range(0, len(dtm), 256):
                    e = min(s + 256, len(dtm))
                    vp = val_pos[s:e]
                    ok_v = vp >= 0
                    pool_pos = lib_pos[aidx_p[s:e]]
                    pool_ok = pool_pos >= 0
                    vec = Zn[np.where(ok_v, vp, 0)]
                    pv = Zn[np.clip(pool_pos, 0, None)]
                    d2 = ((pv - vec[:, None, :]) ** 2).sum(axis=2)
                    medv = np.nanmedian(np.where(pool_ok, d2, np.nan), axis=1)
                    d2 = np.where(pool_ok, d2, medv[:, None])
                    d2[~ok_v] = np.nan
                    ed[s:e] = d2
                erank = np.argsort(np.argsort(ed, axis=1, kind="stable"), axis=1, kind="stable")
                order = np.argsort(np.arange(P)[None, :] + erank, axis=1, kind="stable")
                idx200 = np.take_along_axis(aidx_p, order[:, :200], axis=1)
                bad = np.isnan(ed[:, 0])
                idx200[bad] = aidx_p[bad, :200]
                anen_emb["g2rs"] = np.sort(lib_vals[idx200], axis=1)
                knn_e = NearestNeighbors(n_neighbors=200).fit(Zn[lib_pos[lib_has]])
                lib_map = np.where(lib_has)[0]
                _, eidx_raw = knn_e.kneighbors(Zn[np.clip(val_pos, 0, None)])
                idx_e = lib_map[eidx_raw]
                idx_e[val_pos < 0] = aidx_p[val_pos < 0, :200]
                anen_emb["g2ek"] = np.sort(lib_vals[idx_e], axis=1)

            cache_g3 = None
            for v in VARIANTS:
                if not has_cnn:
                    if cache_g3 is None:
                        cache_g3 = build_pred(qp_gbm, anen_c, dist_c, cap, a_bar, None, None)
                    pred = cache_g3
                else:
                    mask = np.ones(len(dtm), bool) if v == "base" else ~summer
                    anen_use = anen_c
                    if tgt == "kpx_group_2" and v in anen_emb:
                        anen_use = anen_emb[v]
                    pred = build_pred(qp_gbm, anen_use, dist_c, cap, a_bar, cnn75, mask)
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                dc[v].append((nm, fi))
                if tgt == "kpx_group_2":
                    g2fold[v][fold] = s_
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v83 결합 프로브 CV — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:4s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")
    print("-- g2 fold별 (rse4/embk 선택 재료) --")
    for v in VARIANTS:
        print(f"{v:4s}: " + " ".join(f"{f}:{g2fold[v][f]:.4f}" for f in sorted(g2fold[v])))
    for v in VARIANTS:
        print(f"{v:4s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
