"""#128 v81: AnEn 검색을 CNN 임베딩 계량으로 — v80(비지도 장 거리 실패)의 정밀 우회.

가설: v80의 사망 원인은 '비지도 유클리드가 발전량 무관 픽셀 분산에 지배됨'.
fold별 SisterCNN(v75d 프로토콜 그대로, q3 가중 pinball 지도학습)의 conv GAP
64차원 임베딩은 '발전량에 중요한 공간 구조'만 남긴 표현 → 이 공간의 거리로
아날로그를 찾으면 학습된 관련성 계량이 된다. 상수 이식 없음 (구조 변경).

변형 (전 그룹):
  base  : 공식 cur75 (캐시 anen200)
  rse4  : 피처 kNN 풀 400 → rank(피처)+rank(임베딩) 합산 → 상위 200
  ffe4  : 풀 400 → 임베딩 거리 단독 정렬 → 상위 200
  embk  : 임베딩 공간 전역 kNN top-200 (피처 무시, 강한 버전 진단)
d̄/NA 3분위는 캐시 dist 고정 (v80과 동일 — 이웃 집합 변화만 분리).

캐시: {fold}_emb.npz (전 그리드 시간 z 35040×64) — 존재 시 학습 스킵.
판정 #91 (바 0.0016), 판정은 사용자. v80 대비 부호 반전 여부가 1차 관전점.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from scripts.train_v75c_cnn_scaled import SisterCNN, load_grid_hours, pinball_loss
from src.decision import interp_atoms, optimize_submission
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
VARIANTS = ["base", "rse4", "ffe4", "embk"]
torch.manual_seed(42)
np.random.seed(42)


def train_fold_embedding(fold, cut_ns, df, w_q3, g_dtm, g_field, gidx, t0):
    """v75d 프로토콜 그대로 CNN 학습 → 전 그리드 시간 임베딩 (35040, 64)."""
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)
    Xs, ys, ws, gs, ds_ = [], [], [], [], []
    for gi, tgt in enumerate(TARGET_COLS):
        cap = CAPACITY_KWH[tgt]
        m = (lab_dtm < cut_ns) & df[tgt].notna().to_numpy()
        keep = [r for r in np.where(m)[0] if lab_dtm[r] in gidx]
        Xs.append(np.stack([g_field[gidx[lab_dtm[r]]] for r in keep]))
        ys.append((df[tgt].to_numpy()[keep] / cap).astype(np.float32))
        ws.append(w_q3[tgt].to_numpy()[keep].astype(np.float32))
        gs.append(np.full(len(keep), gi, dtype=np.int64))
        ds_.append(lab_dtm[keep])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    g = np.concatenate(gs)
    dt = np.concatenate(ds_)
    mu = X.mean((0, 2, 3), keepdims=True)
    sd = X.std((0, 2, 3), keepdims=True)
    X = (X - mu) / sd
    w = w / w.mean()
    thr = np.quantile(dt, 0.88)
    va_i = np.where(dt >= thr)[0]
    tr_i = np.where(dt < thr)[0]
    print(f"[{fold}] 학습 {len(tr_i)} / val {len(va_i)} ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(42)
    model = SisterCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, yt, wt, gt = map(torch.tensor, (X, y, w, g))
    best, best_state, patience = 9e9, None, 0
    for ep in range(100):
        model.train()
        ep_perm = rng.permutation(tr_i)
        for b in range(0, len(ep_perm), 256):
            idx = ep_perm[b:b + 256]
            opt.zero_grad()
            loss = pinball_loss(model(Xt[idx], gt[idx]), yt[idx], wt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_parts = [float(pinball_loss(model(Xt[va_i[s:s + 2048]], gt[va_i[s:s + 2048]]),
                                           yt[va_i[s:s + 2048]], wt[va_i[s:s + 2048]]))
                        for s in range(0, len(va_i), 2048)]
        va_loss = float(np.mean(va_parts))
        if va_loss < best - 1e-5:
            best, best_state, patience = va_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 10:
            break
    model.load_state_dict(best_state)
    model.eval()
    Z = np.empty((len(g_dtm), 64), dtype=np.float32)
    Xg_all = ((g_field - mu) / sd).astype(np.float32)
    with torch.no_grad():
        for s in range(0, len(g_dtm), 2048):
            xb = torch.tensor(Xg_all[s:s + 2048])
            Z[s:s + 2048] = model.conv(xb).flatten(1).numpy()
    return Z


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    g_dtm, g_field = load_grid_hours()
    gidx = {int(t): i for i, t in enumerate(g_dtm)}
    print(f"grid {len(g_dtm)}시간 ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    gsc = {v: {t: [] for t in TARGET_COLS} for v in VARIANTS}

    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        fold = f"{va_start:%Y-%m}"
        emb_f = CACHE / f"{fold}_emb.npz"
        if emb_f.exists():
            Z = np.load(emb_f)["z"]
        else:
            Z = train_fold_embedding(fold, va_start.value, df, w_q3, g_dtm, g_field, gidx, t0)
            np.savez_compressed(emb_f, z=Z, dtm=g_dtm)
            print(f"[{fold}] 임베딩 캐시 저장 ({time.time()-t0:.0f}s)", flush=True)

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

            tr_ok = tr.loc[tr[tgt].notna()].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            Xlib = ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy()
            va_rows = df.set_index("key").loc[dtm]
            Xva = ((va_rows[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy()
            lib_vals = tr_ok[tgt].to_numpy()
            lib_keys = tr_ok.forecast_kst_dtm.astype("datetime64[ns]").astype("int64").to_numpy()

            # 임베딩 행렬 (라이브러리 기준 표준화)
            lib_pos = np.array([gidx.get(int(k), -1) for k in lib_keys])
            val_pos = np.array([gidx.get(int(k), -1) for k in dtm])
            lib_has = lib_pos >= 0
            Zmu = Z[lib_pos[lib_has]].mean(0)
            Zsd = Z[lib_pos[lib_has]].std(0) + 1e-8
            Zn = (Z - Zmu) / Zsd

            Pmax = min(400, len(Xlib))
            knn = NearestNeighbors(n_neighbors=Pmax).fit(Xlib)
            _, aidx_p = knn.kneighbors(Xva)

            # 풀 내 임베딩 거리
            ed = np.full(aidx_p.shape, np.nan, dtype=np.float32)
            for s in range(0, n_row, 256):
                e = min(s + 256, n_row)
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

            # embk: 임베딩 공간 전역 kNN (장 있는 라이브러리만)
            knn_e = NearestNeighbors(n_neighbors=200).fit(Zn[lib_pos[lib_has]])
            lib_map = np.where(lib_has)[0]
            _, eidx_raw = knn_e.kneighbors(Zn[np.clip(val_pos, 0, None)])
            eidx = lib_map[eidx_raw]

            anen_v = {"base": anen_c}
            P = Pmax
            for v in ["rse4", "ffe4"]:
                if v == "ffe4":
                    order = np.argsort(ed[:, :P], axis=1, kind="stable")
                else:
                    erank = np.argsort(np.argsort(ed[:, :P], axis=1, kind="stable"),
                                       axis=1, kind="stable")
                    order = np.argsort(np.arange(P)[None, :] + erank, axis=1, kind="stable")
                idx200 = np.take_along_axis(aidx_p[:, :P], order[:, :200], axis=1)
                bad = np.isnan(ed[:, 0])
                idx200[bad] = aidx_p[bad, :200]
                anen_v[v] = np.sort(lib_vals[idx200], axis=1)
            idx_e = eidx.copy()
            bad = val_pos < 0
            idx_e[bad] = aidx_p[bad, :200]
            anen_v["embk"] = np.sort(lib_vals[idx_e], axis=1)

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
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v81 AnEn CNN 임베딩 검색 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
