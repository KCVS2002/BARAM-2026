"""#121 v75d: CNN sister 6-fold 확증 — g1/g2 전용 풀링, #91 완전 판정.

v75c(2/6 fold)에서 g1/g2 풀링 +0.006~+0.019 / g3 파괴적 → 6-fold 전체로 확증.
변형: base_off / cnn150(g1·g2만 +150, g3 불변) / cnn75(동일 +75)
출력: fold평균·풀링 채점·NMAE/FICR 분해 + fold별. CNN qp는 {fold}_{tgt}_cnnqp.npz
캐시 저장 (제출·후속 재사용). 아키텍처·프로토콜 = v75c (시간 블록 조기종료).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from scripts.train_v75c_cnn_scaled import SisterCNN, load_grid_hours, pinball_loss, QS
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
VARIANTS = ["base_off", "cnn150", "cnn75"]
POOL_N = {"cnn150": 150, "cnn75": 75}
torch.manual_seed(42)
np.random.seed(42)


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    g_dtm, g_field = load_grid_hours()
    gidx = {t: i for i, t in enumerate(g_dtm)}
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)
    print(f"grid {len(g_dtm)}시간 ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        fold = f"{va_start:%Y-%m}"
        cut_ns = va_start.value

        cnn_cached = all((CACHE / f"{fold}_{t}_cnnqp.npz").exists() for t in TARGET_COLS)
        if not cnn_cached:
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
            for tgt_i, tgt in enumerate(TARGET_COLS):
                z = np.load(CACHE / f"{fold}_{tgt}.npz")
                cap = CAPACITY_KWH[tgt]
                Xv = np.stack([g_field[gidx[t]] for t in z["dtm"]])
                Xv = ((Xv - mu) / sd).astype(np.float32)
                with torch.no_grad():
                    qcnn = model(torch.tensor(Xv), torch.full((len(Xv),), tgt_i, dtype=torch.long)).numpy()
                qcnn = np.sort(qcnn, axis=1) * cap
                qcnn = np.sort(smooth_quantiles_by_day(qcnn, pd.Series(pd.to_datetime(z["dtm"]))), axis=1)
                np.savez_compressed(CACHE / f"{fold}_{tgt}_cnnqp.npz", qp=qcnn)
            print(f"[{fold}] CNN 학습·캐시 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            qcnn = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            cnn300 = interp_atoms(qcnn, n=2 * K)
            for v in VARIANTS:
                npool = POOL_N.get(v, 0) if tgt != "kpx_group_3" else 0
                cnnA = cnn300[:, np.linspace(0, 2 * K - 1, npool).astype(int)] if npool else None
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + ([cnnA[i]] if npool else [])
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

    print("\n=== v75d CNN sister 6-fold (g1/g2 전용) — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:8s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")
    for v in VARIANTS:
        print(f"{v:8s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))


if __name__ == "__main__":
    main()
