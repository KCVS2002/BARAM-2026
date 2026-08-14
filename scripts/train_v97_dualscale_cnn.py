"""#146 v97: 이중 스케일 CNN sister — LDAPS 국지(28×28, ±21km) + GFS 종관(11×11, 275km).

동기: 사용자 방향 전환("수집 말고 가진 데이터에 새 방법"). GFS 광역 격자는 GBM
통계 피처로만 소비·기각(#94)됐고, 그 뒤 확립된 법칙 — 공간장은 비지도 요약으로
죽고(v61·v80) 지도학습 CNN으로만 가치가 나온다(CNN sister) — 의 광역판이 미검증.
크롭 확대 재수집이 노리던 '더 넓은 시야'를 이미 가진 데이터로 얻는 경로.

구조: convL(현 SisterCNN 스택)→64 + convG(11×11 풍속 지도)→32 + 도메인 u/v 평균(2)
+ 그룹 임베딩(8) → head → 19분위. 프로토콜 = v75d (q3 가중, 시간블록 12%, seed 42).
GFS 결측 슬롯(0.03%)은 ±24h 폴백.

변형 (g1/g2, g3 불변, tpf75 포함 공식 구성 위):
  base : cnn75 + tpf75 (공식 sub_055 구성)
  rep  : dual75 + tpf75 (국지 CNN을 이중 스케일로 교체)
  add  : cnn75 + dual75 + tpf75 (병렬 추가)
캐시: {fold}_{tgt}_cnn2qp.npz. 진단: dual↔cnn 오차 상관·pinball. 판정 #91, 판정은 사용자.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from scripts.train_v75c_cnn_scaled import load_grid_hours, pinball_loss, QS
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
GFS_CSV = PROJECT / "external_data" / "noaa_gfs" / "gfs_grid100_20220101_20251231.csv"
K = 150
NA = {0: 125, 1: 150, 2: 175}
HOUR_NS = 3600 * 10 ** 9
VARIANTS = ["base", "rep", "add"]
G12 = ["kpx_group_1", "kpx_group_2"]
torch.manual_seed(42)
np.random.seed(42)


class DualScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convL = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.convG = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gemb = nn.Embedding(3, 8)
        self.head = nn.Sequential(nn.Linear(64 + 32 + 2 + 8, 64), nn.ReLU(), nn.Linear(64, 19))

    def forward(self, xl, xg, uv, g):
        zl = self.convL(xl).flatten(1)
        zg = self.convG(xg).flatten(1)
        return torch.sigmoid(self.head(torch.cat([zl, zg, uv, self.gemb(g)], 1)))


def load_gfs_maps():
    df = pd.read_csv(GFS_CSV)
    valid = (pd.to_datetime(df["run"], format="%Y%m%d%H")
             + pd.to_timedelta(df["fxx"], unit="h") + pd.Timedelta(hours=9))
    dtm = valid.values.astype("datetime64[ns]").astype(np.int64)
    ws_cols = [f"ws_{i}_{j}" for i in range(11) for j in range(11)]
    maps = df[ws_cols].to_numpy(dtype=np.float32).reshape(-1, 1, 11, 11)
    uv = df[["u_mean", "v_mean"]].to_numpy(dtype=np.float32)
    order = np.argsort(dtm)
    return dtm[order], maps[order], uv[order]


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    g_dtm, g_field = load_grid_hours()
    gidx = {int(t): i for i, t in enumerate(g_dtm)}
    f_dtm, f_maps, f_uv = load_gfs_maps()
    fidx = {int(t): i for i, t in enumerate(f_dtm)}
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)
    print(f"grid L {len(g_dtm)} / G {len(f_dtm)}시간 ({time.time()-t0:.0f}s)", flush=True)

    def gpos_of(t):
        for cand in (t, t - 24 * HOUR_NS, t + 24 * HOUR_NS):
            if int(cand) in fidx:
                return fidx[int(cand)]
        return -1

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        fold = f"{va_start:%Y-%m}"
        cut_ns = va_start.value

        cached = all((CACHE / f"{fold}_{t}_cnn2qp.npz").exists() for t in G12)
        if not cached:
            Xs, Gs, Us, ys, ws, gs, ds_ = [], [], [], [], [], [], []
            for gi, tgt in enumerate(TARGET_COLS):
                cap = CAPACITY_KWH[tgt]
                m = (lab_dtm < cut_ns) & df[tgt].notna().to_numpy()
                keep = [r for r in np.where(m)[0]
                        if lab_dtm[r] in gidx and gpos_of(lab_dtm[r]) >= 0]
                Xs.append(np.stack([g_field[gidx[lab_dtm[r]]] for r in keep]))
                Gs.append(np.stack([f_maps[gpos_of(lab_dtm[r])] for r in keep]))
                Us.append(np.stack([f_uv[gpos_of(lab_dtm[r])] for r in keep]))
                ys.append((df[tgt].to_numpy()[keep] / cap).astype(np.float32))
                ws.append(w_q3[tgt].to_numpy()[keep].astype(np.float32))
                gs.append(np.full(len(keep), gi, dtype=np.int64))
                ds_.append(lab_dtm[keep])
            X = np.concatenate(Xs)
            G = np.concatenate(Gs)
            U = np.concatenate(Us)
            y = np.concatenate(ys)
            w = np.concatenate(ws)
            g = np.concatenate(gs)
            dt = np.concatenate(ds_)
            muL = X.mean((0, 2, 3), keepdims=True)
            sdL = X.std((0, 2, 3), keepdims=True)
            X = (X - muL) / sdL
            muG = G.mean()
            sdG = G.std()
            G = (G - muG) / sdG
            muU = U.mean(0, keepdims=True)
            sdU = U.std(0, keepdims=True)
            U = (U - muU) / sdU
            w = w / w.mean()
            thr = np.quantile(dt, 0.88)
            va_i = np.where(dt >= thr)[0]
            tr_i = np.where(dt < thr)[0]
            print(f"[{fold}] 학습 {len(tr_i)} / val {len(va_i)} ({time.time()-t0:.0f}s)", flush=True)
            rng = np.random.default_rng(42)
            model = DualScaleCNN()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            Xt = torch.from_numpy(X)
            Gt = torch.from_numpy(G)
            Ut = torch.from_numpy(U)
            yt, wt, gt = map(torch.tensor, (y, w, g))
            best, best_state, patience = 9e9, None, 0
            for ep in range(100):
                model.train()
                ep_perm = rng.permutation(tr_i)
                for b in range(0, len(ep_perm), 256):
                    idx = ep_perm[b:b + 256]
                    opt.zero_grad()
                    loss = pinball_loss(model(Xt[idx], Gt[idx], Ut[idx], gt[idx]), yt[idx], wt[idx])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    va_parts = [float(pinball_loss(
                        model(Xt[va_i[s:s + 2048]], Gt[va_i[s:s + 2048]],
                              Ut[va_i[s:s + 2048]], gt[va_i[s:s + 2048]]),
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
                if tgt not in G12:
                    continue
                z = np.load(CACHE / f"{fold}_{tgt}.npz")
                cap = CAPACITY_KWH[tgt]
                Xv = np.stack([g_field[gidx[int(t)]] for t in z["dtm"]])
                Xv = ((Xv - muL) / sdL).astype(np.float32)
                Gv = np.stack([f_maps[gpos_of(int(t))] for t in z["dtm"]])
                Gv = ((Gv - muG) / sdG).astype(np.float32)
                Uv = np.stack([f_uv[gpos_of(int(t))] for t in z["dtm"]])
                Uv = ((Uv - muU) / sdU).astype(np.float32)
                with torch.no_grad():
                    qcnn = model(torch.from_numpy(Xv), torch.from_numpy(Gv),
                                 torch.from_numpy(Uv),
                                 torch.full((len(Xv),), tgt_i, dtype=torch.long)).numpy()
                qcnn = np.sort(qcnn, axis=1) * cap
                qcnn = np.sort(smooth_quantiles_by_day(qcnn, pd.Series(pd.to_datetime(z["dtm"]))), axis=1)
                np.savez_compressed(CACHE / f"{fold}_{tgt}_cnn2qp.npz", qp=qcnn)
            print(f"[{fold}] dual CNN 학습·캐시 완료 ({time.time()-t0:.0f}s)", flush=True)

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
                qtp = np.load(CACHE / f"{fold}_{tgt}_tpfqp.npz")["qp"]
                qd = np.load(CACHE / f"{fold}_{tgt}_cnn2qp.npz")["qp"]
                cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                tp75 = interp_atoms(qtp, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                d75 = interp_atoms(qd, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                cc = np.corrcoef(qd[:, 9] - actual, qcur[:, 9] - actual)[0, 1]
                pbg = sum(np.mean(np.maximum(q * (actual - qp_gbm[:, j]), (q - 1) * (actual - qp_gbm[:, j])))
                          for j, q in enumerate(QS)) / len(QS)
                pbd = sum(np.mean(np.maximum(q * (actual - qd[:, j]), (q - 1) * (actual - qd[:, j])))
                          for j, q in enumerate(QS)) / len(QS)
                pbc = sum(np.mean(np.maximum(q * (actual - qcur[:, j]), (q - 1) * (actual - qcur[:, j])))
                          for j, q in enumerate(QS)) / len(QS)
                print(f"  {fold} {tgt[-1]}: dual pinball비 {pbd/pbg:.3f} (국지 {pbc/pbg:.3f}) | dual↔cnn 상관 {cc:.3f}", flush=True)
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
                        parts.append(tp75[i])
                        if v == "base":
                            parts.append(cnn75[i])
                        elif v == "rep":
                            parts.append(d75[i])
                        else:
                            parts += [cnn75[i], d75[i]]
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

    print("\n=== v97 이중 스케일 CNN — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
    for v in VARIANTS:
        print(f"{v:4s} fold별: " + " ".join(f"{f}:{res[v][f]:.4f}" for f in res[v]))
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
