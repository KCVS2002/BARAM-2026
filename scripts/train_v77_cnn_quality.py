"""#124 v77: CNN 품질 개선 3종 — 스케줄(a) / 시각 임베딩(b) / 3시드 평균(c).

sub_053 채택 후 잔여 지렛대 = CNN 품질 (현 pinball 비율 1.09~1.16).
  v77a: cosine LR(T60) + patience 20 (소형 아키텍처 그대로 — v75b는 대형화와 교락)
  v77b: a + 시각(hour) 임베딩 — 산악 일주기 순환 위상 (공간장이 못 보는 정보)
  v77c: a의 3시드(42/7/123) qp 평균 — 순도 리스크(#114) 실증 확인용
평가: 각 변형 cnn75 풀링 (g1/g2) vs base_off, 품질 비율·상관 병기.
캐시: {fold}_{tgt}_cnnqp_{a|b|c}.npz. 판정 #91 + 현 공식(cnn75, fold평균 0.6626) 대비.
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
K = 150
NA = {0: 125, 1: 150, 2: 175}
HOUR_NS = 3600 * 10 ** 9
VARIANTS = ["base_off", "v77a", "v77b", "v77c"]


class SisterCNN2(nn.Module):
    def __init__(self, use_hour=False):
        super().__init__()
        self.use_hour = use_hour
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gemb = nn.Embedding(3, 8)
        emb_dim = 64 + 8 + (8 if use_hour else 0)
        if use_hour:
            self.hemb = nn.Embedding(24, 8)
        self.head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 19))

    def forward(self, x, g, h=None):
        z = [self.conv(x).flatten(1), self.gemb(g)]
        if self.use_hour:
            z.append(self.hemb(h))
        return torch.sigmoid(self.head(torch.cat(z, 1)))


def train_one(X, y, w, g, h, dt, use_hour, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    thr = np.quantile(dt, 0.88)
    va_i = np.where(dt >= thr)[0]
    tr_i = np.where(dt < thr)[0]
    model = SisterCNN2(use_hour)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    Xt, yt, wt, gt, ht = map(torch.tensor, (X, y, w, g, h))
    best, best_state, patience = 9e9, None, 0
    for ep in range(100):
        model.train()
        ep_perm = rng.permutation(tr_i)
        for b in range(0, len(ep_perm), 256):
            idx = ep_perm[b:b + 256]
            opt.zero_grad()
            loss = pinball_loss(model(Xt[idx], gt[idx], ht[idx] if use_hour else None),
                                yt[idx], wt[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            parts = [float(pinball_loss(model(Xt[va_i[s:s + 2048]], gt[va_i[s:s + 2048]],
                                              ht[va_i[s:s + 2048]] if use_hour else None),
                                        yt[va_i[s:s + 2048]], wt[va_i[s:s + 2048]]))
                     for s in range(0, len(va_i), 2048)]
        va = float(np.mean(parts))
        if va < best - 1e-5:
            best, best_state, patience = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 20:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


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
        need = not all((CACHE / f"{fold}_{t}_cnnqp_{s}.npz").exists()
                       for t in TARGET_COLS for s in ("a", "b", "c"))
        if need:
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
            h = ((dt // HOUR_NS) % 24).astype(np.int64)
            mu = X.mean((0, 2, 3), keepdims=True)
            sd = X.std((0, 2, 3), keepdims=True)
            X = (X - mu) / sd
            w = w / w.mean()
            models = {
                "a": [train_one(X, y, w, g, h, dt, False, 42)],
                "b": [train_one(X, y, w, g, h, dt, True, 42)],
            }
            models["c"] = [models["a"][0],
                           train_one(X, y, w, g, h, dt, False, 7),
                           train_one(X, y, w, g, h, dt, False, 123)]
            print(f"[{fold}] 학습 {len(X)}행, 4모델 완료 ({time.time()-t0:.0f}s)", flush=True)
            for tgt_i, tgt in enumerate(TARGET_COLS):
                z = np.load(CACHE / f"{fold}_{tgt}.npz")
                cap = CAPACITY_KWH[tgt]
                Xv = np.stack([g_field[gidx[t]] for t in z["dtm"]])
                Xv = ((Xv - mu) / sd).astype(np.float32)
                hv = ((z["dtm"] // HOUR_NS) % 24).astype(np.int64)
                dts = pd.Series(pd.to_datetime(z["dtm"]))
                for s in ("a", "b", "c"):
                    qs_list = []
                    for mdl in models[s]:
                        with torch.no_grad():
                            q_ = mdl(torch.tensor(Xv), torch.full((len(Xv),), tgt_i, dtype=torch.long),
                                     torch.tensor(hv) if mdl.use_hour else None).numpy()
                        qs_list.append(np.sort(q_, axis=1))
                    qcnn = np.sort(np.mean(qs_list, axis=0), axis=1) * cap
                    qcnn = np.sort(smooth_quantiles_by_day(qcnn, dts), axis=1)
                    np.savez_compressed(CACHE / f"{fold}_{tgt}_cnnqp_{s}.npz", qp=qcnn)
            print(f"[{fold}] 추론·캐시 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        ratios = {v: [] for v in ("v77a", "v77b", "v77c")}
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            pb = lambda qq: np.mean([np.mean(np.maximum(q * (actual - qq[:, j]), (q - 1) * (actual - qq[:, j])))
                                     for j, q in enumerate(QS)])
            pb_g = pb(qp_gbm)
            qcnn_by = {v: np.load(CACHE / f"{fold}_{tgt}_cnnqp_{s}.npz")["qp"]
                       for v, s in (("v77a", "a"), ("v77b", "b"), ("v77c", "c"))}
            for v in VARIANTS:
                if v == "base_off":
                    cnnA = None
                else:
                    if tgt != "kpx_group_3":
                        ratios[v].append(pb(qcnn_by[v]) / pb_g)
                    cnn300 = interp_atoms(qcnn_by[v], n=2 * K)
                    cnnA = cnn300[:, np.linspace(0, 2 * K - 1, 75).astype(int)] \
                        if tgt != "kpx_group_3" else None
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + ([cnnA[i]] if cnnA is not None else [])
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
        rt = " ".join(f"{v}비율{np.mean(ratios[v]):.3f}" for v in ratios)
        print(f"fold {fold}: " + " ".join(f"{v}={res[v][fold]:.4f}" for v in VARIANTS)
              + f" | {rt} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v77 CNN 품질 3종 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
