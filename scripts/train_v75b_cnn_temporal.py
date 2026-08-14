"""#118 v75b: CNN sister 개선판 — 시간 문맥 + 학습 개선 (2024 데이터, 아키텍처 확정용).

v75 대비 변경:
  - 입력 2×28×28 (단일 시각) → 6×28×28 (t−1, t, t+1) — 램프 위상 오차는
    연속 프레임에서만 보임. 경계 시각은 중심 프레임 복제.
  - cosine LR (120ep) + EMA(0.995) 평가 + patience 15
프로토콜 v75 동일 (fold09 ≤08-31 / fold11 ≤10-31, 관문 3종). 비교 기준(v75):
  pinball 비율 1.24~1.56 / 상관 0.699~0.760 / 풀링 5/6 음수.
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
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
GRID = PROJECT / "external_data" / "kma_ldaps_grid"
QS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
K = 150
NA = {0: 125, 1: 150, 2: 175}
torch.manual_seed(42)
np.random.seed(42)


class SisterCNN(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gemb = nn.Embedding(3, 8)
        self.head = nn.Sequential(nn.Linear(96 + 8, 96), nn.ReLU(), nn.Linear(96, 19))

    def forward(self, x, g):
        z = self.conv(x).flatten(1)
        return torch.sigmoid(self.head(torch.cat([z, self.gemb(g)], 1)))


def pinball_loss(pred, y, w):
    q = torch.tensor(QS, dtype=torch.float32)[None, :]
    diff = y[:, None] - pred
    return (torch.maximum(q * diff, (q - 1) * diff).mean(1) * w).mean()


def load_grid_hours():
    dtms, fields = [], []
    for f in sorted(GRID.glob("2024-*.npz")) + sorted(GRID.glob("2025-*.npz")):
        z = np.load(f)
        d = pd.Timestamp(f.stem)
        hrs = pd.date_range(d + pd.Timedelta(hours=1), periods=24, freq="h")
        dtms.append(hrs.values.astype("datetime64[ns]").astype(np.int64))
        fields.append(np.stack([z["u"], z["v"]], axis=1))
    return np.concatenate(dtms), np.concatenate(fields)


HOUR_NS = 3600 * 10 ** 9


def stack_ctx(dtm_arr, g_field, gidx):
    """각 시각의 (t−1,t,t+1) 6채널 스택. 이웃 부재 시 중심 복제."""
    out = np.empty((len(dtm_arr), 6, 28, 28), dtype=np.float32)
    for i, t in enumerate(dtm_arr):
        c = g_field[gidx[t]]
        p = g_field[gidx.get(t - HOUR_NS, gidx[t])]
        n = g_field[gidx.get(t + HOUR_NS, gidx[t])]
        out[i] = np.concatenate([p, c, n])
    return out


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    g_dtm, g_field = load_grid_hours()
    gidx = {t: i for i, t in enumerate(g_dtm)}
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)
    print(f"grid {len(g_dtm)}시간 ({time.time()-t0:.0f}s)", flush=True)

    for fold, cut in [("2024-09", "2024-09-01"), ("2024-11", "2024-11-01")]:
        cut_ns = pd.Timestamp(cut).value
        Xs, ys, ws, gs = [], [], [], []
        for gi, tgt in enumerate(TARGET_COLS):
            cap = CAPACITY_KWH[tgt]
            m = (lab_dtm >= pd.Timestamp("2024-01-01 01:00").value) & (lab_dtm < cut_ns) \
                & df[tgt].notna().to_numpy()
            keep = [r for r in np.where(m)[0] if lab_dtm[r] in gidx]
            Xs.append(stack_ctx(lab_dtm[keep], g_field, gidx))
            ys.append((df[tgt].to_numpy()[keep] / cap).astype(np.float32))
            ws.append(w_q3[tgt].to_numpy()[keep].astype(np.float32))
            gs.append(np.full(len(keep), gi, dtype=np.int64))
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        g = np.concatenate(gs)
        mu = X.mean((0, 2, 3), keepdims=True)
        sd = X.std((0, 2, 3), keepdims=True)
        X = (X - mu) / sd
        w = w / w.mean()
        print(f"[{fold}] 학습 {len(X)}행 ({time.time()-t0:.0f}s)", flush=True)

        rng = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        n_va = int(len(X) * 0.15)
        va_i, tr_i = perm[:n_va], perm[n_va:]
        model = SisterCNN()
        ema = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.Adam(model.parameters(), lr=1.2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
        Xt, yt, wt, gt = map(torch.tensor, (X, y, w, g))
        best, best_ema, patience = 9e9, None, 0
        for ep in range(120):
            model.train()
            ep_perm = rng.permutation(tr_i)
            for b in range(0, len(ep_perm), 256):
                idx = ep_perm[b:b + 256]
                opt.zero_grad()
                loss = pinball_loss(model(Xt[idx], gt[idx]), yt[idx], wt[idx])
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(0.995).add_(v, alpha=0.005)
                        else:
                            ema[k] = v.clone()
            sched.step()
            model.eval()
            bak = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
            with torch.no_grad():
                va_loss = float(pinball_loss(model(Xt[va_i], gt[va_i]), yt[va_i], wt[va_i]))
            model.load_state_dict(bak)
            if va_loss < best - 1e-5:
                best, best_ema, patience = va_loss, {k: v.clone() for k, v in ema.items()}, 0
            else:
                patience += 1
            if ep % 10 == 0 or patience >= 15:
                print(f"[{fold}] ep{ep} va(EMA) {va_loss:.5f} (best {best:.5f}) ({time.time()-t0:.0f}s)",
                      flush=True)
            if patience >= 15:
                break
        model.load_state_dict(best_ema)
        model.eval()

        for tgt_i, tgt in enumerate(TARGET_COLS):
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            Xv = stack_ctx(np.array([t for t in z["dtm"]]), g_field, gidx)
            Xv = (Xv - mu) / sd
            with torch.no_grad():
                qcnn = model(torch.tensor(Xv), torch.full((len(Xv),), tgt_i, dtype=torch.long)).numpy()
            qcnn = np.sort(qcnn, axis=1) * cap
            qcnn = np.sort(smooth_quantiles_by_day(qcnn, pd.Series(pd.to_datetime(z["dtm"]))), axis=1)

            e_c = qcnn[:, 9] - actual
            e_g = qp_gbm[:, 9] - actual
            pb = lambda qq: np.mean([np.mean(np.maximum(q * (actual - qq[:, j]), (q - 1) * (actual - qq[:, j])))
                                     for j, q in enumerate(QS)])
            print(f"[{fold}] {tgt}: pinball CNN {pb(qcnn):.0f} / GBM {pb(qp_gbm):.0f} "
                  f"(비율 {pb(qcnn)/pb(qp_gbm):.3f}) | 상관 {np.corrcoef(e_c, e_g)[0,1]:.3f}", flush=True)

            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            cnn150 = interp_atoms(qcnn, n=2 * K)[:, np.linspace(0, 2 * K - 1, K).astype(int)]
            sc = {}
            for v in ("base", "pool"):
                atoms_l = []
                for i in range(n_row):
                    na = NA[terc[i]]
                    ng = 2 * K - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                    parts = [an, gb] + ([cnn150[i]] if v == "pool" else [])
                    atoms_l.append(np.sort(np.concatenate(parts)))
                pred = optimize_submission(np.array(atoms_l), cap, a_bar)
                sc[v], _, _, _ = metric_single(actual, pred, cap)
            print(f"[{fold}] {tgt}: base {sc['base']:.4f} → +CNN150 {sc['pool']:.4f} "
                  f"(Δ {sc['pool']-sc['base']:+.4f}) ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
