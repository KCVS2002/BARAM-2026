"""#120 v75c: CNN sister 본 실험 — 데이터 확장(2023+2024, 2022 도착 시 자동 포함).

v75 소형 아키텍처 유지 (v75b에서 대형화 기각). 변경 2건:
  ① 학습창 동적: 그리드 존재 ∧ 라벨 존재 ∧ dtm < fold cut (2023 완성으로 ~2.4배)
  ② 조기종료 검증: 무작위 15% → 시간 블록 (학습창 마지막 12%) — v75b 누출 교훈
프로토콜 동일: fold09 ≤08-31 / fold11 ≤10-31, 관문 3종.
v75 비교 기준: pinball 비율 1.24~1.56 / 상관 0.699~0.760 / 풀링 5/6 음수.
데이터 스케일링 기울기 = (v75c 비율 − v75 비율) — 2022 추가분의 기대값 추정 재료.
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
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gemb = nn.Embedding(3, 8)
        self.head = nn.Sequential(nn.Linear(64 + 8, 64), nn.ReLU(), nn.Linear(64, 19))

    def forward(self, x, g):
        z = self.conv(x).flatten(1)
        return torch.sigmoid(self.head(torch.cat([z, self.gemb(g)], 1)))


def pinball_loss(pred, y, w):
    q = torch.tensor(QS, dtype=torch.float32)[None, :]
    diff = y[:, None] - pred
    return (torch.maximum(q * diff, (q - 1) * diff).mean(1) * w).mean()


def load_grid_hours():
    dtms, fields = [], []
    for f in sorted(GRID.glob("20*.npz")):
        z = np.load(f)
        d = pd.Timestamp(f.stem)
        hrs = pd.date_range(d + pd.Timedelta(hours=1), periods=24, freq="h")
        dtms.append(hrs.values.astype("datetime64[ns]").astype(np.int64))
        fields.append(np.stack([z["u"], z["v"]], axis=1))
    return np.concatenate(dtms), np.concatenate(fields)


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    g_dtm, g_field = load_grid_hours()
    gidx = {t: i for i, t in enumerate(g_dtm)}
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)
    print(f"grid {len(g_dtm)}시간 ({pd.to_datetime(g_dtm.min())} ~) ({time.time()-t0:.0f}s)", flush=True)

    for fold, cut in [("2024-09", "2024-09-01"), ("2024-11", "2024-11-01")]:
        cut_ns = pd.Timestamp(cut).value
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
        # 시간 블록 조기종료 검증: 학습창 마지막 12% 시각
        thr = np.quantile(dt, 0.88)
        va_i = np.where(dt >= thr)[0]
        tr_i = np.where(dt < thr)[0]
        print(f"[{fold}] 학습 {len(tr_i)} / 시간블록 val {len(va_i)}행 "
              f"(경계 {pd.to_datetime(int(thr))}) ({time.time()-t0:.0f}s)", flush=True)

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
            if ep % 10 == 0 or patience >= 10:
                print(f"[{fold}] ep{ep} va(시간블록) {va_loss:.5f} (best {best:.5f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if patience >= 10:
                break
        model.load_state_dict(best_state)
        model.eval()

        for tgt_i, tgt in enumerate(TARGET_COLS):
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            Xv = np.stack([g_field[gidx[t]] for t in z["dtm"]])
            Xv = (Xv - mu) / sd
            with torch.no_grad():
                qcnn = model(torch.tensor(Xv.astype(np.float32)),
                             torch.full((len(Xv),), tgt_i, dtype=torch.long)).numpy()
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
            cnn300 = interp_atoms(qcnn, n=2 * K)
            pool_n = {"pool150": K, "pool75": 75}
            sc = {}
            for v, npool in [("base", 0)] + list(pool_n.items()):
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
                sc[v], _, _, _ = metric_single(actual, pred, cap)
            print(f"[{fold}] {tgt}: base {sc['base']:.4f} → +150 {sc['pool150']:.4f} "
                  f"(Δ {sc['pool150']-sc['base']:+.4f}) / +75 {sc['pool75']:.4f} "
                  f"(Δ {sc['pool75']-sc['base']:+.4f}) ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
