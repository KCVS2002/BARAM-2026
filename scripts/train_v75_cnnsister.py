"""#117 v75: CNN sister — LDAPS 28×28 공간장 직접 학습 (공간장 축 최종 카드).

가설: 손공학 피처(#115)가 못 뽑은 '부호' 정보가 심층 표현엔 있을 수 있다.
설계:
  - 입력 2×28×28 (u,v @875hPa 표준화) + 그룹 임베딩 → 소형 CNN → 19분위 (cf)
  - 가중 pinball, q3 가중 승계
  - 공정 CV 미러링: fold 2024-09용은 ≤08-31 학습, fold 2024-11용은 ≤10-31 학습
    (해당 fold GBM과 동일 정보 컷) — 초반 fold는 2024-only 데이터 부족으로 불가
  - 판정 재료: ①CNN 단독 pinball/MAE vs 캐시 GBM ②q50 오차 상관 (관문:
    CatBoost 0.97+ 사망 전례 — 0.9 미만이어야 생존) ③공식 원자+CNN150 풀링 델타
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
DEV = "cpu"


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
    q = torch.tensor(QS, dtype=torch.float32, device=pred.device)[None, :]
    diff = y[:, None] - pred
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return (loss.mean(1) * w).mean()


def load_grid_hours() -> tuple:
    """전 수집일 → dtm(int64 ns) 배열, 필드 (N,2,28,28) float32."""
    dtms, fields = [], []
    for f in sorted(GRID.glob("2024-*.npz")) + sorted(GRID.glob("2025-*.npz")):
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
    print(f"grid {len(g_dtm)}시간 로드 ({time.time()-t0:.0f}s)", flush=True)

    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)

    for fold, cut in [("2024-09", "2024-09-01"), ("2024-11", "2024-11-01")]:
        cut_ns = pd.Timestamp(cut).value
        # ── 학습 표본: 2024-01-01 이후 ~ cut 이전, 그리드 존재 & 라벨 존재 ──
        Xs, ys, ws, gs = [], [], [], []
        for gi, tgt in enumerate(TARGET_COLS):
            cap = CAPACITY_KWH[tgt]
            m = (lab_dtm >= pd.Timestamp("2024-01-01 01:00").value) & (lab_dtm < cut_ns) \
                & df[tgt].notna().to_numpy()
            rows = np.where(m)[0]
            keep = [r for r in rows if lab_dtm[r] in gidx]
            Xs.append(np.stack([g_field[gidx[lab_dtm[r]]] for r in keep]))
            ys.append((df[tgt].to_numpy()[keep] / cap).astype(np.float32))
            ws.append(w_q3[tgt].to_numpy()[keep].astype(np.float32))
            gs.append(np.full(len(keep), gi, dtype=np.int64))
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        g = np.concatenate(gs)
        mu = X.mean((0, 2, 3), keepdims=True)
        sd = X.std((0, 2, 3), keepdims=True)
        Xn = (X - mu) / sd
        w = w / w.mean()
        print(f"[{fold}] 학습 {len(X)}행 (~{cut} 이전) ({time.time()-t0:.0f}s)", flush=True)

        # 내부 검증 분리 (마지막 15% 시간순) — 조기 종료용
        order = np.argsort([0] * 0 + list(range(len(X))))  # 이미 그룹블록순 — 시간순 셔플 대신 인덱스 분할
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        n_va = int(len(X) * 0.15)
        va_i, tr_i = perm[:n_va], perm[n_va:]

        model = SisterCNN().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xt = torch.tensor(Xn)
        yt = torch.tensor(y)
        wt = torch.tensor(w)
        gt = torch.tensor(g)
        best, best_state, patience = 9e9, None, 0
        for ep in range(80):
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
                va_loss = float(pinball_loss(model(Xt[va_i], gt[va_i]), yt[va_i], wt[va_i]))
            if va_loss < best - 1e-5:
                best, best_state, patience = va_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
            if ep % 10 == 0 or patience >= 8:
                print(f"[{fold}] ep{ep} va_pinball {va_loss:.5f} (best {best:.5f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if patience >= 8:
                break
        model.load_state_dict(best_state)
        model.eval()

        # ── fold 검증 시간에 대해 CNN 분위 생성 → 진단 + 풀링 ──
        for tgt_i, tgt in enumerate(TARGET_COLS):
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]
            actual, a_bar, cap = z["actual"], float(z["a_bar"]), float(z["cap"])
            ok = np.array([t in gidx for t in z["dtm"]])
            if ok.mean() < 0.99:
                print(f"[{fold}] {tgt}: 그리드 커버 {ok.mean()*100:.1f}% — 누락 중앙값 대체", flush=True)
            Xv = np.stack([g_field[gidx.get(t, -1)] if t in gidx else g_field[0] for t in z["dtm"]])
            Xv = (Xv - mu) / sd
            with torch.no_grad():
                qcnn = model(torch.tensor(Xv.astype(np.float32)),
                             torch.full((len(Xv),), tgt_i, dtype=torch.long)).numpy()
            qcnn = np.sort(qcnn, axis=1) * cap
            dts = pd.Series(pd.to_datetime(z["dtm"]))
            qcnn = np.sort(smooth_quantiles_by_day(qcnn, dts), axis=1)

            e_c = qcnn[:, 9] - actual
            e_g = qp_gbm[:, 9] - actual
            pb_c = np.mean([np.mean(np.maximum(q * (actual - qcnn[:, j]), (q - 1) * (actual - qcnn[:, j])))
                            for j, q in enumerate(QS)])
            pb_g = np.mean([np.mean(np.maximum(q * (actual - qp_gbm[:, j]), (q - 1) * (actual - qp_gbm[:, j])))
                            for j, q in enumerate(QS)])
            corr = np.corrcoef(e_c, e_g)[0, 1]
            print(f"[{fold}] {tgt}: pinball CNN {pb_c:.0f} / GBM {pb_g:.0f} (비율 {pb_c/pb_g:.3f}) "
                  f"| MAE {np.abs(e_c).mean():.0f}/{np.abs(e_g).mean():.0f} | q50 오차상관 {corr:.3f}", flush=True)

            # 공식 원자 조립 + CNN150 풀링
            n_row = len(qp_gbm)
            gbm300 = interp_atoms(qp_gbm, n=2 * K)
            med = np.median(gbm300, axis=1)
            d_bar = dist[:, :K].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
            cnn300 = interp_atoms(qcnn, n=2 * K)
            cnn150 = cnn300[:, np.linspace(0, 2 * K - 1, K).astype(int)]
            scores = {}
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
                scores[v], nm, fi, _ = metric_single(actual, pred, cap)
            print(f"[{fold}] {tgt}: base {scores['base']:.4f} → +CNN150 {scores['pool']:.4f} "
                  f"(Δ {scores['pool']-scores['base']:+.4f}) ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
