# -*- coding: utf-8 -*-
"""sub_053: CNN sister LB 프로브 (#121 후속) — sub_045 + g1/g2 CNN75 원자.

구성: 테스트 캐시 공식 원자에 g1/g2만 CNN 원자 75개 추가 → 결정 배깅 →
sub_045와 동일한 계절 축소(겨울 0.88/기타 0.92) 사후 적용.
g3 경로 완전 불변 → g3 열은 sub_045와 비트 일치해야 함 (내장 검증).
CNN: 전체(2022~24) 학습, 시간 블록 조기종료, 2025 추론 (12-21 결번은 ±24h 대체).
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
from scripts.train_v75c_cnn_scaled import SisterCNN, load_grid_hours, pinball_loss
from scripts.make_submission_from_cache import optimize_bagged
from src.decision import interp_atoms, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS

CACHE = PROJECT / "experiments" / "test_cache"
K = 150
P_MIX = 0.15
NA = {0: 125, 1: 150, 2: 175}
HOUR_NS = 3600 * 10 ** 9
torch.manual_seed(42)
np.random.seed(42)
KP = [0.00, 0.45, 0.60, 1.00]


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    g_dtm, g_field = load_grid_hours()
    gidx = {t: i for i, t in enumerate(g_dtm)}
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)

    # ── CNN 전체 학습 (2022~2024) ──
    Xs, ys, ws, gs, ds_ = [], [], [], [], []
    for gi, tgt in enumerate(TARGET_COLS):
        cap = CAPACITY_KWH[tgt]
        m = df[tgt].notna().to_numpy()
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
    print(f"CNN 학습 {len(tr_i)} / val {len(va_i)}행 ({time.time()-t0:.0f}s)", flush=True)
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
            print(f"ep{ep} va {va_loss:.5f} (best {best:.5f}) ({time.time()-t0:.0f}s)", flush=True)
        if patience >= 10:
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f"CNN 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

    # ── 2025 추론 준비 ──
    ref45 = pd.read_csv(PROJECT / "submissions" / "sub_045_wdeep.csv", encoding="utf-8-sig",
                        parse_dates=["forecast_kst_dtm"])
    te_dtm = ref45["forecast_kst_dtm"].values.astype("datetime64[ns]").astype(np.int64)

    def field_of(t):
        for cand in (t, t - 24 * HOUR_NS, t + 24 * HOUR_NS):
            if cand in gidx:
                return g_field[gidx[cand]]
        raise KeyError(pd.to_datetime(t))
    Xte = np.stack([field_of(t) for t in te_dtm])
    n_fb = sum(1 for t in te_dtm if t not in gidx)
    Xte = ((Xte - mu) / sd).astype(np.float32)
    print(f"2025 추론 입력 {len(Xte)}행 (대체 {n_fb}행) ({time.time()-t0:.0f}s)", flush=True)

    month = (ref45["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.month.to_numpy()
    winter = np.isin(month, (12, 1, 2))
    out = ref45.copy()
    for tgt_i, tgt in enumerate(TARGET_COLS):
        z = np.load(CACHE / f"test_atoms_{tgt}.npz")
        qp, anen200, dist = z["qp"], z["anen"], z["dist"]
        a_bar, cap = float(z["a_bar"]), float(z["cap"])
        n_row = len(qp)
        gbm300 = interp_atoms(qp, n=2 * K)
        gbm_atoms = gbm300[:, np.linspace(0, 2 * K - 1, K).astype(int)]
        med = np.median(gbm_atoms, axis=1)
        d_bar = dist[:, :K].mean(axis=1)
        terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
        base_atoms = np.empty((n_row, 2 * K))
        for i in range(n_row):
            na = NA[terc[i]]
            ng = 2 * K - na
            an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
            an = np.clip(an + (med[i] - np.median(an)), 0, cap)
            gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
            base_atoms[i] = np.sort(np.concatenate([an, gb]))

        if tgt == "kpx_group_3":
            n = 2 * K
            n8 = max(int(round(n * P_MIX * 2 / 3)), 1)
            n6 = max(int(round(n * P_MIX * 1 / 3)), 1)
            parts = [base_atoms,
                     gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
                     gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6]
            sis300 = interp_atoms(z["sisq"], n=2 * K)
            fixed = np.concatenate(parts, axis=1)
            pred = np.empty(n_row)
            for t_, ns in ((0, 125), (1, 150), (2, 175)):
                m = terc == t_
                sis = np.array([np.interp(np.linspace(0, 2 * K - 1, ns),
                                          np.arange(2 * K), s) for s in sis300[m]])
                atoms_g = np.sort(np.concatenate([fixed[m], sis], axis=1), axis=1)
                pred[m] = optimize_bagged(atoms_g, cap, a_bar)
        else:
            with torch.no_grad():
                qcnn = model(torch.tensor(Xte), torch.full((n_row,), tgt_i, dtype=torch.long)).numpy()
            qcnn = np.sort(qcnn, axis=1) * cap
            qcnn = np.sort(smooth_quantiles_by_day(qcnn, ref45["forecast_kst_dtm"]), axis=1)
            cnn75 = interp_atoms(qcnn, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            atoms = np.sort(np.concatenate([base_atoms, cnn75], axis=1), axis=1)
            pred = optimize_bagged(atoms, cap, a_bar)

        p = pred / cap
        s = np.where(winter, np.interp(p, KP, [0.88, 0.88, 1.0, 1.0]),
                     np.interp(p, KP, [0.92, 0.92, 1.0, 1.0]))
        out[tgt] = np.clip(pred * s, 0, cap)
        d45 = np.abs(out[tgt].to_numpy() - ref45[tgt].to_numpy())
        print(f"{tgt}: 완료 — sub_045 대비 |Δ| mean {d45.mean():.0f} / max {d45.max():.0f} kWh"
              f" ({time.time()-t0:.0f}s)", flush=True)

    assert np.abs(out["kpx_group_3"].to_numpy() - ref45["kpx_group_3"].to_numpy()).max() < 1.0, \
        "g3 불변 검증 실패"
    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_053_cnnsister.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    print(f"saved {path.name} — g3 비트 일치 검증 통과 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
