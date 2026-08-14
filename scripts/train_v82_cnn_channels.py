"""#129 v82: CNN 입력 표현 변형 — 파생 채널·정규화가 소형 CNN의 정보 추출을 돕는가.

#124(학습 부드러움 축 봉인)와 별개 축: 아키텍처·학습은 고정, 입력 표현만 변경.
소형 CNN(4conv)이 u/v에서 스스로 합성하기 어려운 비선형·미분 구조를 명시 제공:
  a3  : (u, v, |V|)            — 풍속장 명시 (발전량의 1차 물리량)
  b5  : (u, v, |V|, vort, div) — 흐름 구조(회전·수렴) 미분장 추가
  px2 : (u, v) 픽셀별 표준화    — 지형 기후학 제거, 아노말리 장 (정규화 축)
평가: 공식 구성(g1/g2 +75 원자, g3 불변) base=cur75 대비. 학습 프로토콜 = v75d
(시드 42, 시간블록 12% 조기종료, patience 10). 변형 qp 캐시: _cnnqp_{d,e,f}.npz.
판정 #91 (바 0.0016), 판정은 사용자.
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
from scripts.train_v75c_cnn_scaled import load_grid_hours, pinball_loss
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
VARIANTS = ["base", "a3", "b5", "px2"]
SUF = {"a3": "d", "b5": "e", "px2": "f"}
NCH = {"a3": 3, "b5": 5, "px2": 2}
torch.manual_seed(42)
np.random.seed(42)


class SisterCNNCh(nn.Module):
    """SisterCNN과 동일 구조, 입력 채널 수만 가변."""

    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
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


def make_fields(g_field, variant):
    """(N,2,28,28) u/v → 변형별 채널 스택."""
    if variant == "px2":
        return g_field
    u, v = g_field[:, 0], g_field[:, 1]
    spd = np.sqrt(u ** 2 + v ** 2)
    if variant == "a3":
        return np.stack([u, v, spd], axis=1)
    dvdx = np.gradient(v, axis=2)
    dvdy = np.gradient(v, axis=1)
    dudx = np.gradient(u, axis=2)
    dudy = np.gradient(u, axis=1)
    vort = dvdx - dudy
    div = dudx + dvdy
    return np.stack([u, v, spd, vort, div], axis=1)


def norm_stats(X, variant):
    """px2는 픽셀별, 나머지는 채널 전역 표준화."""
    if variant == "px2":
        return X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    return X.mean((0, 2, 3), keepdims=True), X.std((0, 2, 3), keepdims=True)


def train_variant_qp(fold, cut_ns, variant, df, w_q3, g_dtm, fields, gidx, t0):
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)
    Xs, ys, ws, gs, ds_ = [], [], [], [], []
    for gi, tgt in enumerate(TARGET_COLS):
        cap = CAPACITY_KWH[tgt]
        m = (lab_dtm < cut_ns) & df[tgt].notna().to_numpy()
        keep = [r for r in np.where(m)[0] if lab_dtm[r] in gidx]
        Xs.append(np.stack([fields[gidx[lab_dtm[r]]] for r in keep]))
        ys.append((df[tgt].to_numpy()[keep] / cap).astype(np.float32))
        ws.append(w_q3[tgt].to_numpy()[keep].astype(np.float32))
        gs.append(np.full(len(keep), gi, dtype=np.int64))
        ds_.append(lab_dtm[keep])
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    g = np.concatenate(gs)
    dt = np.concatenate(ds_)
    mu, sd = norm_stats(X, variant)
    X = (X - mu) / sd
    w = w / w.mean()
    thr = np.quantile(dt, 0.88)
    va_i = np.where(dt >= thr)[0]
    tr_i = np.where(dt < thr)[0]
    rng = np.random.default_rng(42)
    model = SisterCNNCh(NCH[variant])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X)
    yt, wt, gt = map(torch.tensor, (y, w, g))
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
        if tgt == "kpx_group_3":
            continue
        z = np.load(CACHE / f"{fold}_{tgt}.npz")
        cap = CAPACITY_KWH[tgt]
        Xv = np.stack([fields[gidx[int(t)]] for t in z["dtm"]]).astype(np.float32)
        Xv = (Xv - mu) / sd
        with torch.no_grad():
            qcnn = model(torch.from_numpy(Xv), torch.full((len(Xv),), tgt_i, dtype=torch.long)).numpy()
        qcnn = np.sort(qcnn, axis=1) * cap
        qcnn = np.sort(smooth_quantiles_by_day(qcnn, pd.Series(pd.to_datetime(z["dtm"]))), axis=1)
        np.savez_compressed(CACHE / f"{fold}_{tgt}_cnnqp_{SUF[variant]}.npz", qp=qcnn)
    del X, Xt


def main() -> None:
    t0 = time.time()
    df, w_q3, cols, _ = load_base_frame()
    g_dtm, g_field = load_grid_hours()
    gidx = {int(t): i for i, t in enumerate(g_dtm)}
    print(f"grid {len(g_dtm)}시간 ({time.time()-t0:.0f}s)", flush=True)
    fields_v = {v: make_fields(g_field, v) for v in VARIANTS[1:]}

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        fold = f"{va_start:%Y-%m}"
        for v in VARIANTS[1:]:
            done = all((CACHE / f"{fold}_{t}_cnnqp_{SUF[v]}.npz").exists()
                       for t in TARGET_COLS if t != "kpx_group_3")
            if not done:
                train_variant_qp(fold, va_start.value, v, df, w_q3, g_dtm, fields_v[v], gidx, t0)
                print(f"[{fold}] {v} 학습·캐시 완료 ({time.time()-t0:.0f}s)", flush=True)

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
            has_cnn = tgt != "kpx_group_3"
            qv = {}
            if has_cnn:
                qv["base"] = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
                for v in VARIANTS[1:]:
                    qv[v] = np.load(CACHE / f"{fold}_{tgt}_cnnqp_{SUF[v]}.npz")["qp"]
            cache_pred = {}
            for v in VARIANTS:
                if not has_cnn and "g3" in cache_pred:
                    pred = cache_pred["g3"]
                else:
                    cnn75 = (interp_atoms(qv[v], n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
                             if has_cnn else None)
                    atoms_l = []
                    for i in range(n_row):
                        na = NA[terc[i]]
                        ng = 2 * K - na
                        an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                        gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                        parts = [an, gb] + ([cnn75[i]] if has_cnn else [])
                        atoms_l.append(np.sort(np.concatenate(parts)))
                    pred = optimize_submission(np.array(atoms_l), cap, a_bar)
                    if not has_cnn:
                        cache_pred["g3"] = pred
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

    print("\n=== v82 CNN 입력 채널 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
