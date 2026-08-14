# -*- coding: utf-8 -*-
"""sub_056: TabPFN 컨텍스트 스케일링 프로브 (#136 후속) — TabPFN을 cmix(6k75+12k50)로.

CV 확증(v86): tpf75 fold평균 +0.0037 (NMAE +0.0015 / FICR +0.0059), 보완>대체.
구성: sub_053 흐름 그대로 (CNN75 유지) + g1/g2에 TabPFN 75원자 추가. g3 완전 불변.
TabPFN 프로토콜 = CV와 동일: TabPFN v2 (tabpfn 2.2.1), 전체 train에서 q3 가중
서브샘플 ctx 6000, n_estimators 2, seed 42, 19분위 → 정렬·3h평활.
테스트 피처(134)는 dump_test_atoms와 동일 재구성.
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
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from scripts.train_v75c_cnn_scaled import SisterCNN, load_grid_hours, pinball_loss
from scripts.make_submission_from_cache import optimize_bagged
from src.decision import interp_atoms, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
CACHE = PROJECT / "experiments" / "test_cache"
K = 150
P_MIX = 0.15
NA = {0: 125, 1: 150, 2: 175}
HOUR_NS = 3600 * 10 ** 9
KP = [0.00, 0.45, 0.60, 1.00]
QS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CTX = 6000
torch.manual_seed(42)
np.random.seed(42)


def build_test_features(ref):
    """dump_test_atoms와 동일한 134피처 테스트 프레임 (ref 행 순서 정렬)."""
    feat_te = build_features(
        pd.read_csv(DATA / "test/ldaps_test.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "test/gfs_test.csv", encoding="utf-8-sig"),
    )
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat_tr = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    df_tr = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat_tr, on="forecast_kst_dtm", how="left")
    te = ref[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
    ifs = load_ifs_features()
    df_tr = df_tr.merge(ifs, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs, on="forecast_kst_dtm", how="left")
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    mu_t, sd_t = df_tr[tri].mean(), df_tr[tri].std()
    for d_ in (df_tr, te):
        z = (d_[tri] - mu_t) / sd_t
        d_["cons3_mean"] = z.mean(axis=1)
        d_["cons3_std"] = z.std(axis=1)
    ifs2 = load_ifs2_features()
    te = te.merge(ifs2, on="forecast_kst_dtm", how="left")
    te["ifs_shear"] = te["ifs_ws700"] - te["ifs_ws925"]
    return te


def main() -> None:
    t0 = time.time()
    from tabpfn import TabPFNRegressor
    df, w_q3, cols, _ = load_base_frame()
    g_dtm, g_field = load_grid_hours()
    gidx = {int(t): i for i, t in enumerate(g_dtm)}
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)

    # ── CNN 전체 학습 (sub_053과 동일) ──
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
        if patience >= 10:
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f"CNN 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

    ref45 = pd.read_csv(PROJECT / "submissions" / "sub_045_wdeep.csv", encoding="utf-8-sig",
                        parse_dates=["forecast_kst_dtm"])
    ref53 = pd.read_csv(PROJECT / "submissions" / "sub_055_tabpfn.csv", encoding="utf-8-sig",
                        parse_dates=["forecast_kst_dtm"])  # 비교 기준 = 현 공식 sub_055
    te_dtm = ref45["forecast_kst_dtm"].values.astype("datetime64[ns]").astype(np.int64)

    def field_of(t):
        for cand in (t, t - 24 * HOUR_NS, t + 24 * HOUR_NS):
            if int(cand) in gidx:
                return g_field[gidx[int(cand)]]
        raise KeyError(pd.to_datetime(t))
    Xte = np.stack([field_of(t) for t in te_dtm])
    Xte = ((Xte - mu) / sd).astype(np.float32)

    te_feat = build_test_features(ref45)
    n_nan = int(te_feat[cols].isna().any(axis=1).sum())
    print(f"테스트 피처 준비 (NaN 포함 행 {n_nan}) ({time.time()-t0:.0f}s)", flush=True)

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

            # ── TabPFN sister — cmix: ctx 6k(75원자) + ctx 12k(50원자), v88 CV 프로토콜 ──
            trm = df[tgt].notna()
            tr = df[trm]
            wv = w_q3[tgt].to_numpy()[trm.to_numpy()]
            Xtr_full = tr[cols].to_numpy(dtype=np.float32)
            ytr_full = (tr[tgt] / cap).to_numpy(dtype=np.float32)
            Xva = te_feat[cols].to_numpy(dtype=np.float32)
            p = wv / wv.sum()
            tp_atoms = []
            for ctx_n, natom in ((6000, 75), (12000, 50)):
                rng_t = np.random.default_rng(42)
                idx = rng_t.choice(len(tr), size=min(ctx_n, len(tr)), replace=False, p=p)
                reg = TabPFNRegressor(n_estimators=2, ignore_pretraining_limits=True,
                                      random_state=42, device="auto", memory_saving_mode=True)
                reg.fit(Xtr_full[idx], ytr_full[idx])
                qp_t = np.empty((n_row, len(QS)), dtype=np.float64)
                for s in range(0, n_row, 128):
                    r = reg.predict(Xva[s:s + 128], output_type="quantiles", quantiles=QS)
                    qp_t[s:s + 128] = np.column_stack(r) if isinstance(r, list) else r
                qp_t = np.clip(np.sort(qp_t, axis=1), 0, 1) * cap
                qp_t = np.sort(smooth_quantiles_by_day(qp_t, ref45["forecast_kst_dtm"]), axis=1)
                tp_atoms.append(interp_atoms(qp_t, n=2 * K)[:, np.linspace(0, 2 * K - 1, natom).astype(int)])
                print(f"{tgt}: TabPFN ctx{ctx_n} 추론 완료 ({time.time()-t0:.0f}s)", flush=True)

            atoms = np.sort(np.concatenate([base_atoms, cnn75] + tp_atoms, axis=1), axis=1)
            pred = optimize_bagged(atoms, cap, a_bar)

        p = pred / cap
        s = np.where(winter, np.interp(p, KP, [0.88, 0.88, 1.0, 1.0]),
                     np.interp(p, KP, [0.92, 0.92, 1.0, 1.0]))
        out[tgt] = np.clip(pred * s, 0, cap)
        d53 = np.abs(out[tgt].to_numpy() - ref53[tgt].to_numpy())
        print(f"{tgt}: 완료 — sub_055 대비 |Δ| mean {d53.mean():.0f} / max {d53.max():.0f} kWh"
              f" ({time.time()-t0:.0f}s)", flush=True)

    assert np.abs(out["kpx_group_3"].to_numpy() - ref45["kpx_group_3"].to_numpy()).max() < 1.0, \
        "g3 불변 검증 실패"
    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_056_tabpfn_cmix.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    print(f"saved {path.name} — g3 비트 일치 검증 통과 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
