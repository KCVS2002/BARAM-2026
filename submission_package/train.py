# -*- coding: utf-8 -*-
"""BARAM 2026 최종 제출물(sub_055_tabpfn.csv) 재현 — 학습 코드.

학습되는 모델 3종 (models/ 에 저장):
  1) 메인 LightGBM 19분위 (전체 train 2022~2024, 3그룹 공유 학습, q3 가중)
     → models/lgb_main_q{lv}.joblib × 19
  2) OM sister LightGBM 19분위 (2024년, Open-Meteo 3모델 피처, 정제 가중; group_3 전용 원자)
     → models/lgb_sister_q{lv}.joblib × 19
  3) CNN sister (LDAPS 875hPa u/v 28×28 공간장 → 19분위; group_1/2 전용 원자)
     → models/cnn_sister.pt + models/cnn_norm.npz
참고: TabPFN sister는 사전학습 모델(in-context 학습)이라 학습 단계가 없음 —
      추론 코드에서 가중치(external_data/tabpfn/)를 로컬 로드해 사용.

실행:  python train.py       (Data/ 폴더에 대회 제공 데이터 필요, 총 ~40분)
"""

import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from lib import (BASE_PARAMS, CAPACITY_KWH, DATA, GROUP_META, MODELS, QUANTILES,
                 TARGET_COLS, SisterCNN, add_consensus, label_weights, load_grid_hours,
                 load_ifs2_features, load_ifs_features, load_om, pinball_loss, q3_weights)
from src.features import build_features

torch.manual_seed(42)
np.random.seed(42)


def main() -> None:
    t0 = time.time()
    MODELS.mkdir(exist_ok=True)

    # ── 데이터·피처 (제공 데이터 + 외부데이터) ──
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    sub = pd.read_csv(DATA / "sample_submission.csv", encoding="utf-8-sig")
    sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"])

    feat_tr = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    feat_te = build_features(
        pd.read_csv(DATA / "test/ldaps_test.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "test/gfs_test.csv", encoding="utf-8-sig"),
    )
    feature_cols = [c for c in feat_tr.columns if c != "forecast_kst_dtm"]
    weights = label_weights(lab)
    weights_main = q3_weights(lab, weights)

    om = load_om()
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    df = (lab.rename(columns={"kst_dtm": "forecast_kst_dtm"})
          .merge(feat_tr, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))
    te = (sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))
    ifs = load_ifs_features()
    df = df.merge(ifs, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs, on="forecast_kst_dtm", how="left")
    ifs2 = load_ifs2_features()
    df = df.merge(ifs2, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs2, on="forecast_kst_dtm", how="left")
    add_consensus(df, te)
    print(f"features ready ({time.time()-t0:.0f}s)", flush=True)

    # ── 1) 메인 GBM 19분위 (3그룹 공유 스택, q3 가중) ──
    ifs_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
                "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    stack_X, stack_y, stack_w = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna()
        Xg = df.loc[trm, feature_cols + ifs_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xg["g_rated"], Xg["g_rotor"] = rated, rotor
        Xg["g_id"] = list(GROUP_META).index(tgt)
        stack_X.append(Xg)
        stack_y.append(df.loc[trm, tgt] / cap)
        stack_w.append(weights_main.loc[trm.to_numpy(), tgt])
    X_all, y_all, w_all = pd.concat(stack_X), pd.concat(stack_y), pd.concat(stack_w)
    shared_cols = feature_cols + ifs_cols + ["g_rated", "g_rotor", "g_id"]
    for qi, q in enumerate(QUANTILES):
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            X_all[shared_cols], y_all, sample_weight=w_all)
        joblib.dump(m, MODELS / f"lgb_main_q{int(q*100):02d}.joblib")
        print(f"main GBM {qi+1}/19 ({time.time()-t0:.0f}s)", flush=True)

    # ── 2) OM sister 19분위 (2024년, 정제 가중) ──
    sis_cols = feature_cols + om_cols
    shared_sis = sis_cols + ["g_rated", "g_rotor", "g_id"]
    in24 = df.forecast_kst_dtm.dt.year == 2024
    sX, sy, sw = [], [], []
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        trm = df[tgt].notna() & in24
        Xg = df.loc[trm, sis_cols].copy()
        rated, rotor = GROUP_META[tgt]
        Xg["g_rated"], Xg["g_rotor"] = rated, rotor
        Xg["g_id"] = list(GROUP_META).index(tgt)
        sX.append(Xg)
        sy.append(df.loc[trm, tgt] / cap)
        sw.append(weights.loc[trm.to_numpy(), tgt])
    sX, sy, sw = pd.concat(sX), pd.concat(sy), pd.concat(sw)
    for qi, q in enumerate(QUANTILES):
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
            sX[shared_sis], sy, sample_weight=sw)
        joblib.dump(m, MODELS / f"lgb_sister_q{int(q*100):02d}.joblib")
    print(f"OM sister done ({time.time()-t0:.0f}s)", flush=True)

    # ── 3) CNN sister (전체 train, 시간블록 12% 조기종료) ──
    g_dtm, g_field = load_grid_hours()
    gidx = {int(t): i for i, t in enumerate(g_dtm)}
    lab_dtm = df.forecast_kst_dtm.values.astype("datetime64[ns]").astype(np.int64)
    Xs, ys, ws, gs, ds_ = [], [], [], [], []
    for gi, tgt in enumerate(TARGET_COLS):
        cap = CAPACITY_KWH[tgt]
        m = df[tgt].notna().to_numpy()
        keep = [r for r in np.where(m)[0] if lab_dtm[r] in gidx]
        Xs.append(np.stack([g_field[gidx[lab_dtm[r]]] for r in keep]))
        ys.append((df[tgt].to_numpy()[keep] / cap).astype(np.float32))
        ws.append(weights_main[tgt].to_numpy()[keep].astype(np.float32))
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
    torch.save(model.state_dict(), MODELS / "cnn_sister.pt")
    np.savez(MODELS / "cnn_norm.npz", mu=mu, sd=sd)
    print(f"=== 학습 완료, models/ 저장 ({time.time()-t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
