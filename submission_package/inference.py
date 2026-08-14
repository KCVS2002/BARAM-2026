# -*- coding: utf-8 -*-
"""BARAM 2026 최종 제출물(sub_055_tabpfn.csv) 재현 — 추론 코드.

models/ 의 학습 산출물(train.py)을 로드해 2025년 8,760시간 × 3그룹 예측을 생성한다.

파이프라인 (그룹별):
  [공통] 메인 GBM 19분위 → 정렬 → 3h 평활 → 300원자 보간
         + AnEn(Analog Ensemble): train 실측 라이브러리 kNN 200이웃,
           GBM 중앙값 재정렬, 유사도 거리 3분위별 원자 수 125/150/175
  [g1/g2] + CNN sister 75원자 + TabPFN sister 75원자 (사전학습 가중치 로컬 로드,
           컨텍스트 = train 6,000행 q3 가중 서브샘플, 앙상블 2, seed 42)
  [g3]    + 가용률 혼합 원자(×0.8/×0.6) + OM sister 조건부 원자(125/150/175)
  [결정층] FICR 기대점수 최적화 + 결정 배깅 (원자 60% × 15회, seed 42)
  [사후]  계절 축소 — 겨울(12·1·2월) 0.88 / 그 외 0.92, p∈[0.45,0.60] 램프 → [0, 용량] 클리핑

실행:  python inference.py [비교대상.csv]
       (train.py 선행 필요. 비교대상 CSV를 주면 재현 오차를 출력. 총 ~60분, GPU 시 단축)
출력:  output/sub_055_reproduced.csv
"""

import os
import sys
import time

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")  # 오프라인 실행 안전

import joblib
import numpy as np
import pandas as pd
import torch

from lib import (ANEN_FEATS, CAPACITY_KWH, DATA, EXT, FEAT_W, GROUP_META, HOUR_NS, K, KP,
                 MODELS, NA_TERC, P_MIX, QUANTILES, ROOT, TARGET_COLS, SisterCNN,
                 add_consensus, label_weights, load_grid_hours, load_ifs2_features,
                 load_ifs_features, load_om, optimize_bagged, q3_weights)
from src.decision import interp_atoms, smooth_quantiles_by_day
from src.features import build_features

from sklearn.neighbors import NearestNeighbors

torch.manual_seed(42)
np.random.seed(42)
CTX = 6000


def main() -> None:
    t0 = time.time()
    (ROOT / "output").mkdir(exist_ok=True)

    # ── 데이터·피처 (train.py와 동일 재구성) ──
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
    ifs_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
                "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols = feature_cols + ifs_cols
    shared_cols = cols + ["g_rated", "g_rotor", "g_id"]
    sis_cols = feature_cols + om_cols
    shared_sis = sis_cols + ["g_rated", "g_rotor", "g_id"]
    print(f"features ready ({time.time()-t0:.0f}s)", flush=True)

    # ── 모델 로드 ──
    qmodels = [joblib.load(MODELS / f"lgb_main_q{int(q*100):02d}.joblib") for q in QUANTILES]
    sisters = [joblib.load(MODELS / f"lgb_sister_q{int(q*100):02d}.joblib") for q in QUANTILES]
    cnn = SisterCNN()
    cnn.load_state_dict(torch.load(MODELS / "cnn_sister.pt", weights_only=True))
    cnn.eval()
    nz = np.load(MODELS / "cnn_norm.npz")
    mu_c, sd_c = nz["mu"], nz["sd"]

    # ── CNN 테스트 입력 (LDAPS 공간장, 결측일 ±24h 대체) ──
    g_dtm, g_field = load_grid_hours()
    gidx = {int(t): i for i, t in enumerate(g_dtm)}
    te_dtm = sub["forecast_kst_dtm"].values.astype("datetime64[ns]").astype(np.int64)

    def field_of(t):
        for cand in (t, t - 24 * HOUR_NS, t + 24 * HOUR_NS):
            if int(cand) in gidx:
                return g_field[gidx[int(cand)]]
        raise KeyError(pd.to_datetime(t))
    Xte_f = np.stack([field_of(t) for t in te_dtm])
    n_fb = sum(1 for t in te_dtm if int(t) not in gidx)
    Xte_f = ((Xte_f - mu_c) / sd_c).astype(np.float32)
    print(f"2025 공간장 입력 {len(Xte_f)}행 (대체 {n_fb}행) ({time.time()-t0:.0f}s)", flush=True)

    from tabpfn import TabPFNRegressor
    tpf_ckpt = str(EXT / "tabpfn" / "tabpfn-v2-regressor.ckpt")

    month = (sub["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.month.to_numpy()
    winter = np.isin(month, (12, 1, 2))
    out = sub.copy()
    for tgt_i, tgt in enumerate(TARGET_COLS):
        cap = CAPACITY_KWH[tgt]
        rated, rotor = GROUP_META[tgt]

        # ── 메인 GBM 19분위 → 정렬·평활 ──
        Xv = te[cols].copy()
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = tgt_i
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)

        # ── AnEn kNN 200 ──
        trm = df[tgt].notna()
        tr_ok = df.loc[trm].dropna(subset=ANEN_FEATS)
        mu_a = tr_ok[ANEN_FEATS].mean()
        sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
        nn = NearestNeighbors(n_neighbors=200).fit(
            ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
        dist, idx = nn.kneighbors(((te[ANEN_FEATS].fillna(mu_a) - mu_a) / sd_a * FEAT_W).to_numpy())
        anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[idx] * cap, axis=1)

        a_tr = df.loc[trm, tgt]
        a_bar = float(a_tr[a_tr >= cap * 0.10].mean())
        n_row = len(qp)

        # ── 원자 조립 ──
        gbm300 = interp_atoms(qp, n=2 * K)
        gbm_atoms = gbm300[:, np.linspace(0, 2 * K - 1, K).astype(int)]
        med = np.median(gbm_atoms, axis=1)
        d_bar = dist[:, :K].mean(axis=1)
        terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
        base_atoms = np.empty((n_row, 2 * K))
        for i in range(n_row):
            na = NA_TERC[terc[i]]
            ng = 2 * K - na
            an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
            an = np.clip(an + (med[i] - np.median(an)), 0, cap)
            gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
            base_atoms[i] = np.sort(np.concatenate([an, gb]))

        if tgt == "kpx_group_3":
            # ── g3: 가용률 혼합 + OM sister 조건부 ──
            Xvs = te[sis_cols].copy()
            Xvs["g_rated"], Xvs["g_rotor"] = rated, rotor
            Xvs["g_id"] = tgt_i
            sq = np.column_stack([np.clip(m.predict(Xvs[shared_sis]) * cap, 0, cap) for m in sisters])
            sq.sort(axis=1)
            sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
            sis300 = interp_atoms(sq, n=2 * K)
            n = 2 * K
            n8 = max(int(round(n * P_MIX * 2 / 3)), 1)
            n6 = max(int(round(n * P_MIX * 1 / 3)), 1)
            fixed = np.concatenate(
                [base_atoms,
                 gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
                 gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6], axis=1)
            pred = np.empty(n_row)
            for t_, ns in ((0, 125), (1, 150), (2, 175)):
                m = terc == t_
                sis = np.array([np.interp(np.linspace(0, 2 * K - 1, ns),
                                          np.arange(2 * K), s) for s in sis300[m]])
                atoms_g = np.sort(np.concatenate([fixed[m], sis], axis=1), axis=1)
                pred[m] = optimize_bagged(atoms_g, cap, a_bar)
        else:
            # ── g1/g2: CNN sister 75원자 ──
            with torch.no_grad():
                qcnn = cnn(torch.tensor(Xte_f), torch.full((n_row,), tgt_i, dtype=torch.long)).numpy()
            qcnn = np.sort(qcnn, axis=1) * cap
            qcnn = np.sort(smooth_quantiles_by_day(qcnn, sub["forecast_kst_dtm"]), axis=1)
            cnn75 = interp_atoms(qcnn, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]

            # ── g1/g2: TabPFN sister 75원자 (사전학습, 로컬 가중치) ──
            tr_all = df[trm]
            wv = weights_main[tgt].to_numpy()[trm.to_numpy()]
            rng_t = np.random.default_rng(42)
            p = wv / wv.sum()
            idx_c = rng_t.choice(len(tr_all), size=min(CTX, len(tr_all)), replace=False, p=p)
            Xtr = tr_all[cols].to_numpy(dtype=np.float32)[idx_c]
            ytr = (tr_all[tgt] / cap).to_numpy(dtype=np.float32)[idx_c]
            Xva = te[cols].to_numpy(dtype=np.float32)
            reg = TabPFNRegressor(n_estimators=2, ignore_pretraining_limits=True,
                                  random_state=42, device="auto", memory_saving_mode=True,
                                  model_path=tpf_ckpt)
            reg.fit(Xtr, ytr)
            qp_t = np.empty((n_row, len(QUANTILES)), dtype=np.float64)
            for s in range(0, n_row, 128):
                r = reg.predict(Xva[s:s + 128], output_type="quantiles", quantiles=QUANTILES)
                qp_t[s:s + 128] = np.column_stack(r) if isinstance(r, list) else r
            qp_t = np.clip(np.sort(qp_t, axis=1), 0, 1) * cap
            qp_t = np.sort(smooth_quantiles_by_day(qp_t, sub["forecast_kst_dtm"]), axis=1)
            tp75 = interp_atoms(qp_t, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
            print(f"{tgt}: sister 추론 완료 ({time.time()-t0:.0f}s)", flush=True)

            atoms = np.sort(np.concatenate([base_atoms, cnn75, tp75], axis=1), axis=1)
            pred = optimize_bagged(atoms, cap, a_bar)

        # ── 계절 축소 + 클리핑 ──
        pnorm = pred / cap
        s = np.where(winter, np.interp(pnorm, KP, [0.88, 0.88, 1.0, 1.0]),
                     np.interp(pnorm, KP, [0.92, 0.92, 1.0, 1.0]))
        out[tgt] = np.clip(pred * s, 0, cap)
        print(f"{tgt}: 완료 ({time.time()-t0:.0f}s)", flush=True)

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = ROOT / "output" / "sub_055_reproduced.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    print(f"saved {path} ({time.time()-t0:.0f}s)", flush=True)

    # ── (선택) 원본 제출물과 비교 ──
    if len(sys.argv) > 1:
        ref = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
        for tgt in TARGET_COLS:
            d = np.abs(chk[tgt].to_numpy() - ref[tgt].to_numpy())
            print(f"[비교] {tgt}: |Δ| mean {d.mean():.2f} / max {d.max():.2f} kWh "
                  f"(용량 대비 max {d.max()/CAPACITY_KWH[tgt]*100:.3f}%)")


if __name__ == "__main__":
    main()
