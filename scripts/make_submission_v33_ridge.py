"""제출 sub_033: sub_029 + ridge 6피처 (#91 v58/v58b, 이론 조사 티어1 생존 축).
능선 상대 풍향 U⊥ = ws·|sin(wd−주향)| (주향 N20°/N140° — OSM 실측 말굽 두 다리)
× Froude Fr = U⊥850/(N·h) (N: IFS 온위차). perp·fr 단독은 음(-0.0007/-0.0012),
결합만 +0.0015 (FICR +0.0028) — 초가산 상호작용 (능선 가속의 안정도 레짐 의존).
main GBM 전용 — sister/AnEn/결정층 불변.

sub_012(공식 최고 0.64069)와 동일하되, kpx_group_3에만 OM sister 원자 150개를 추가:
- sister = LGBM 19q, 학습 2024년 전체(3그룹 공유 스택), 피처 = 기존 125 + OM 18
  (ECMWF IFS025/ICON/GFS previous_day2·3 — 항상 D-1 13:00 이전 발표, 누수 안전)
- raw 결합(재정렬 없음): 홀드아웃에서 raw(+0.018) > shift(+0.010)
- g1/g2는 홀드아웃에서 악화(-0.01~-0.03)라 미적용
홀드아웃(2024-10~12) ablation: 이득 +0.018 = 최근성 +0.009 + OM 정보 +0.009.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import BASE_PARAMS, GROUP_META, QUANTILES, label_weights
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W, K
from scripts.train_v22_omsister import load_om
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day


def optimize_bagged(atoms, cap, a_bar, B=15, frac=0.6, seed=42):
    """부트스트랩 원자 서브셋 B회 결정의 평균 — argmax 노이즈 안정화 (#64)."""
    rng = np.random.default_rng(seed)
    n = atoms.shape[1]
    k = int(n * frac)
    gs = np.empty((atoms.shape[0], B))
    for b in range(B):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        gs[:, b] = optimize_submission(atoms[:, idx], cap, a_bar)
    return np.clip(gs.mean(axis=1), 0, cap)
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
P_MIX = 0.15


def main() -> None:
    t0 = time.time()
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
    # 고출력 가중 (#83 w2): main GBM 전용 ×(1+2·cf) — sister는 불변
    weights_main = weights.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        weights_main[tgt] = weights_main[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    om = load_om()
    om_cols = [c for c in om.columns if c != "forecast_kst_dtm"]
    df = (lab.rename(columns={"kst_dtm": "forecast_kst_dtm"})
          .merge(feat_tr, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))
    te = (sub[["forecast_kst_dtm"]].merge(feat_te, on="forecast_kst_dtm", how="left")
          .merge(om, on="forecast_kst_dtm", how="left"))
    # IFS 컨센서스 5피처 (main GBM 전용 — sister/AnEn 불변)
    ifs = load_ifs_features()
    df = df.merge(ifs, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs, on="forecast_kst_dtm", how="left")
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    mu_t, sd_t = df[tri].mean(), df[tri].std()  # 학습 통계로 표준화 (test에 동일 적용)
    for d_ in (df, te):
        z = (d_[tri] - mu_t) / sd_t
        d_["cons3_mean"] = z.mean(axis=1)
        d_["cons3_std"] = z.std(axis=1)
    # wave-2 물리 5피처 (v31b: 안정도·시어·습도 — main GBM 전용)
    ifs2 = load_ifs2_features()
    df = df.merge(ifs2, on="forecast_kst_dtm", how="left")
    te = te.merge(ifs2, on="forecast_kst_dtm", how="left")
    for d_ in (df, te):
        d_["ifs_shear"] = d_["ifs_ws700"] - d_["ifs_ws925"]
    # ridge 6피처 (#91): 능선 수직 성분 + Froude — train/test 동일 계산
    def add_ridge(d_):
        cols = []
        for c, pre in (("ldaps_ws50max", "l50"), ("gfs_ws100", "g100"), ("gfs_ws850", "g850")):
            wd = np.degrees(np.arctan2(d_[f"{c}_dsin"], d_[f"{c}_dcos"])) % 360
            for az in (20.0, 140.0):
                col = f"perp_{pre}_{int(az)}"
                d_[col] = d_[c] * np.abs(np.sin(np.radians(wd - az)))
                if pre != "g850":
                    cols.append(col)
        t925 = d_["ifs_t925"].to_numpy()
        t925 = np.where(t925 < 200, t925 + 273.15, t925)
        t850 = t925 - d_["ifs_dt"].to_numpy()
        th925 = t925 * (1000 / 925) ** 0.286
        th850 = t850 * (1000 / 850) ** 0.286
        n2 = 9.81 / ((th850 + th925) / 2) * (th850 - th925) / 650.0
        n_bv = np.sqrt(np.clip(n2, 1e-7, None))
        for az in (20.0, 140.0):
            col = f"fr_{int(az)}"
            d_[col] = np.clip(d_[f"perp_g850_{int(az)}"] / (n_bv * 1000.0), 0, 10)
            cols.append(col)
        return cols

    ridge_cols = add_ridge(df)
    assert add_ridge(te) == ridge_cols
    ifs_cols = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
                "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"] + ridge_cols
    cov25 = te[ifs_cols].notna().mean()
    print(f"IFS 2025 커버리지 min {cov25.min()*100:.1f}% / mean {cov25.mean()*100:.1f}%", flush=True)

    # ── base GBM (sub_009/012와 동일: 전 기간 공유 학습) ──
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
    qmodels = [lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
               .fit(X_all[shared_cols], y_all, sample_weight=w_all) for q in QUANTILES]
    print(f"base GBM done ({time.time()-t0:.0f}s)", flush=True)

    # ── OM sister GBM (2024년 전체, 공유 학습, 기존+OM 피처) ──
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
    sisters = [lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
               .fit(sX[shared_sis], sy, sample_weight=sw) for q in QUANTILES]
    print(f"OM sister done: {len(sX)}행 학습 ({time.time()-t0:.0f}s)", flush=True)

    out = sub[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        cap = CAPACITY_KWH[tgt]
        rated, rotor = GROUP_META[tgt]
        Xv = te[feature_cols + ifs_cols].copy()
        Xv["g_rated"], Xv["g_rotor"] = rated, rotor
        Xv["g_id"] = list(GROUP_META).index(tgt)
        qp = np.column_stack([np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels])
        qp.sort(axis=1)
        qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
        gbm300 = interp_atoms(qp, n=2 * K)
        gbm_atoms = gbm300[:, np.linspace(0, 2 * K - 1, K).astype(int)]

        trm = df[tgt].notna()
        tr_ok = df.loc[trm].dropna(subset=ANEN_FEATS)
        mu = tr_ok[ANEN_FEATS].mean()
        sd = tr_ok[ANEN_FEATS].std().replace(0, 1)
        nn = NearestNeighbors(n_neighbors=200).fit(((tr_ok[ANEN_FEATS] - mu) / sd * FEAT_W).to_numpy())
        dist, idx = nn.kneighbors(((te[ANEN_FEATS].fillna(mu) - mu) / sd * FEAT_W).to_numpy())
        anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[idx] * cap, axis=1)
        med = np.median(gbm_atoms, axis=1)
        # 희귀 레짐 재배분 (#77 inv_soft): d̄ 3분위 — 멂 anen175 / 중간 150 / 가까움 125
        d_bar = dist[:, :K].mean(axis=1)
        terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
        NA = {0: 125, 1: 150, 2: 175}
        base_atoms = np.empty((len(te), 2 * K))
        for i in range(len(te)):
            na = NA[terc[i]]
            ng = 2 * K - na
            an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
            an = np.clip(an + (med[i] - np.median(an)), 0, cap)
            gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
            base_atoms[i] = np.sort(np.concatenate([an, gb]))
        parts = [base_atoms]
        n_terc = np.bincount(terc, minlength=3)
        print(f"{tgt}: 거리 3분위 분포 {n_terc.tolist()}", flush=True)

        if tgt == "kpx_group_3":
            # (1) 가용률 혼합 (sub_012 채택분)
            n = 2 * K
            n8 = max(int(round(n * P_MIX * 2 / 3)), 1)
            n6 = max(int(round(n * P_MIX * 1 / 3)), 1)
            parts += [gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
                      gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6]
            # (2) OM sister 원자 (raw, 이번 A/B 대상)
            Xvs = te[sis_cols].copy()
            Xvs["g_rated"], Xvs["g_rotor"] = rated, rotor
            Xvs["g_id"] = list(GROUP_META).index(tgt)
            sq = np.column_stack([np.clip(m.predict(Xvs[shared_sis]) * cap, 0, cap) for m in sisters])
            sq.sort(axis=1)
            sq = np.sort(smooth_quantiles_by_day(sq, sub["forecast_kst_dtm"]), axis=1)
            sis300 = interp_atoms(sq, n=2 * K)
            print(f"g3: 혼합 +{n8+n6}, OM sister 조건부(125/150/175) 원자", flush=True)

        a = df.loc[trm, tgt]
        a_bar = a[a >= cap * 0.10].mean()
        if tgt == "kpx_group_3":
            # sister 조건부 (#78 v46): 3분위 그룹별로 조립·배깅 결정 후 병합
            fixed = np.concatenate(parts, axis=1)
            g_out = np.empty(len(te))
            for t_, ns in ((0, 125), (1, 150), (2, 175)):
                m = terc == t_
                sis = np.array([np.interp(np.linspace(0, 2 * K - 1, ns),
                                          np.arange(2 * K), s) for s in sis300[m]])
                atoms_g = np.sort(np.concatenate([fixed[m], sis], axis=1), axis=1)
                g_out[m] = optimize_bagged(atoms_g, cap, a_bar)
            out[tgt] = g_out
        else:
            atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
            out[tgt] = optimize_bagged(atoms, cap, a_bar)
        print(f"{tgt}: pred mean={out[tgt].mean():.0f} ({time.time()-t0:.0f}s)", flush=True)

    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = PROJECT / "submissions" / "sub_033_ridge.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(path, encoding="utf-8-sig")
    assert len(chk) == 8760 and chk[TARGET_COLS].notna().all().all()
    assert (chk["forecast_id"] == sub["forecast_id"]).all()
    print(f"saved {path.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
