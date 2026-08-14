"""#104 v68b (SCADA ① 강화): 그룹별(다리별) 실측 풍속 MOS — 방향 의존 분해.

물리 사슬 NWP→실제 허브 바람→발전량 중 첫 단계를 SCADA 나셀 풍속(사이트가 실제로
맞은 바람)으로 명시 학습. 예보 조사(deep-research)가 지목한 'history-only MOS 편향
보정'의 구현 — 테스트엔 NWP만 입력되므로 누수 없음. DeepAR 연구에서 실측 기상
공변량이 오차 절반(금지)이었는데, 그 합법 근사가 "실측을 타깃으로 한 MOS 예측값".

- 타깃: VESTAS 12기 나셀 ws의 시간 중앙값 (2022~ 전 기간, 사이트 대표 바람)
- ws_hat: 학습 행 = 시간순 3블록 내부 OOF (전 40%는 NaN — LGBM 네이티브 처리,
  스태킹 누수 차단) / 검증 행 = 전체 학습 모델
- 진단: corr(ws_hat, 실측) vs corr(ldaps_ws50max, 실측) — MOS의 정보 우위 확인
변형: base / mos(+ws_hat) / mosg(+ws_hat, ws_hat−ldaps_ws50max, ws_hat−gfs_ws100)
하네스 v58 동일. 판정 #91 기준 (봄 fold 페널티 -0.005를 넘어야 생존).
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

from scripts.train_v2_clean_shared import BASE_PARAMS, label_weights
from scripts.train_v3_sister import group_X, stack_groups
from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from scripts.train_v25_ifs import load_ifs_features
from scripts.train_v31b_phys import load_ifs2_features
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
NA_OFF = {2: 175, 1: 150, 0: 125}
VARIANTS = ["base", "mosg", "mg3"]


def load_site_ws() -> pd.DataFrame:
    dv = pd.read_csv(DATA / "train/scada_vestas_train.csv", encoding="utf-8-sig",
                     parse_dates=["kst_dtm"])
    du = pd.read_csv(DATA / "train/scada_unison_train.csv", encoding="utf-8-sig",
                     parse_dates=["kst_dtm"])
    out = {}
    for name, d, cols in (
            ("ws_g1", dv, [f"vestas_wtg{i:02d}_ws" for i in range(1, 7)]),
            ("ws_g2", dv, [f"vestas_wtg{i:02d}_ws" for i in range(7, 13)]),
            ("ws_g3", du, [f"unison_wtg{i:02d}_ws" for i in range(1, 6)])):
        hour = d["kst_dtm"].dt.ceil("h")
        out[name] = d[cols].median(axis=1).groupby(hour).mean()
    df = pd.DataFrame(out)
    df["site_ws"] = df[["ws_g1", "ws_g2"]].mean(axis=1)  # v68 호환 (전기간)
    return df


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    clean_w = label_weights(lab)
    w_q3 = clean_w.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        w_q3[tgt] = w_q3[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    site = load_site_ws()
    df = df.merge(site.rename_axis("forecast_kst_dtm").reset_index(), on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    ifs10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
             "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]
    cols0 = base_cols + ifs10
    mos_in = cols0  # MOS 입력 = 동일 NWP 피처
    print(f"site_ws 커버리지: {df['site_ws'].notna().mean()*100:.1f}% "
          f"(ready {time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in VARIANTS}
    dec = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)

        # ── MOS: 타깃 4개 (사이트 + 그룹별) — 3블록 내부 OOF + 검증 전체 모델 ──
        TGTS = ["site_ws", "ws_g1", "ws_g2", "ws_g3"]
        for name in TGTS:
            hat = "ws_hat" if name == "site_ws" else f"hat_{name[3:]}"
            mos_ok = tr_idx & df[name].notna()
            tmin = df.loc[mos_ok, "forecast_kst_dtm"].min()
            span = va_start - tmin
            edges = [tmin + span * f for f in (0.4, 0.6, 0.8)] + [va_start]
            df[hat] = np.nan
            for b in range(3):
                m_tr = mos_ok & (df.forecast_kst_dtm < edges[b])
                m_bl = mos_ok & (df.forecast_kst_dtm >= edges[b]) & (df.forecast_kst_dtm < edges[b + 1])
                if m_tr.sum() < 1000 or m_bl.sum() == 0:
                    continue
                m = lgb.LGBMRegressor(objective="regression_l2", **BASE_PARAMS).fit(
                    df.loc[m_tr, mos_in], df.loc[m_tr, name])
                df.loc[m_bl, hat] = m.predict(df.loc[m_bl, mos_in])
            mos_full = lgb.LGBMRegressor(objective="regression_l2", **BASE_PARAMS).fit(
                df.loc[mos_ok, mos_in], df.loc[mos_ok, name])
            df.loc[va_idx, hat] = mos_full.predict(df.loc[va_idx, mos_in])
        # 진단: 검증 구간 MOS 스킬 (site_ws는 2024까지 존재 → 2024 fold 검증 가능)
        vd = df[va_idx & df.site_ws.notna()]
        if len(vd) > 100:
            c_mos = np.corrcoef(vd.ws_hat, vd.site_ws)[0, 1]
            c_raw = np.corrcoef(vd.ldaps_ws50max, vd.site_ws)[0, 1]
            rmse_mos = np.sqrt(((vd.ws_hat - vd.site_ws) ** 2).mean())
            print(f"fold {fold}: MOS r={c_mos:.3f} vs raw r={c_raw:.3f} | RMSE {rmse_mos:.2f}m/s", flush=True)
        df["gap_l"] = df["ws_hat"] - df["ldaps_ws50max"]
        df["gap_g"] = df["ws_hat"] - df["gfs_ws100"]
        for g in ("g1", "g2", "g3"):
            df[f"gapl_{g}"] = df[f"hat_{g}"] - df["ldaps_ws50max"]
        df["leg_diff"] = df["hat_g1"] - df["hat_g2"]  # 다리 간 바람비 (방향·웨이크 신호)

        COLSET = {"base": cols0,
                  "mosg": cols0 + ["ws_hat", "gap_l", "gap_g"],
                  "mg3": cols0 + ["hat_g1", "hat_g2", "hat_g3",
                                   "gapl_g1", "gapl_g2", "gapl_g3", "leg_diff"]}
        tr = df[tr_idx]
        va = df[va_idx]
        models = {}
        for v in VARIANTS:
            cols = COLSET[v]
            X, y, w = stack_groups(tr, cols, w_q3[tr_idx.to_numpy()])
            shared = cols + ["g_rated", "g_rotor", "g_id"]
            models[v] = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS).fit(
                X[shared], y, sample_weight=w) for q in QUANTILES_FULL}
        print(f"fold {fold}: 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

        fs = {v: [] for v in VARIANTS}
        dc = {v: [] for v in VARIANTS}
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm").dropna(subset=ANEN_FEATS)
            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=200).fit(
                ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            dist, aidx = knn.kneighbors(((sub[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            anen200 = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            actual = sub[tgt].to_numpy()
            d_bar = dist[:, :150].mean(axis=1)
            terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)

            for v in VARIANTS:
                cols = COLSET[v]
                Xv = group_X(sub, cols, tgt)
                shared = cols + ["g_rated", "g_rotor", "g_id"]
                qp = np.column_stack([np.clip(models[v][q].predict(Xv[shared]) * cap, 0, cap)
                                      for q in QUANTILES_FULL])
                qp.sort(axis=1)
                qp = np.sort(smooth_quantiles_by_day(qp, sub["forecast_kst_dtm"]), axis=1)
                gbm300 = interp_atoms(qp, n=300)
                med = np.median(gbm300, axis=1)
                atoms = np.empty((len(sub), 300))
                for i in range(len(sub)):
                    na = NA_OFF[terc[i]]
                    ng = 300 - na
                    an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                    an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                    gb = np.interp(np.linspace(0, 299, ng), np.arange(300), gbm300[i])
                    atoms[i] = np.sort(np.concatenate([an, gb]))
                pred = optimize_submission(atoms, cap, a_bar)
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

    print("\n=== v68b 그룹별 MOS — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
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
        print(f"{v:5s}: fold평균 {fm:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f} | 풀링 {np.mean(ps):.4f}")


if __name__ == "__main__":
    main()
