"""실험 v2: SCADA 라벨 정제 + 그룹 공유 학습.

(1) 라벨 정제: SCADA 실측 풍속 대비 발전량이 비정상적으로 낮은 시간(정지/출력제한 추정)을
    학습에서 다운웨이트. 파워커브 잔차 기반.
(2) 그룹 공유 학습: 3그룹을 capacity-factor 타깃으로 쌓아 단일 퀀타일 모델 학습
    (그룹 원-핫 + 터빈 스펙 피처). group_3의 학습량 부족 보완.

CV는 기존과 동일 (2024년 2개월 × 6 fold), FICR 최적화 포함 비교:
  v1(그룹별, 정제 없음) vs v2a(그룹별+정제) vs v2b(공유+정제)
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import optimize_submission
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# 2026-07-12 rand1 튜닝 채택했다가 LB 하락(-0.009)으로 철회 — 2024 CV↔2025 LB 분포이동.
# LB 검증된 설정 유지. (튜닝 재시도 시 무작위성 보존 설정 + LB 확인 후 확정)
BASE_PARAMS = dict(
    n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=42, verbose=-1,
)
GROUP_TURBINES = {
    "kpx_group_1": ("scada_vestas_train.csv", [f"vestas_wtg{i:02d}" for i in range(1, 7)]),
    "kpx_group_2": ("scada_vestas_train.csv", [f"vestas_wtg{i:02d}" for i in range(7, 13)]),
    "kpx_group_3": ("scada_unison_train.csv", [f"unison_wtg{i:02d}" for i in range(1, 6)]),
}
GROUP_META = {  # (정격 kW, 로터 m)
    "kpx_group_1": (3600, 126),
    "kpx_group_2": (3600, 126),
    "kpx_group_3": (4200, 136),
}


def label_weights(lab: pd.DataFrame) -> pd.DataFrame:
    """SCADA 파워커브 잔차로 이상운영 시간 다운웨이트 (기본 1.0, 이상 0.2)."""
    w = pd.DataFrame(1.0, index=lab.index, columns=TARGET_COLS)
    scada_cache = {}
    for tgt, (fname, turbines) in GROUP_TURBINES.items():
        if fname not in scada_cache:
            sc = pd.read_csv(DATA / "train" / fname, encoding="utf-8-sig", parse_dates=["kst_dtm"])
            sc["hour_end"] = sc["kst_dtm"].dt.ceil("h")
            scada_cache[fname] = sc
        sc = scada_cache[fname]
        ws_cols = [f"{t}_ws" for t in turbines]
        ws = sc[ws_cols].where((sc[ws_cols] >= 0) & (sc[ws_cols] < 60))
        farm_ws = ws.mean(axis=1)
        hourly_ws = farm_ws.groupby(sc["hour_end"]).mean()

        m = lab[["kst_dtm", tgt]].merge(hourly_ws.rename("ws"), left_on="kst_dtm", right_index=True)
        m = m.dropna()
        # 구간별(0.5 m/s bin) 발전량 90퍼센타일을 '정상 상한 곡선'으로
        bins = (m.ws / 0.5).round().astype(int)
        curve = m.groupby(bins)[tgt].transform(lambda s: s.quantile(0.9))
        # 정상풍속(>5m/s)인데 발전량이 곡선의 30% 미만 → 이상운영 추정
        anomal = (m.ws > 5) & (m[tgt] < 0.3 * curve)
        idx = m.index[anomal]
        w.loc[w.index.isin(lab.index[lab.index.isin(idx)]), tgt] = 1.0  # placeholder
        w.loc[idx, tgt] = 0.2
        print(f"{tgt}: anomalous hours {anomal.sum()} / {len(m)} ({anomal.mean()*100:.1f}%)")
    return w


def eval_fold(actual, qp, cap, a_bar):
    qp = np.sort(qp, axis=1)
    pred = optimize_submission(qp, cap, a_bar)
    return metric_single(actual, pred, cap)


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    ldaps = pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig")
    feat = build_features(ldaps, gfs)
    weights = label_weights(lab)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    feature_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    print(f"ready ({time.time()-t0:.0f}s)")

    folds = [(pd.Timestamp(2024, m0, 1, 1),) for m0 in range(1, 12, 2)]
    rows = []
    for (va_start,) in folds:
        va_end = va_start + pd.DateOffset(months=2)
        tr_idx = df.forecast_kst_dtm < va_start
        va_idx = (df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)
        tr, va = df[tr_idx], df[va_idx]
        w_tr = weights[tr_idx.to_numpy()]
        row = {"fold": f"{va_start:%Y-%m}"}

        # ---- v2b: 그룹 공유 학습 (정제 가중치 적용) ----
        stack_X, stack_y, stack_w = [], [], []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            trm = tr[tgt].notna()
            Xg = tr.loc[trm, feature_cols].copy()
            rated, rotor = GROUP_META[tgt]
            Xg["g_rated"], Xg["g_rotor"] = rated, rotor
            Xg["g_id"] = list(GROUP_META).index(tgt)
            stack_X.append(Xg)
            stack_y.append(tr.loc[trm, tgt] / cap)
            stack_w.append(w_tr.loc[trm.to_numpy(), tgt])
        X_all = pd.concat(stack_X)
        y_all = pd.concat(stack_y)
        w_all = pd.concat(stack_w)

        shared_cols = feature_cols + ["g_rated", "g_rotor", "g_id"]
        qmodels = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
            m.fit(X_all[shared_cols], y_all, sample_weight=w_all)
            qmodels.append(m)

        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            vam = va[tgt].notna()
            actual = va.loc[vam, tgt].to_numpy()
            Xv = va.loc[vam, feature_cols].copy()
            rated, rotor = GROUP_META[tgt]
            Xv["g_rated"], Xv["g_rotor"] = rated, rotor
            Xv["g_id"] = list(GROUP_META).index(tgt)
            qp = np.column_stack([
                np.clip(m.predict(Xv[shared_cols]) * cap, 0, cap) for m in qmodels
            ])
            a_tr = tr.loc[tr[tgt].notna(), tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()
            s, nm, fi, n = eval_fold(actual, qp, cap, a_bar)
            row[f"{tgt}_v2b"] = s

            # ---- v2a: 그룹별 + 정제 ----
            trm = tr[tgt].notna()
            qp2 = np.empty((int(vam.sum()), len(QUANTILES)))
            for j, q in enumerate(QUANTILES):
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS)
                m.fit(tr.loc[trm, feature_cols], tr.loc[trm, tgt] / cap,
                      sample_weight=w_tr.loc[trm.to_numpy(), tgt])
                qp2[:, j] = np.clip(m.predict(va.loc[vam, feature_cols]) * cap, 0, cap)
            s2, _, _, _ = eval_fold(actual, qp2, cap, a_bar)
            row[f"{tgt}_v2a"] = s2

        for v in ("v2a", "v2b"):
            row[f"score_{v}"] = np.nanmean([row[f"{t}_{v}"] for t in TARGET_COLS])
        rows.append(row)
        print(f"fold {row['fold']}: v2a={row['score_v2a']:.4f} v2b={row['score_v2b']:.4f} "
              f"({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    print("\n=== CV mean (참고: v1 opt = 0.6220) ===")
    for v in ("v2a", "v2b"):
        print(f"{v}: {res[f'score_{v}'].mean():.4f} (std {res[f'score_{v}'].std():.4f})")
    for t in TARGET_COLS:
        print(f"  {t}: v2a={res[f'{t}_v2a'].mean():.4f} v2b={res[f'{t}_v2b'].mean():.4f}")
    (PROJECT / "experiments").mkdir(exist_ok=True)
    res.to_csv(PROJECT / "experiments" / "v2_clean_shared_cv.csv", index=False, encoding="utf-8-sig")
    print(f"saved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
