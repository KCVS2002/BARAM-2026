"""v33: 실현 적중 분류기 게이트 (two-head 아이디어의 재구성 — 다른 각도 2차 시도).

- 'FICR 헤드' = 값 회귀(v32, 실패)가 아니라 **베팅 적중 확률 분류기**:
  원자 분포가 원리적으로 모르는 정보(파이프라인 결정의 실현 적중 기록 OOF)를 학습.
- 후보 2개/시간: g_opt(현행 최적화기), med(중앙값). 각 후보에 대해
  피처(베팅거리, 원자산포, 암시확률, 계절/시각, 풍속 수준)로 P(6%적중)·P(8%적중) 예측.
- 행동: J_clf(g) = -0.5·E_atoms|g-A|/C + 0.5·â·(4P6+3(P8-P6))/(4Ā) 비교, 큰 쪽 제출.
- 시간 존중: fold f 게이트는 이전 fold들의 OOF로만 학습 (fold1 제외, base 동일 집합 비교).
기준: 캐시 base (동일 fold 부분집합).
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache"
FOLDS = ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"]
GATE_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=100, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=42, verbose=-1)


def build_rows(fold: str, tgt: str):
    """fold×그룹의 후보별 행 (피처 + 라벨 + 채점 재료)."""
    z = np.load(CACHE / f"{fold}_{tgt}.npz")
    cap, a_bar, a = float(z["cap"]), float(z["a_bar"]), z["actual"]
    dtm = pd.to_datetime(z["dtm"])
    gbm = interp_atoms(z["qp"], n=150)
    med_g = np.median(gbm, axis=1)
    anen = np.clip(z["anen"] + (med_g - np.median(z["anen"], axis=1))[:, None], 0, cap)
    atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
    g_opt = optimize_submission(atoms, cap, a_bar)
    med = np.median(atoms, axis=1)
    a_hat = atoms.mean(axis=1)
    spread = (atoms[:, int(0.9 * atoms.shape[1])] - atoms[:, int(0.1 * atoms.shape[1])]) / cap
    rows = []
    for cand_name, g in (("opt", g_opt), ("med", med)):
        e_at = np.abs(g[:, None] - atoms) / cap
        p6 = (e_at <= 0.06).mean(axis=1)
        p8 = (e_at <= 0.08).mean(axis=1)
        err = np.abs(g - a) / cap
        df = pd.DataFrame({
            "fold": fold, "tgt": tgt, "cand": cand_name, "dtm": dtm,
            "g": g, "cap": cap, "a_bar": a_bar, "actual": a, "a_hat": a_hat,
            "bet_dist": (g - med) / cap, "spread": spread,
            "imp6": p6, "imp8": p8, "level": g / cap,
            "month_sin": np.sin(2 * np.pi * dtm.month / 12),
            "month_cos": np.cos(2 * np.pi * dtm.month / 12),
            "hour_sin": np.sin(2 * np.pi * dtm.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dtm.hour / 24),
            "eabs_atoms": np.abs(g[:, None] - atoms).mean(axis=1) / cap,
            "hit6": (err <= 0.06).astype(int), "hit8": (err <= 0.08).astype(int),
            "eval_h": (a >= 0.1 * cap).astype(int),
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


FEATS = ["bet_dist", "spread", "imp6", "imp8", "level", "eabs_atoms",
         "month_sin", "month_cos", "hour_sin", "hour_cos"]


def main() -> None:
    t0 = time.time()
    data = {f: pd.concat([build_rows(f, t) for t in TARGET_COLS], ignore_index=True)
            for f in FOLDS}
    print(f"OOF 후보 데이터 구축 완료 ({time.time()-t0:.0f}s)", flush=True)

    res = {v: {} for v in ("base", "gate")}
    dec = {v: {} for v in ("base", "gate")}
    for fi, fold in enumerate(FOLDS):
        if fi == 0:
            continue
        train = pd.concat([data[f] for f in FOLDS[:fi]], ignore_index=True)
        trn = train[train.eval_h == 1]
        clf6 = lgb.LGBMClassifier(**GATE_PARAMS).fit(trn[FEATS], trn.hit6)
        clf8 = lgb.LGBMClassifier(**GATE_PARAMS).fit(trn[FEATS], trn.hit8)

        fs = {v: [] for v in res}
        dc = {v: [] for v in res}
        for tgt in TARGET_COLS:
            d = data[fold]
            d = d[d.tgt == tgt]
            opt = d[d.cand == "opt"].reset_index(drop=True)
            med = d[d.cand == "med"].reset_index(drop=True)
            cap, a_bar = float(opt.cap.iloc[0]), float(opt.a_bar.iloc[0])
            actual = opt.actual.to_numpy()

            def j_clf(c):
                p6 = clf6.predict_proba(c[FEATS])[:, 1]
                p8 = np.maximum(clf8.predict_proba(c[FEATS])[:, 1], p6)
                eprice = 4 * p6 + 3 * (p8 - p6)
                return (-0.5 * c.eabs_atoms.to_numpy()
                        + 0.5 * c.a_hat.to_numpy() * eprice / (4 * a_bar))

            take_opt = j_clf(opt) >= j_clf(med)
            g_gate = np.where(take_opt, opt.g.to_numpy(), med.g.to_numpy())
            for v, g in (("base", opt.g.to_numpy()), ("gate", g_gate)):
                s, nm, fi_v, _ = metric_single(actual, g, cap)
                fs[v].append(s)
                dc[v].append((nm, fi_v))
        for v in res:
            res[v][fold] = np.nanmean(fs[v])
            dec[v][fold] = (np.nanmean([x[0] for x in dc[v]]), np.nanmean([x[1] for x in dc[v]]))
        n_sw = int((~take_opt).sum())
        print(f"fold {fold}: base={res['base'][fold]:.4f} gate={res['gate'][fold]:.4f} "
              f"| FICR {dec['base'][fold][1]:.3f}→{dec['gate'][fold][1]:.3f} "
              f"(마지막 그룹 med 전환 {n_sw}시간) ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== v33 실현 적중 게이트 (fold≥2) ===")
    for v in res:
        nm = np.mean([dec[v][f][0] for f in res[v]])
        fi_v = np.mean([dec[v][f][1] for f in res[v]])
        print(f"{v}: 총 {np.mean(list(res[v].values())):.4f} | 1-NMAE {1-nm:.4f} | FICR {fi_v:.4f}")


if __name__ == "__main__":
    main()
