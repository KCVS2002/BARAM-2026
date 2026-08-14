"""v21: GBM 분위 캘리브레이션 진단 + 시간순 conformal 재보정 (OOF 캐시 기반).

1부 진단: 19개 분위 예측의 경험적 커버리지 P(actual ≤ q_τ)를 그룹별로 측정
  (FICR 유효시간대: actual ≥ 10% cap). 명목 τ와의 괴리 = 분포 모양의 체계 오차.
2부 재보정: fold t의 분위를 이전 fold들의 커버리지 곡선으로 리매핑(τ→τ')한 뒤
  전체 파이프라인(원자 결합→FICR 최적화) 점수 비교. 첫 fold는 보정 불가라 제외하고
  base도 동일 fold 집합으로 비교.

주의: '외부 불확실성 주입'(4회 기각)과 다름 — 외부 스칼라로 폭을 조절하는 게 아니라
자기 자신의 실측 커버리지 오차를 되먹임. 단, 최근성 이득 부풀림(v8d) 주의 대상.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache"
FOLDS = ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"]
TAUS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])


def load(fold, tgt):
    z = np.load(CACHE / f"{fold}_{tgt}.npz")
    return dict(dtm=z["dtm"], actual=z["actual"], qp=z["qp"], anen=z["anen"],
                a_bar=float(z["a_bar"]), cap=float(z["cap"]))


def coverage(qp, actual, cap):
    m = actual >= 0.10 * cap
    return (actual[m, None] <= qp[m]).mean(axis=0)


def remap_qp(qp, cov_cal):
    """cov_cal: 보정셋에서 잰 커버리지(19,). 목표 τ에 대해 cov(τ')=τ인 τ'를 역보간해
    각 행의 분위곡선을 τ'에서 재평가."""
    cov_mono = np.maximum.accumulate(np.clip(cov_cal, 1e-4, 1 - 1e-4))
    tau_new = np.interp(TAUS, cov_mono, TAUS)  # cov(τ')=τ ← (cov_i, τ_i) 역보간
    out = np.empty_like(qp)
    for i in range(qp.shape[0]):
        out[i] = np.interp(tau_new, TAUS, qp[i])
    return np.sort(out, axis=1)


def score(d, qp_use):
    gbm = interp_atoms(np.sort(qp_use, axis=1), n=150)
    med = np.median(gbm, axis=1)
    anen = np.clip(d["anen"] + (med - np.median(d["anen"], axis=1))[:, None], 0, d["cap"])
    atoms = np.sort(np.concatenate([gbm, anen], axis=1), axis=1)
    pred = optimize_submission(atoms, d["cap"], d["a_bar"])
    s, _, _, _ = metric_single(d["actual"], pred, d["cap"])
    return s


def main() -> None:
    print("=== 1부: 커버리지 진단 (유효시간대, fold 풀링) ===")
    print("τ      :", " ".join(f"{t:.2f}" for t in TAUS))
    cov_all = {}
    for tgt in TARGET_COLS:
        covs = []
        for fold in FOLDS:
            d = load(fold, tgt)
            covs.append(coverage(d["qp"], d["actual"], d["cap"]))
        cov_all[tgt] = np.array(covs)
        print(f"{tgt}:", " ".join(f"{c:.2f}" for c in cov_all[tgt].mean(axis=0)),
              f"| MAE(τ괴리)={np.abs(cov_all[tgt].mean(axis=0)-TAUS).mean():.3f}")

    print("\n=== 2부: 시간순 conformal 재보정 (fold≥2) ===")
    res_b, res_c = {}, {}
    for fi, fold in enumerate(FOLDS):
        if fi == 0:
            continue
        sb, sc = [], []
        for tgt in TARGET_COLS:
            d = load(fold, tgt)
            cal = [coverage((e := load(f2, tgt))["qp"], e["actual"], e["cap"])
                   for f2 in FOLDS[:fi]]
            qp_new = remap_qp(d["qp"], np.mean(cal, axis=0))
            sb.append(score(d, d["qp"]))
            sc.append(score(d, qp_new))
        res_b[fold], res_c[fold] = np.nanmean(sb), np.nanmean(sc)
        print(f"fold {fold}: base={res_b[fold]:.4f} recal={res_c[fold]:.4f}", flush=True)
    print(f"\nbase={np.mean(list(res_b.values())):.4f} recal={np.mean(list(res_c.values())):.4f}")


if __name__ == "__main__":
    main()
