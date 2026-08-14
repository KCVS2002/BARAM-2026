"""의사결정 레이어 실험 (OOF 퀀타일 캐시 기반, 재학습 없음).

변형:
  base     : 현행 optimize_submission (재현 확인용)
  interp   : 퀀타일 함수 선형보간으로 원자 19→99개 (분포 근사 정밀화)
  calib    : 교차 fold 퀀타일 캘리브레이션(커버리지 보정) 후 interp
  smooth   : 인접 시간 퀀타일 이동평균(3h, 같은 날 블록 내) 후 interp
  smooth+calib : 둘 다
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import optimize_submission
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

QLEVELS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
QCOLS = [f"q{j}" for j in range(len(QLEVELS))]


def interp_atoms(qp: np.ndarray, n: int = 99) -> np.ndarray:
    """퀀타일 함수 선형보간 → 등확률 원자 n개."""
    p_new = (np.arange(n) + 0.5) / n
    out = np.empty((qp.shape[0], n))
    for i in range(qp.shape[0]):
        out[i] = np.interp(p_new, QLEVELS, qp[i])
    return out


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tgt in TARGET_COLS:
        sub = df[df.target == tgt]
        cov = [(sub.actual <= sub[c]).mean() for c in QCOLS]
        rows.append(pd.Series(cov, index=QLEVELS, name=tgt))
    return pd.DataFrame(rows)


def calibrate(df: pd.DataFrame) -> dict:
    """교차 fold 캘리브레이션: 각 fold의 보정맵은 '다른 fold들'의 커버리지로 산출.

    보정: 명목레벨 p에서 실제 커버리지 c(p)를 측정 → 목표 p를 갖는 보정레벨
    p' = c^{-1}(p)를 역보간으로 구해, 예측 퀀타일 함수를 p'에서 재평가.
    """
    maps = {}
    for fold in df.fold.unique():
        other = df[df.fold != fold]
        for tgt in TARGET_COLS:
            sub = other[other.target == tgt]
            cov = np.array([(sub.actual <= sub[c]).mean() for c in QCOLS])
            cov = np.maximum.accumulate(cov)  # 단조화
            # 목표 p에 대해 c(p')=p 가 되는 p' 탐색
            p_adj = np.interp(QLEVELS, cov, QLEVELS, left=QLEVELS[0], right=QLEVELS[-1])
            maps[(fold, tgt)] = p_adj
    return maps


def apply_calib(qp: np.ndarray, p_adj: np.ndarray) -> np.ndarray:
    out = np.empty_like(qp)
    for i in range(qp.shape[0]):
        out[i] = np.interp(p_adj, QLEVELS, qp[i])
    return out


def smooth_time(sub: pd.DataFrame) -> np.ndarray:
    """같은 날(사이클) 블록 내 3h 이동평균으로 퀀타일 평활화."""
    sub = sub.sort_values("forecast_kst_dtm")
    block = (pd.to_datetime(sub.forecast_kst_dtm) - pd.Timedelta(hours=1)).dt.date
    qp = sub[QCOLS].to_numpy().copy()
    sm = sub[QCOLS].groupby(block.values).transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean())
    return sm.to_numpy(), sub.index


def evaluate(df: pd.DataFrame, variant: str, calib_maps=None) -> float:
    scores = []
    for fold in sorted(df.fold.unique()):
        fold_scores = []
        for tgt in TARGET_COLS:
            sub = df[(df.fold == fold) & (df.target == tgt)]
            cap = CAPACITY_KWH[tgt]
            a_bar = sub.a_bar_train.iloc[0]
            actual = sub.actual.to_numpy()
            qp = sub[QCOLS].to_numpy()

            if "smooth" in variant:
                qp_sm, order = smooth_time(sub)
                # sub 순서 유지 위해 재정렬된 actual 사용
                actual = sub.loc[order, "actual"].to_numpy()
                qp = qp_sm
            if "calib" in variant:
                qp = apply_calib(qp, calib_maps[(fold, tgt)])
                qp.sort(axis=1)
            if variant != "base":
                atoms = interp_atoms(qp)
            else:
                atoms = qp
            pred = optimize_submission(atoms, cap, a_bar)
            s, _, _, _ = metric_single(actual, pred, cap)
            fold_scores.append(s)
        scores.append(np.nanmean(fold_scores))
    return float(np.mean(scores))


def main() -> None:
    df = pd.read_parquet(PROJECT / "experiments" / "oof_quantiles.parquet")
    print("=== 퀀타일 커버리지 (명목 vs 실측) ===")
    print(coverage_report(df).round(3).to_string())

    maps = calibrate(df)
    results = {}
    for variant in ["base", "interp", "calib", "smooth", "smooth_calib"]:
        results[variant] = evaluate(df, variant, maps)
        print(f"{variant}: {results[variant]:.4f}")

    best = max(results, key=results.get)
    print(f"\nbest: {best} = {results[best]:.4f} (기준 v2b: 0.6283)")


if __name__ == "__main__":
    main()
