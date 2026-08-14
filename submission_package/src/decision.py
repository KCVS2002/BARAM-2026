"""FICR 기대점수 최적화 의사결정 레이어.

퀀타일 예측(시간별 예측분포)에서 대회 점수 기대값을 최대화하는 제출값을 선택한다.

시간별 기대 기여도 (유효시간 A >= 0.1C 조건부):
    J(g) = E[ -0.5 * |g-A|/C  +  0.5 * A * price(|g-A|/C) / (4 * A_bar) ]
    price: 오차율 <=6% -> 4, <=8% -> 3, else 0
    A_bar: 유효시간 평균 실제발전량 (학습 데이터에서 추정)

분포 근사: 퀀타일 값들을 등확률 원자(atom)로 취급, 0.1C 미만 원자는 평가 제외이므로 절단.
"""

import numpy as np
import pandas as pd

QLEVELS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])


def smooth_quantiles_by_day(qp: np.ndarray, forecast_dtm: pd.Series) -> np.ndarray:
    """같은 예보 사이클(대상일 01시~익일00시 블록) 내 3h 이동평균 평활화.

    OOF 실험(tune_decision.py)에서 +0.006 검증. 블록 경계를 넘지 않아 누수 없음.
    입력 행 순서는 forecast_dtm 오름차순이어야 한다.
    """
    dtm = pd.to_datetime(forecast_dtm).reset_index(drop=True)
    assert dtm.is_monotonic_increasing, "forecast_dtm must be sorted"
    block = (dtm - pd.Timedelta(hours=1)).dt.date
    out = pd.DataFrame(qp).groupby(block.values).transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean())
    return out.to_numpy()


def interp_atoms(qp: np.ndarray, n: int = 99, levels: np.ndarray = QLEVELS) -> np.ndarray:
    """퀀타일 함수 선형보간 → 등확률 원자 n개 (분포 근사 정밀화, OOF 검증 +0.003)."""
    p_new = (np.arange(n) + 0.5) / n
    out = np.empty((qp.shape[0], n))
    for i in range(qp.shape[0]):
        out[i] = np.interp(p_new, levels, qp[i])
    return out


def _expected_score(g: float, atoms: np.ndarray, cap: float, a_bar: float) -> float:
    err = np.abs(g - atoms) / cap
    price = np.where(err <= 0.06, 4.0, np.where(err <= 0.08, 3.0, 0.0))
    j = -0.5 * err + 0.5 * atoms * price / (4.0 * a_bar)
    return float(j.mean())


def optimize_submission(quantile_preds: np.ndarray, cap: float, a_bar: float,
                        n_grid: int = 61) -> np.ndarray:
    """시간별 최적 제출값.

    quantile_preds: (n_hours, n_quantiles) — 열은 퀀타일 레벨 오름차순, kWh 단위.
    반환: (n_hours,) 제출값.
    """
    n_hours = quantile_preds.shape[0]
    out = np.empty(n_hours)
    for i in range(n_hours):
        atoms = np.sort(quantile_preds[i])
        median = atoms[len(atoms) // 2]
        valid = atoms[atoms >= cap * 0.10]
        if len(valid) < 2:
            # 유효시간일 확률 낮음 -> 점수 기여 미미, 중앙값 제출
            out[i] = median
            continue
        # 후보: 유효 원자 범위 위 그리드 + 원자 자신들
        lo, hi = valid.min(), valid.max()
        cands = np.unique(np.concatenate([np.linspace(lo, hi, n_grid), valid, [median]]))
        scores = [_expected_score(g, valid, cap, a_bar) for g in cands]
        out[i] = cands[int(np.argmax(scores))]
    return np.clip(out, 0, cap)
