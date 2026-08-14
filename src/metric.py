"""대회 공식 평가 산식 (평가_산식 코드.ipynb 그대로)."""

import numpy as np
import pandas as pd

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}


def metric(answer_df: pd.DataFrame, pred_df: pd.DataFrame):
    """총점, 1-NMAE, FICR 반환. 실제 발전량 >= 설비용량 10% 시간대만 평가."""
    group_nmae = []
    group_ficr = []

    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[col]

        valid = actual >= capacity * 0.10
        actual = actual[valid]
        forecast = forecast[valid]

        error_rate = np.abs(forecast - actual) / capacity
        group_nmae.append(np.mean(error_rate))

        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        earned = np.sum(actual * unit_price)
        max_settlement = np.sum(actual * 4.0)
        group_ficr.append(earned / max_settlement)

    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    total = 0.5 * one_minus_nmae + 0.5 * ficr
    return total, one_minus_nmae, ficr


def metric_single(actual: np.ndarray, forecast: np.ndarray, capacity: float):
    """단일 그룹용 (라벨 NaN 제거 후 사용). 총점 구성요소 반환."""
    valid = actual >= capacity * 0.10
    a, f = actual[valid], forecast[valid]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, 0
    err = np.abs(f - a) / capacity
    nmae = err.mean()
    price = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    ficr = (a * price).sum() / (a * 4.0).sum()
    return 0.5 * (1 - nmae) + 0.5 * ficr, 1 - nmae, ficr, int(len(a))
