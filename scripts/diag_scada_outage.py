"""#101 진단: SCADA 터빈별 정지/결손 탐지 — 라벨 정제(③)의 실현 가능성 측정.

그룹 합계만 보면 안 보이는 "일부 터빈 정지" 시간을 터빈별 SCADA로 탐지:
- 정지: 터빈 시간 발전량 < 정격의 1% AND 동료(같은 그룹 타 터빈) 중앙값 cf > 15%
  (동료가 잘 도는데 이 터빈만 0 = 바람이 아니라 터빈 문제)
- 부분 결손: 터빈 발전량 < 동료 중앙값의 40% AND 동료 중앙값 cf > 20%
출력: 터빈-시간 정지율 / 오염 시간 비중(전체·유효시간) / 에너지 결손 추정 /
현행 label_weights 이상시간과의 중첩 / 터빈합 vs 라벨 정합(송전단 손실).
집계: kst_dtm은 구간 종료 — 10분 스탬프 (H-1,H] → 시간 H. 에너지 = Σ(10분 평균 kW)/6.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v2_clean_shared import GROUP_TURBINES, label_weights
from src.metric import CAPACITY_KWH, TARGET_COLS

DATA = PROJECT / "Data"
RATED = {"kpx_group_1": 3600, "kpx_group_2": 3600, "kpx_group_3": 4200}


def hourly_turbines(scada_file, turbines, rated):
    d = pd.read_csv(DATA / "train" / scada_file, encoding="utf-8-sig", parse_dates=["kst_dtm"])
    # 시간 매핑: (H-1, H] → H (종료 시각 관례)
    hour = d["kst_dtm"].dt.ceil("h")
    pcols = [f"{t}_power_kw10m" for t in turbines]
    wcols = [f"{t}_ws" for t in turbines]
    # 센서 스파이크 제거: 10분 에너지 물리 한계 = 정격/6 × 1.15 (실측 4.5e7 kWh 스파이크 존재)
    d[pcols] = d[pcols].clip(lower=0, upper=rated / 6 * 1.15)
    g = d.groupby(hour)
    # power_kw10m는 10분 에너지(kWh) — 라벨/터빈합 비율 ≈6 실측으로 확인 → 그대로 합산
    energy = g[pcols].sum(min_count=3)
    nobs = g[pcols].count()
    ws = g[wcols].mean()
    energy.columns = turbines
    nobs.columns = turbines
    ws.columns = turbines
    return energy, nobs, ws


def main() -> None:
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    lab = lab.set_index("kst_dtm")
    clean_w = label_weights(lab.reset_index())

    for tgt in TARGET_COLS:
        scada_file, turbines = GROUP_TURBINES[tgt]
        cap = CAPACITY_KWH[tgt]
        rated = RATED[tgt]
        energy, nobs, ws = hourly_turbines(scada_file, turbines, rated)
        n_t = len(turbines)

        # 그룹 라벨과 정렬
        idx = energy.index.intersection(lab.index)
        E = energy.loc[idx]
        W = ws.loc[idx]
        y = lab.loc[idx, tgt]
        m = y.notna() & E.notna().all(axis=1)
        E, W, y = E[m], W[m], y[m]

        # 터빈 합 vs 라벨 (송전단 손실)
        tsum = E.sum(axis=1)
        hi = y > cap * 0.3
        ratio = (y[hi] / tsum[hi]).median()

        # 동료 중앙값 cf (자기 제외)
        peer_med = np.empty(E.shape)
        Ev = E.to_numpy()
        for j in range(n_t):
            others = np.delete(Ev, j, axis=1)
            peer_med[:, j] = np.median(others, axis=1)
        peer_cf = peer_med / rated  # 시간 에너지/정격 ≈ cf

        Ecf = Ev / rated
        out_mask = (Ecf < 0.01) & (peer_cf > 0.15)
        part_mask = (Ev < 0.4 * peer_med) & (peer_cf > 0.20) & ~out_mask

        n_hours = len(E)
        turb_out_rate = out_mask.mean()
        hrs_any_out = (out_mask.any(axis=1)).mean()
        hrs_any_part = ((out_mask | part_mask).any(axis=1)).mean()
        n_out_per_hr = out_mask.sum(axis=1)

        # 유효시간(라벨 cf>=10%) 기준
        valid = (y >= 0.1 * cap).to_numpy()
        hrs_out_valid = (out_mask.any(axis=1) & valid).sum() / max(valid.sum(), 1)
        hrs_part_valid = ((out_mask | part_mask).any(axis=1) & valid).sum() / max(valid.sum(), 1)

        # 에너지 결손 추정 (정지·부분 모두: 동료 중앙값 - 실제, 양수만)
        deficit = np.where(out_mask | part_mask, np.maximum(peer_med - Ev, 0), 0).sum(axis=1)
        deficit_frac = deficit[valid].sum() / max(y[valid].sum(), 1)

        # 현행 label_weights 이상시간과 중첩
        w_ser = clean_w.set_index(lab.index[:len(clean_w)])[tgt] if len(clean_w) == len(lab) else None
        if w_ser is not None:
            w_al = w_ser.reindex(E.index)
            anom_cur = (w_al < 1).to_numpy()
            both = (out_mask.any(axis=1) & anom_cur).sum()
            only_new = (out_mask.any(axis=1) & ~anom_cur).sum()
            cur_n = anom_cur.sum()
        else:
            both = only_new = cur_n = -1

        print(f"\n===== {tgt} ({n_t}기, SCADA {n_hours}시간) =====")
        print(f"라벨/터빈합 중앙 비율 (고출력시): {ratio:.4f}  (송전단 손실 {1-ratio:+.1%})")
        print(f"터빈-시간 정지율: {turb_out_rate*100:.2f}%")
        print(f"정지 ≥1기 시간: {hrs_any_out*100:.2f}% (전체) / {hrs_out_valid*100:.2f}% (유효시간)")
        print(f"정지+부분결손 ≥1기: {hrs_any_part*100:.2f}% (전체) / {hrs_part_valid*100:.2f}% (유효시간)")
        print(f"정지 기수 분포: 1기 {(n_out_per_hr==1).mean()*100:.2f}% / 2기+ {(n_out_per_hr>=2).mean()*100:.2f}%")
        print(f"유효시간 에너지 결손 추정: {deficit_frac*100:.2f}% (라벨 대비)")
        if cur_n >= 0:
            print(f"현행 label_weights 이상시간 {cur_n} vs SCADA 정지시간: 중첩 {both} / 신규 탐지 {only_new}")


if __name__ == "__main__":
    main()
