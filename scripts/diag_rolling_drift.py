"""#106 Phase 0: 롤링 관측 보정의 설계 진단 (게이트 아님 — 설계 정보 수집).

측정:
0-1 관측 정합성: 지점별·연도별 커버리지 (2022~2025)
0-2 대표성: 각 지점 관측 ws vs 단지 LDAPS ws50max의 상관 (시간별)
0-3 드리프트 특성: 괴리(관측−예보)의 연/분기 이동 + **편향 지속성**
    (직전 W일 평균 괴리 → 다음날 괴리 예측력, W=7/14/28) — 롤링 보정의
    착취 가능 신호 크기. 가법(차이)·승법(비율) 두 형태 병기.
누수 근거: AWS/ASOS는 관측 즉시 공개 (수집기 문서) — D-1 13:00 이전 관측만
사용하는 롤링 프로토콜에서 안전.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "Data"

STN_NAME = {216: "태백ASOS(712m)", 100: "대관령", 116: "?116", 314: "?314(1500m)",
             315: "매봉산(1088m)", 316: "함백산(912m)", 320: "백운산(1264m)", 45: "?45"}


def farm_ldaps_ws() -> pd.Series:
    """단지 16격자 평균 ws50max (2022~2025 전체)."""
    parts = []
    for f in ["train/ldaps_train.csv", "test/ldaps_test.csv"]:
        d = pd.read_csv(DATA / f, encoding="utf-8-sig",
                        usecols=["forecast_kst_dtm", "heightAboveGround_50_50MUmax",
                                 "heightAboveGround_50_50MVmax"],
                        parse_dates=["forecast_kst_dtm"])
        ws = np.hypot(d["heightAboveGround_50_50MUmax"], d["heightAboveGround_50_50MVmax"])
        parts.append(ws.groupby(d["forecast_kst_dtm"]).mean())
    return pd.concat(parts).sort_index()


def main() -> None:
    obs = pd.read_csv(PROJECT / "external_data/kma_aws/aws_hourly_merged.csv",
                      encoding="utf-8-sig", parse_dates=["tm"])
    # KMA 결측 센티널(-99 등) 제거 — 실측 -99가 상관·괴리 전부 오염 (실측 확인)
    obs.loc[obs["ws"] < 0, "ws"] = np.nan
    obs.loc[obs["ws"] > 60, "ws"] = np.nan
    ldws = farm_ldaps_ws()
    print(f"LDAPS 단지 ws: {ldws.index.min()} ~ {ldws.index.max()} ({len(ldws)}시간)")

    print("\n[0-1] 지점별 연도 커버리지 (ws 비결측 %)")
    for stn, g in obs.groupby("stn"):
        s = g.set_index("tm")["ws"]
        cov = s.notna().groupby(s.index.year).mean()
        full = pd.date_range("2022-01-01", "2025-12-31 23:00", freq="h")
        n_hours = s.reindex(full).notna().mean()
        print(f"  {stn} {STN_NAME.get(stn, '?'):14s}: 전체 {n_hours*100:5.1f}% | "
              + " ".join(f"{y}:{v*100:.0f}%" for y, v in cov.items()))

    print("\n[0-2] 대표성: corr(관측 ws, LDAPS 단지 ws) — 시간별, 전 기간")
    reps = {}
    for stn, g in obs.groupby("stn"):
        s = g.set_index("tm")["ws"]
        j = pd.concat([s, ldws], axis=1, join="inner").dropna()
        j.columns = ["obs", "ld"]
        if len(j) < 5000:
            continue
        r = j["obs"].corr(j["ld"])
        reps[stn] = (r, len(j))
        print(f"  {stn} {STN_NAME.get(stn, '?'):14s}: r={r:.3f} (n={len(j)})")

    print("\n[0-3a] 괴리의 연도별 이동 (가법: 관측−LDAPS / 승법: 관측/LDAPS, 일평균 기준)")
    for stn in sorted(reps, key=lambda s: -reps[s][0])[:4]:
        g = obs[obs.stn == stn].set_index("tm")["ws"]
        j = pd.concat([g, ldws], axis=1, join="inner").dropna()
        j.columns = ["obs", "ld"]
        day = j.resample("D").mean()
        day = day[day["ld"] > 2]  # 무풍일 비율 왜곡 방지
        gap_a = day["obs"] - day["ld"]
        gap_m = day["obs"] / day["ld"]
        ya = gap_a.groupby(gap_a.index.year).mean()
        ym = gap_m.groupby(gap_m.index.year).mean()
        print(f"  {stn} {STN_NAME.get(stn, '?'):14s} 가법: "
              + " ".join(f"{y}:{v:+.2f}" for y, v in ya.items())
              + " | 승법: " + " ".join(f"{y}:{v:.3f}" for y, v in ym.items()))
        g25 = gap_a[gap_a.index.year == 2025]
        if len(g25):
            qm = g25.groupby(g25.index.quarter).mean()
            print(f"      2025 분기별(가법): " + " ".join(f"Q{k}:{v:+.2f}" for k, v in qm.items()))

    print("\n[0-3b] 편향 지속성: corr(직전 W일 평균 괴리, 당일 괴리) — 일 단위, 전 기간/2025")
    for stn in sorted(reps, key=lambda s: -reps[s][0])[:4]:
        g = obs[obs.stn == stn].set_index("tm")["ws"]
        j = pd.concat([g, ldws], axis=1, join="inner").dropna()
        j.columns = ["obs", "ld"]
        day = j.resample("D").mean()
        day = day[day["ld"] > 2]
        gap = (day["obs"] - day["ld"]).asfreq("D")
        line = f"  {stn} {STN_NAME.get(stn, '?'):14s}: "
        for W in (7, 14, 28):
            trail = gap.rolling(W, min_periods=max(3, W // 2)).mean().shift(1)
            m = trail.notna() & gap.notna()
            r_all = trail[m].corr(gap[m])
            m25 = m & (gap.index.year == 2025)
            r_25 = trail[m25].corr(gap[m25]) if m25.sum() > 30 else float("nan")
            line += f"W{W}: 전기간 r={r_all:.3f}/2025 r={r_25:.3f}  "
        print(line)


if __name__ == "__main__":
    main()
