"""v20: g3 가용률 상태 혼합 원자 (OOF 캐시 기반, 재학습 없음).

가설: g3(UNISON 5기)의 병목 원인 중 하나는 부분정지(터빈당 20% 양자화, 실험 #40).
GBM은 이상운영을 다운웨이트(0.2)해 '완전 가용' 조건부 분포를 배우므로, 실제 분포의
하방 모드(4/5, 3/5 가동)가 원자 풀에 부족할 수 있다. AnEn이 일부 커버하지만
GBM 원자엔 없다 → GBM 원자에 k/5 스케일 사본을 소량 섞어 혼합분포화.

- 폭 조절(기각 4회)과 다름: 대칭 확대가 아니라 이산 하방 모드 추가 (비대칭·물리 기반).
- p 배분: 質量 p를 0.8배(4/5)와 0.6배(3/5)에 2:1로 배분, p ∈ {0.05, 0.10, 0.15} 그리드.
- 대조: 같은 처리를 g1/g2에도 적용해 g3 특이성인지 확인.
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


def load(fold, tgt):
    z = np.load(CACHE / f"{fold}_{tgt}.npz")
    return dict(actual=z["actual"], qp=z["qp"], anen=z["anen"],
                a_bar=float(z["a_bar"]), cap=float(z["cap"]))


def score(d, p_mix):
    gbm = interp_atoms(d["qp"], n=150)
    med = np.median(gbm, axis=1)
    anen = np.clip(d["anen"] + (med - np.median(d["anen"], axis=1))[:, None], 0, d["cap"])
    parts = [gbm, anen]
    if p_mix > 0:
        n = gbm.shape[1] + anen.shape[1]
        n8 = max(int(round(n * p_mix * 2 / 3)), 1)
        n6 = max(int(round(n * p_mix * 1 / 3)), 1)
        idx8 = np.linspace(0, gbm.shape[1] - 1, n8).astype(int)
        idx6 = np.linspace(0, gbm.shape[1] - 1, n6).astype(int)
        parts += [gbm[:, idx8] * 0.8, gbm[:, idx6] * 0.6]
    atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
    pred = optimize_submission(atoms, d["cap"], d["a_bar"])
    s, _, _, _ = metric_single(d["actual"], pred, d["cap"])
    return s


def main() -> None:
    for tgt in TARGET_COLS:
        print(f"\n=== {tgt} ===")
        for p in (0.0, 0.05, 0.10, 0.15):
            ss = [score(load(f, tgt), p) for f in FOLDS]
            wins = sum(1 for f, s0 in zip(FOLDS, ss)
                       if p > 0 and s0 > score(load(f, tgt), 0.0))
            tag = f" (fold승 {wins}/6)" if p > 0 else " (base)"
            print(f"p={p:.2f}: {np.mean(ss):.4f}{tag}", flush=True)


if __name__ == "__main__":
    main()
