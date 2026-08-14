"""캐시 기반 제출 생성기 — 재학습 0회, 결정층만 계산 (분 단위).

experiments/test_cache/의 불변 재료(qp/anen/dist/a_bar/sisq)로 공식(sub_029)
원자 구성·결정 배깅을 재현한다. 기본 실행은 검증 모드: 재현 결과를
submissions/sub_029_q3.csv와 대조해 캐시 파이프라인의 무결성을 확인.

사용:
    python scripts/make_submission_from_cache.py            # sub_029 재현·검증
    python scripts/make_submission_from_cache.py out.csv    # 재현 결과를 저장

향후 원자 구성 변경 제출은 이 파일을 복제해 조립부만 수정한다 (캐시 우선 규칙).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import TARGET_COLS

CACHE = PROJECT / "experiments" / "test_cache"
K = 150
P_MIX = 0.15
NA = {0: 125, 1: 150, 2: 175}


def optimize_bagged(atoms, cap, a_bar, B=15, frac=0.6, seed=42):
    rng = np.random.default_rng(seed)
    n = atoms.shape[1]
    k = int(n * frac)
    gs = np.empty((atoms.shape[0], B))
    for b in range(B):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        gs[:, b] = optimize_submission(atoms[:, idx], cap, a_bar)
    return np.clip(gs.mean(axis=1), 0, cap)


def build_official(tgt: str) -> np.ndarray:
    """공식(sub_029) 구성: gbm300+AnEn(NA 3분위) [+g3: 가용률 혼합+sister 조건부]."""
    z = np.load(CACHE / f"test_atoms_{tgt}.npz")
    qp, anen200, dist = z["qp"], z["anen"], z["dist"]
    a_bar, cap = float(z["a_bar"]), float(z["cap"])
    n_row = len(qp)
    gbm300 = interp_atoms(qp, n=2 * K)
    gbm_atoms = gbm300[:, np.linspace(0, 2 * K - 1, K).astype(int)]
    med = np.median(gbm_atoms, axis=1)
    d_bar = dist[:, :K].mean(axis=1)
    terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
    base_atoms = np.empty((n_row, 2 * K))
    for i in range(n_row):
        na = NA[terc[i]]
        ng = 2 * K - na
        an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
        base_atoms[i] = np.sort(np.concatenate([an, gb]))
    parts = [base_atoms]

    if tgt == "kpx_group_3":
        n = 2 * K
        n8 = max(int(round(n * P_MIX * 2 / 3)), 1)
        n6 = max(int(round(n * P_MIX * 1 / 3)), 1)
        parts += [gbm_atoms[:, np.linspace(0, K - 1, n8).astype(int)] * 0.8,
                  gbm_atoms[:, np.linspace(0, K - 1, n6).astype(int)] * 0.6]
        sis300 = interp_atoms(z["sisq"], n=2 * K)
        fixed = np.concatenate(parts, axis=1)
        g_out = np.empty(n_row)
        for t_, ns in ((0, 125), (1, 150), (2, 175)):
            m = terc == t_
            sis = np.array([np.interp(np.linspace(0, 2 * K - 1, ns),
                                      np.arange(2 * K), s) for s in sis300[m]])
            atoms_g = np.sort(np.concatenate([fixed[m], sis], axis=1), axis=1)
            g_out[m] = optimize_bagged(atoms_g, cap, a_bar)
        return g_out
    atoms = np.sort(np.concatenate(parts, axis=1), axis=1)
    return optimize_bagged(atoms, cap, a_bar)


def main() -> None:
    t0 = time.time()
    ref = pd.read_csv(PROJECT / "submissions" / "sub_029_q3.csv", encoding="utf-8-sig")
    out = ref[["forecast_id", "forecast_kst_dtm"]].copy()
    for tgt in TARGET_COLS:
        out[tgt] = build_official(tgt)
        d = np.abs(out[tgt].to_numpy() - ref[tgt].to_numpy())
        print(f"{tgt}: 재현 완료 — sub_029 대비 |Δ| max {d.max():.2f} / mean {d.mean():.4f} kWh"
              f" ({time.time()-t0:.0f}s)", flush=True)
    if len(sys.argv) > 1:
        path = PROJECT / "submissions" / sys.argv[1]
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved {path.name}")
    print(f"총 소요 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
