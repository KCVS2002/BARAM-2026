"""#90 v57: a_bar 자기적응화 — 2024 상수를 예측 분포 기반 구조로 대체.

J의 FICR 정규화 상수 a_bar는 산식상 '평가 연도'의 유효시간 평균 에너지인데,
현행은 학습기간 측정치(2024 상수)를 이식 중. 2025 강풍 해에는 ā_2025 > ā_train
→ FICR 항 과대평가 (#54 α=0.7 실패 방향과 정합). 대체: 평가 기간의 원자에서
ā를 자기추정 — ā_hat = Σ_t E[a·1(valid)]_t / Σ_t P(valid)_t (시간별 원자 적분).
2024 측정 무의존 구조 (전이 성공 클래스), 연도 가변량 정렬 (증폭 여지).

변형: cur(ā_train, 현행) / selfbar(ā_hat) / oracle(검증기간 실측 ā — 축 상한).
+ 진단: fold별 ā_train vs ā_hat vs ā_actual — 추정 품질과 이득의 상관 확인.
캐시(oof_cache_q3) 기반 — 재학습 없음. 채점: fold평균 + 풀링 병기.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
FOLDS = ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"]
NA = {2: 175, 1: 150, 0: 125}
N_TOTAL = 300
VARIANTS = ["cur", "selfbar", "oracle"]


def optimize_bagged(atoms, cap, a_bar, B=15, frac=0.6, seed=42):
    rng = np.random.default_rng(seed)
    n = atoms.shape[1]
    k = int(n * frac)
    gs = np.empty((atoms.shape[0], B))
    for b in range(B):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        gs[:, b] = optimize_submission(atoms[:, idx], cap, a_bar)
    return np.clip(gs.mean(axis=1), 0, cap)


def build_atoms(qp, anen, dist, cap):
    gbm300 = interp_atoms(qp, n=N_TOTAL)
    med = np.median(gbm300, axis=1)
    d_bar = dist[:, :150].mean(axis=1)
    terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
    out = np.empty((len(qp), N_TOTAL))
    for i in range(len(qp)):
        na = NA[terc[i]]
        ng = N_TOTAL - na
        an = np.interp(np.linspace(0, anen.shape[1] - 1, na), np.arange(anen.shape[1]), anen[i])
        an = np.clip(an + (med[i] - np.median(an)), 0, cap)
        gb = np.interp(np.linspace(0, N_TOTAL - 1, ng), np.arange(N_TOTAL), gbm300[i])
        out[i] = np.sort(np.concatenate([an, gb]))
    return out


def a_bar_self(atoms, cap):
    """원자 분포에서 평가기간 유효시간 평균 에너지 추정."""
    valid = atoms >= 0.1 * cap
    p_valid = valid.mean(axis=1)
    e_valid = np.where(valid, atoms, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    m = p_valid > 0
    return float((e_valid[m] * p_valid[m]).sum() / p_valid[m].sum())


def main() -> None:
    t0 = time.time()
    fold_scores = {v: {} for v in VARIANTS}
    pooled = {v: {t: {"a": [], "p": []} for t in TARGET_COLS} for v in VARIANTS}

    for fold in FOLDS:
        fs = {v: [] for v in VARIANTS}
        diag = []
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp, anen, dist = z["qp"], z["anen"], z["dist"]
            cap, a_bar_tr, actual = float(z["cap"]), float(z["a_bar"]), z["actual"]
            atoms = build_atoms(qp, anen, dist, cap)
            a_hat = a_bar_self(atoms, cap)
            a_act = float(actual[actual >= 0.1 * cap].mean())
            diag.append((tgt, a_bar_tr, a_hat, a_act))
            abars = {"cur": a_bar_tr, "selfbar": a_hat, "oracle": a_act}
            for v in VARIANTS:
                pred = optimize_bagged(atoms, cap, abars[v])
                s_, nm, fi, _ = metric_single(actual, pred, cap)
                fs[v].append(s_)
                pooled[v][tgt]["a"].append(actual)
                pooled[v][tgt]["p"].append(pred)
        for v in VARIANTS:
            fold_scores[v][fold] = np.nanmean(fs[v])
        dstr = " ".join(f"{t[-1]}:tr{d[1]:.0f}/hat{d[2]:.0f}/act{d[3]:.0f}"
                        for t, d in zip(TARGET_COLS, diag))
        print(f"fold {fold}: " + " ".join(f"{v}={fold_scores[v][fold]:.4f}" for v in VARIANTS)
              + f" | ā {dstr} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== #90 v57 a_bar 자기적응화 — fold평균 | 풀링 (총/1-NMAE/FICR) ===")
    for v in VARIANTS:
        fm = np.mean(list(fold_scores[v].values()))
        ps, pn, pf = [], [], []
        for tgt in TARGET_COLS:
            a = np.concatenate(pooled[v][tgt]["a"])
            p = np.concatenate(pooled[v][tgt]["p"])
            s_, nm, fi, _ = metric_single(a, p, CAPACITY_KWH[tgt])
            ps.append(s_); pn.append(nm); pf.append(fi)
        print(f"{v:8s} fold평균 {fm:.4f} | 풀링 {np.mean(ps):.4f} "
              f"(1-NMAE {np.mean(pn):.4f} / FICR {np.mean(pf):.4f})")


if __name__ == "__main__":
    main()
