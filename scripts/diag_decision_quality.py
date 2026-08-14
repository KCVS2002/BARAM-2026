"""#96 A: 의사결정층 품질 감사 — "하고 있다"가 아니라 "잘하고 있는가".

조사(deep-research 2회) 이론 결론: 밴드형 보상에서 최적 point는 mean도 quantile도
아니며(주장 5·6·11), 비대칭+비선형에서 조건부 평균으로부터 계통 편향돼야 함(주장 13·14).
우리는 기대점수 argmax를 이미 하지만, 그 편향의 방향·크기가 옳은지, 사후 최적 대비
얼마나 남기는지는 미검증. 캐시(oof_cache_q3, 2024) 기반, 제출 0회.

측정:
(1) implied quantile: 우리 제출값 g_ours가 원자 분포의 몇 분위에 앉는가 (편향 진단)
(2) 고정분위 τ 스윕 (0.35~0.65): 사후 최적 τ vs 우리 g_ours 점수 — 겨누는 분위가 옳은가
(3) 사후 최적 편향: 각 시간 actual을 알 때 최선의 단일 원자 분위 → 우리 implied_q와 비교
(4) 기계 손실: 미세격자(n_grid=301) 기대최적 vs 현행(61) vs 배깅 — 해상도 여유
채점: metric_single 풀링 (그룹별 연결). 총점·1-NMAE·FICR 병기.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.decision import interp_atoms, optimize_submission, _expected_score
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
FOLDS = ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"]
NA = {2: 175, 1: 150, 0: 125}
N_TOTAL = 300


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


def optimize_fine(atoms, cap, a_bar, n_grid=301):
    """미세격자 기대점수 최적 (해상도 상한 진단)."""
    out = np.empty(len(atoms))
    for i in range(len(atoms)):
        a = atoms[i]
        valid = a[a >= 0.1 * cap]
        if len(valid) < 2:
            out[i] = a[len(a) // 2]
            continue
        cands = np.unique(np.concatenate([np.linspace(valid.min(), valid.max(), n_grid), valid]))
        scores = [_expected_score(g, valid, cap, a_bar) for g in cands]
        out[i] = cands[int(np.argmax(scores))]
    return np.clip(out, 0, cap)


def score_pool(preds_by_grp):
    """그룹별 (actual, pred) 리스트 → 풀링 총/NMAE/FICR."""
    ss, nn, ff = [], [], []
    for tgt in TARGET_COLS:
        a = np.concatenate([p[0] for p in preds_by_grp[tgt]])
        p = np.concatenate([p[1] for p in preds_by_grp[tgt]])
        s_, nm, fi, _ = metric_single(a, p, CAPACITY_KWH[tgt])
        ss.append(s_); nn.append(nm); ff.append(fi)
    return np.mean(ss), np.mean(nn), np.mean(ff)


def main() -> None:
    t0 = time.time()
    TAUS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    pool_ours = {t: [] for t in TARGET_COLS}
    pool_fine = {t: [] for t in TARGET_COLS}
    pool_tau = {tau: {t: [] for t in TARGET_COLS} for tau in TAUS}
    implied_qs = []          # 우리 제출이 앉는 분위
    hindsight_qs = []        # 사후 최적 단일원자 분위
    # 시간별 점수 기여 (우리 vs 최선고정분위) — 잠재 이득 크기
    for fold in FOLDS:
        for tgt in TARGET_COLS:
            z = np.load(CACHE / f"{fold}_{tgt}.npz")
            qp, anen, dist = z["qp"], z["anen"], z["dist"]
            cap, a_bar, actual = float(z["cap"]), float(z["a_bar"]), z["actual"]
            atoms = build_atoms(qp, anen, dist, cap)
            g_ours = optimize_bagged(atoms, cap, a_bar)
            g_fine = optimize_fine(atoms, cap, a_bar)
            pool_ours[tgt].append((actual, g_ours))
            pool_fine[tgt].append((actual, g_fine))
            # implied quantile of our pick
            iq = (atoms < g_ours[:, None]).mean(axis=1)
            implied_qs.append(iq)
            # fixed-quantile submissions
            for tau in TAUS:
                gq = np.quantile(atoms, tau, axis=1)
                pool_tau[tau][tgt].append((actual, np.clip(gq, 0, cap)))
            # hindsight: 각 시간 actual에 가장 가까운 원자의 분위 (유효시간만)
            vmask = actual >= 0.1 * cap
            if vmask.any():
                nearest = np.argmin(np.abs(atoms[vmask] - actual[vmask, None]), axis=1)
                hq = nearest / (atoms.shape[1] - 1)
                hindsight_qs.append(hq)
        print(f"fold {fold} 완료 ({time.time()-t0:.0f}s)", flush=True)

    iq = np.concatenate(implied_qs)
    hq = np.concatenate(hindsight_qs)
    so, no_, fo = score_pool(pool_ours)
    sf, nf, ff = score_pool(pool_fine)

    print("\n=== #96 의사결정층 품질 감사 ===")
    print(f"[현행 배깅 g_ours]  총 {so:.4f} | 1-NMAE {no_:.4f} | FICR {fo:.4f}")
    print(f"[미세격자 g_fine ]  총 {sf:.4f} | 1-NMAE {nf:.4f} | FICR {ff:.4f}  (기계 해상도 여유)")
    print("\n[고정분위 τ 스윕 — 사후 최적 τ 탐색]")
    best_tau, best_s = None, -1
    for tau in TAUS:
        s_, nm, fi = score_pool(pool_tau[tau])
        star = " *" if s_ > best_s else ""
        if s_ > best_s: best_s, best_tau = s_, tau
        print(f"  τ={tau:.2f}: 총 {s_:.4f} | 1-NMAE {nm:.4f} | FICR {fi:.4f}{star}")
    print(f"  → 사후 최적 고정분위 τ*={best_tau:.2f} (총 {best_s:.4f})")
    print(f"\n[편향 진단]")
    print(f"  우리 g_ours implied quantile: 평균 {iq.mean():.3f} 중앙 {np.median(iq):.3f} "
          f"(5%~95%: {np.percentile(iq,5):.3f}~{np.percentile(iq,95):.3f})")
    print(f"  사후 최적 원자 분위(유효시간): 평균 {hq.mean():.3f} 중앙 {np.median(hq):.3f}")
    print(f"  → 우리가 겨누는 분위 vs 사후 최적 분위 격차: {hq.mean()-iq.mean():+.3f}")


if __name__ == "__main__":
    main()
