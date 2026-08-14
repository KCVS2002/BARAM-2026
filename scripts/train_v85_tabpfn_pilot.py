"""#132 v85: TabPFN sister 파일럿 — 이종 함수 클래스(사전학습 in-context)의 관문 검사.

방법론 후보: TabPFN v2 (Prior Labs License = Apache 2.0 + 저작자 표시, 상업적 이용
허용, 가중치 공개 2025-01 < 규정 2026-07-05, 로컬 로드 — 외부 API 아님).
GBDT 가족은 다양성 부재(#109, 상관 0.97+)였으나 TabPFN은 귀납 편향이 근본적으로
다른 베이지안류 in-context 학습기 + 예측 분포 전체 출력.

게이트 프로토콜 (v75와 동일): 2025 인접 fold 09/11 × g1/g2.
  관문 1: q50 오차 상관 < 0.9 (다양성 존재)
  관문 2: pinball 비율 (TabPFN/GBM) — CNN 전례상 ~1.1이면 풀링 생존권
  관문 3: 75원자 풀링 델타
컨텍스트: q3 가중 확률 비례 8k행 서브샘플 (TabPFN은 sample_weight 미지원).
캐시: {fold}_{tgt}_tpfqp.npz. 첫 (fold,tgt)에서 소요시간 출력 — 과도 시 중단 판단.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp_cache import load_base_frame
from src.decision import interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

CACHE = PROJECT / "experiments" / "oof_cache_q3"
K = 150
NA = {0: 125, 1: 150, 2: 175}
QS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CTX = 6000
PAIRS = [("2024-09", "kpx_group_1"), ("2024-09", "kpx_group_2"),
         ("2024-11", "kpx_group_1"), ("2024-11", "kpx_group_2")]


def pinball(y, qp, qs):
    tot = 0.0
    for j, q in enumerate(qs):
        d = y - qp[:, j]
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / len(qs)


def main() -> None:
    t0 = time.time()
    from tabpfn import TabPFNRegressor
    df, w_q3, cols, _ = load_base_frame()
    df["key"] = df.forecast_kst_dtm.astype("datetime64[ns]").astype("int64")
    print(f"frame ready, 피처 {len(cols)}개 ({time.time()-t0:.0f}s)", flush=True)

    for fold, tgt in PAIRS:
        cap = CAPACITY_KWH[tgt]
        out_f = CACHE / f"{fold}_{tgt}_tpfqp.npz"
        z = np.load(CACHE / f"{fold}_{tgt}.npz")
        dtm = z["dtm"]
        actual, a_bar = z["actual"], float(z["a_bar"])
        qp_gbm, anen200, dist = z["qp"], z["anen"], z["dist"]

        if not out_f.exists():
            va_start = pd.Timestamp(int(fold[:4]), int(fold[5:7]), 1, 1)
            trm = (df.forecast_kst_dtm < va_start) & df[tgt].notna()
            tr = df[trm]
            w = w_q3[tgt].to_numpy()[trm.to_numpy()]
            Xtr_full = tr[cols].to_numpy(dtype=np.float32)
            ytr_full = (tr[tgt] / cap).to_numpy(dtype=np.float32)
            rng = np.random.default_rng(42)
            p = w / w.sum()
            idx = rng.choice(len(tr), size=min(CTX, len(tr)), replace=False, p=p)
            Xtr, ytr = Xtr_full[idx], ytr_full[idx]
            va_rows = df.set_index("key").loc[dtm]
            Xva = va_rows[cols].to_numpy(dtype=np.float32)
            print(f"[{fold} {tgt}] ctx {len(Xtr)} / query {len(Xva)} — 적합 시작 ({time.time()-t0:.0f}s)", flush=True)
            reg = TabPFNRegressor(n_estimators=2, ignore_pretraining_limits=True,
                                  random_state=42, device="auto",
                                  memory_saving_mode=True)
            reg.fit(Xtr, ytr)
            print(f"[{fold} {tgt}] fit 완료, 추론 시작 ({time.time()-t0:.0f}s)", flush=True)
            qp_t = np.empty((len(Xva), len(QS)), dtype=np.float64)
            B = 128
            for s in range(0, len(Xva), B):
                res = reg.predict(Xva[s:s + B], output_type="quantiles", quantiles=QS)
                qp_t[s:s + B] = np.column_stack(res) if isinstance(res, list) else res
                if s == 0:
                    print(f"[{fold} {tgt}] 첫 배치 {B}행 완료 ({time.time()-t0:.0f}s)", flush=True)
            qp_t = np.clip(np.sort(qp_t, axis=1), 0, 1) * cap
            qp_t = np.sort(smooth_quantiles_by_day(qp_t, pd.Series(pd.to_datetime(dtm))), axis=1)
            np.savez_compressed(out_f, qp=qp_t)
            print(f"[{fold} {tgt}] 캐시 저장 ({time.time()-t0:.0f}s)", flush=True)
        qp_t = np.load(out_f)["qp"]

        # 관문 1·2
        corr = np.corrcoef(qp_t[:, 9] - actual, qp_gbm[:, 9] - actual)[0, 1]
        pb_t = pinball(actual, qp_t, QS)
        pb_g = pinball(actual, qp_gbm, QS)
        # 관문 3: 75원자 풀링 (공식 cur75 위에 추가)
        n_row = len(qp_gbm)
        gbm300 = interp_atoms(qp_gbm, n=2 * K)
        med = np.median(gbm300, axis=1)
        d_bar = dist[:, :K].mean(axis=1)
        terc = np.searchsorted(np.quantile(d_bar, [1 / 3, 2 / 3]), d_bar)
        qcur = np.load(CACHE / f"{fold}_{tgt}_cnnqp.npz")["qp"]
        cnn75 = interp_atoms(qcur, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
        tpf75 = interp_atoms(qp_t, n=2 * K)[:, np.linspace(0, 2 * K - 1, 75).astype(int)]
        scores = {}
        for v, extra in (("base", None), ("tpf75", tpf75)):
            atoms_l = []
            for i in range(n_row):
                na = NA[terc[i]]
                ng = 2 * K - na
                an = np.interp(np.linspace(0, 199, na), np.arange(200), anen200[i])
                an = np.clip(an + (med[i] - np.median(an)), 0, cap)
                gb = np.interp(np.linspace(0, 2 * K - 1, ng), np.arange(2 * K), gbm300[i])
                parts = [an, gb, cnn75[i]] + ([extra[i]] if extra is not None else [])
                atoms_l.append(np.sort(np.concatenate(parts)))
            pred = optimize_submission(np.array(atoms_l), cap, a_bar)
            scores[v], nm, fi, _ = metric_single(actual, pred, cap)
        cnn_corr = np.corrcoef(qp_t[:, 9] - actual, qcur[:, 9] - actual)[0, 1]
        print(f"== {fold} {tgt}: GBM상관 {corr:.3f} | CNN상관 {cnn_corr:.3f} | "
              f"pinball비율 {pb_t/pb_g:.3f} | 풀링 {scores['base']:.4f}→{scores['tpf75']:.4f} "
              f"({scores['tpf75']-scores['base']:+.4f}) ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
